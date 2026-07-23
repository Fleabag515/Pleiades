#include "pleiades_engine/engine.h"

#include <chrono>
#include <stdexcept>
#include <vector>

namespace pleiades_engine {

namespace {

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text, bool add_special) {
    std::vector<llama_token> tokens(text.size() + 8);
    int n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(),
                            static_cast<int32_t>(tokens.size()), add_special, /*parse_special=*/true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(),
                            static_cast<int32_t>(tokens.size()), add_special, /*parse_special=*/true);
        if (n < 0) {
            throw std::runtime_error("pleiades_engine: tokenization failed");
        }
    }
    tokens.resize(n);
    return tokens;
}

std::string token_to_piece(const llama_vocab* vocab, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), /*lstrip=*/0, /*special=*/false);
    if (n < 0) {
        // Piece didn't fit -- extremely unlikely for a single token, but
        // handle it rather than silently truncating.
        std::vector<char> big(-n);
        n = llama_token_to_piece(vocab, token, big.data(), static_cast<int32_t>(big.size()), 0, false);
        return std::string(big.data(), n > 0 ? n : 0);
    }
    return std::string(buf, n);
}

}  // namespace

Engine::Engine(ModelManager& models, ContextGovernor& ctx) : models_(models), ctx_(ctx) {}

void Engine::reset_prefix_cache() {
    llama_context* ctx = ctx_.ctx();
    if (ctx) {
        // Hard-clear sequence 0's KV so the tracked cache and the real KV
        // agree (both empty).
        llama_memory_seq_rm(llama_get_memory(ctx), /*seq_id=*/0, /*p0=*/-1, /*p1=*/-1);
    }
    prefix_.invalidate(ctx_.epoch());
}

GenerationResult Engine::complete(const std::string& prompt, int n_predict) {
    return generate(prompt, n_predict, nullptr);
}

GenerationResult Engine::generate(const std::string& prompt, int n_predict,
                                   const std::function<bool(const std::string&)>& on_token) {
    if (!models_.is_loaded()) {
        throw std::runtime_error("pleiades_engine: no model loaded");
    }
    const llama_vocab* vocab = llama_model_get_vocab(models_.model());
    llama_context* ctx = ctx_.ctx();
    llama_memory_t mem = llama_get_memory(ctx);

    GenerationResult result;

    // -- prompt processing (Phase 6 prefix cache) --------------------------
    //
    // The pre-Phase-6 path decoded the ENTIRE prompt every call with no KV
    // bookkeeping, which -- because llama_batch_get_one() leaves batch.pos
    // == nullptr and llama.cpp then appends at memory->seq_pos_max(seq)+1 --
    // duplicated any repeated prefix into the KV at growing positions. Here
    // we instead: (1) drop the cache if the context was recreated under us
    // (resize bumps the governor epoch and empties the KV), (2) find the
    // longest prompt prefix already resident, (3) trim the KV back to just
    // that prefix, and (4) decode only the new suffix.
    auto t0 = std::chrono::steady_clock::now();

    if (prefix_.epoch() != ctx_.epoch()) {
        // Context was (re)created since we last decoded -- KV is empty.
        prefix_.invalidate(ctx_.epoch());
    }

    std::vector<llama_token> prompt_tokens = tokenize(vocab, prompt, /*add_special=*/true);
    result.n_prompt_tokens = static_cast<int>(prompt_tokens.size());

    size_t n_reuse = prefix_.reusable_prefix(prompt_tokens);

    // Guard against a KV that no longer actually holds a clean [0, n_reuse)
    // prefix -- e.g. a sliding-window (SWA) cache can evict early positions,
    // in which case seq_pos_min > 0 and the "reused" prefix would be a lie.
    // Only trust the cache when position 0 is still present. (For the dense
    // caches this engine serves today this is always true; the check is
    // cheap insurance, not dead code -- it's what makes reuse correct rather
    // than merely usually-correct.)
    if (n_reuse > 0) {
        llama_pos pos_min = llama_memory_seq_pos_min(mem, /*seq_id=*/0);
        llama_pos pos_max = llama_memory_seq_pos_max(mem, /*seq_id=*/0);
        if (pos_min > 0 || pos_max < static_cast<llama_pos>(n_reuse) - 1) {
            n_reuse = 0;
        }
    }

    // Trim the KV back to the reusable prefix: remove [n_reuse, inf) from
    // sequence 0. When n_reuse == 0 this is exactly the KV clear the old
    // code was missing (the active-duplication bug) -- it resets seq 0 to
    // empty so the prompt decodes at positions [0, ...).
    if (!llama_memory_seq_rm(mem, /*seq_id=*/0, static_cast<llama_pos>(n_reuse), /*p1=*/-1)) {
        // Partial removal is refused by some cache types (e.g. SWA can't
        // truncate mid-sequence). Fall back to a full clear + cold decode
        // -- correct, just forfeits reuse for this call.
        llama_memory_seq_rm(mem, /*seq_id=*/0, /*p0=*/-1, /*p1=*/-1);
        n_reuse = 0;
    }

    result.n_prompt_cached = static_cast<int>(n_reuse);

    // Decode only the suffix beyond the reused prefix. reusable_prefix()
    // guarantees n_reuse <= prompt.size()-1, so there is always >= 1 token
    // to decode -- the sampler needs fresh logits for the final prompt
    // token. Positions auto-assign to [n_reuse, ...] since we trimmed the
    // KV to seq_pos_max == n_reuse-1.
    int n_suffix = static_cast<int>(prompt_tokens.size() - n_reuse);
    llama_batch prompt_batch = llama_batch_get_one(prompt_tokens.data() + n_reuse, n_suffix);
    if (llama_decode(ctx, prompt_batch) != 0) {
        // Leave the cache consistent with the (now-uncertain) KV state.
        prefix_.invalidate(ctx_.epoch());
        throw std::runtime_error("pleiades_engine: llama_decode failed on prompt");
    }

    // The resident sequence is now exactly this prompt's tokens.
    prefix_.set(prompt_tokens, ctx_.epoch());

    auto t1 = std::chrono::steady_clock::now();
    result.prompt_seconds = std::chrono::duration<double>(t1 - t0).count();

    // -- greedy generation --------------------------------------------------
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());

    std::string text;
    int generated = 0;
    for (; generated < n_predict; ++generated) {
        llama_token next = llama_sampler_sample(chain, ctx, -1);
        if (llama_vocab_is_eog(vocab, next)) {
            break;
        }
        std::string piece = token_to_piece(vocab, next);
        text += piece;
        bool keep_going = on_token ? on_token(piece) : true;
        if (!keep_going) {
            ++generated;  // count the token we just emitted before stopping
            break;
        }

        llama_batch next_batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, next_batch) != 0) {
            llama_sampler_free(chain);
            prefix_.invalidate(ctx_.epoch());
            throw std::runtime_error("pleiades_engine: llama_decode failed during generation");
        }
        // This token is now resident in the KV -- keep the cache mirroring
        // the KV exactly so the next request can reuse it too.
        prefix_.append(next);
    }
    auto t2 = std::chrono::steady_clock::now();
    result.generate_seconds = std::chrono::duration<double>(t2 - t1).count();
    result.n_generated_tokens = generated;
    result.text = std::move(text);

    llama_sampler_free(chain);
    return result;
}

}  // namespace pleiades_engine
