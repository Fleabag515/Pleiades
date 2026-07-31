#include "pleiades_engine/engine.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "mtmd-helper.h"

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

// Builds the sampler chain for one generation from a SamplingParams. Ordering
// mirrors llama.cpp's own canonical default chain (common/sampling.cpp):
// penalties first, then the truncation samplers (top_k -> typical -> top_p ->
// min_p), then temperature, then the seeded final distribution sampler. Each
// stage is only added when it would actually do something, so a
// default-constructed SamplingParams (temperature 0) collapses to exactly the
// single greedy sampler engine.cpp built before this function existed --
// keeping the deterministic bench/test paths byte-identical.
//
// temperature <= 0 is llama.cpp's own "greedy" convention (llama_sampler_init_temp's
// own doc: t <= 0 keeps the max logit and -inf's the rest); we short-circuit
// to a pure argmax greedy sampler so temp==0 requests are deterministic
// regardless of the other knobs, matching llama-cpp-python.
llama_sampler* build_sampler_chain(const SamplingParams& sp) {
    llama_sampler_chain_params cparams = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(cparams);

    if (sp.temperature <= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        return chain;
    }

    if (sp.repeat_penalty != 1.0f || sp.frequency_penalty != 0.0f || sp.presence_penalty != 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(sp.penalty_last_n, sp.repeat_penalty,
                                                                     sp.frequency_penalty, sp.presence_penalty));
    }
    if (sp.top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(sp.top_k));
    }
    if (sp.typical_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_typical(sp.typical_p, /*min_keep=*/1));
    }
    if (sp.top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(sp.top_p, /*min_keep=*/1));
    }
    if (sp.min_p > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(sp.min_p, /*min_keep=*/1));
    }
    llama_sampler_chain_add(chain, llama_sampler_init_temp(sp.temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(sp.seed));
    return chain;
}

// Earliest byte offset at which any complete stop string occurs in `text`, or
// std::string::npos if none does.
size_t earliest_stop_pos(const std::string& text, const std::vector<std::string>& stops) {
    size_t best = std::string::npos;
    for (const std::string& s : stops) {
        if (s.empty()) {
            continue;
        }
        size_t p = text.find(s);
        if (p != std::string::npos && p < best) {
            best = p;
        }
    }
    return best;
}

// Length of the longest suffix of `text` that is a PROPER prefix of some stop
// string -- i.e. the tail that must be held back from a stream because the
// next token could still complete a stop match. 0 when no such overlap exists
// (the whole text is safe to emit). Never returns a length that would match a
// COMPLETE stop (that case is handled by earliest_stop_pos before this is
// consulted), so at most len(stop)-1 bytes are ever withheld.
size_t stop_overlap_suffix(const std::string& text, const std::vector<std::string>& stops) {
    size_t held = 0;
    for (const std::string& s : stops) {
        size_t max_l = std::min(s.size() > 0 ? s.size() - 1 : 0, text.size());
        for (size_t l = max_l; l > held; --l) {
            if (text.compare(text.size() - l, l, s, 0, l) == 0) {
                held = l;
                break;
            }
        }
    }
    return held;
}

}  // namespace

Engine::Engine(ModelManager& models, ContextGovernor& ctx) : models_(models), ctx_(ctx) {}

void Engine::reset_resident_map() {
    llama_context* ctx = ctx_.ctx();
    if (ctx) {
        // Hard-clear the KV so the tracked cache and the real KV agree (both
        // empty). data=true zeroes the backend buffers, not just the cell
        // metadata -- required so a subsequent flash-attention decode can't
        // read leftover K/V from these now-freed cells (see the stale-KV guard
        // in generate() for the full rationale).
        llama_memory_clear(llama_get_memory(ctx), /*data=*/true);
    }
    resident_.invalidate(ctx_.epoch());
}

void Engine::seed_resident_map(std::vector<llama_token> tokens) {
    // Phase 9.4: called by the HTTP shim right after a successful
    // POST /slots/0?action=restore -- llama_state_seq_load_file() repopulated
    // the live KV directly (positions baked into the saved state, nothing for
    // us to recompute), but ResidentMap's own bookkeeping has no way to know
    // that happened. Left uncalled, the next request's reusable_prefix()
    // would compute against an empty tracked sequence and cold-decode the
    // WHOLE prompt on top of KV that already holds it -- duplicating content
    // exactly like the bug Phase 6's PrefixCache (now ResidentMap) was built
    // to fix in the first place, just reintroduced via a different code
    // path. epoch() is current (a restore never resizes), so tag with it
    // rather than bumping.
    resident_.set(std::move(tokens), ctx_.epoch());
}

GenerationResult Engine::complete(const std::string& prompt, int n_predict, const SamplingParams& sampling) {
    return generate(prompt, n_predict, nullptr, sampling);
}

CompactionResult Engine::compact_resident_span(size_t evict_start, size_t evict_end) {
    llama_context* ctx = ctx_.ctx();
    if (!ctx) {
        return {false, "no live llama_context"};
    }
    // Same stale-resident_ guard generate() applies at the top of its own
    // request path (see below) -- without it, calling this in the window
    // between a resize (which bumps the epoch and empties the KV) and the
    // next generate() call would run compact() against a resident_ that
    // still reports its PRE-resize length while the live KV is actually
    // empty, tripping compact()'s desync assertion on essentially every
    // call in that window instead of cleanly refusing on an out-of-range
    // span the way a genuinely stale-but-invalidated map would.
    if (resident_.epoch() != ctx_.epoch()) {
        resident_.invalidate(ctx_.epoch());
    }
    const bool fa_maybe_on = ctx_.params().flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    return pleiades_engine::compact(models_.model(), llama_get_memory(ctx), /*seq_id=*/0,
                                    resident_, evict_start, evict_end, fa_maybe_on);
}

GenerationResult Engine::generate(const std::string& prompt, int n_predict,
                                   const std::function<bool(const std::string&)>& on_token,
                                   const SamplingParams& sampling) {
    if (!models_.is_loaded()) {
        throw std::runtime_error("pleiades_engine: no model loaded");
    }
    const llama_vocab* vocab = llama_model_get_vocab(models_.model());
    llama_context* ctx = ctx_.ctx();
    if (!ctx) {
        // Only reachable when a resize failed AND its rollback failed (see
        // ContextGovernor::resize) -- fail as a clean error the HTTP layer
        // turns into a JSON 500, not a null deref inside llama_decode that
        // takes the whole server (and every conversation on it) down.
        throw std::runtime_error(
            "pleiades_engine: no live llama_context (a context resize failed and could not be "
            "rolled back; POST /resize to a smaller n_ctx to recover)");
    }
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

    if (resident_.epoch() != ctx_.epoch()) {
        // Context was (re)created since we last decoded -- KV is empty.
        resident_.invalidate(ctx_.epoch());
    }

    // High-water mark of the sequence currently resident in the KV -- the
    // number of contiguous positions [0, resident) that hold live K/V (the
    // previous request's prompt + everything it generated). resident_
    // mirrors the KV exactly (see ResidentMap), so this is just its length.
    // Captured before any trim below because the FA-safety check needs the
    // *pre-trim* resident length.
    const size_t resident = resident_.size();

    std::vector<llama_token> prompt_tokens = tokenize(vocab, prompt, /*add_special=*/true);
    result.n_prompt_tokens = static_cast<int>(prompt_tokens.size());

    size_t n_reuse = resident_.reusable_prefix(prompt_tokens);

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

    // -- Flash-attention stale-KV correctness guard ------------------------
    //
    // Trimming the KV with llama_memory_seq_rm() only clears cell *metadata*;
    // the K/V *data* of the freed cells stays in the backend buffer until a
    // later decode overwrites it. The suffix decode below overwrites cells
    // [n_reuse, prompt_len), so any freed cell at position >= prompt_len keeps
    // its stale K/V -- which happens exactly when this prompt is SHORTER than
    // the resident high-water (a shorter/diverged turn, the norm once Anamnesis
    // rewrites the memory block each turn; see launch.py's --cache-reuse note).
    //
    // With the explicit-mask attention path those stale cells are correctly
    // masked (mask_drop == -inf), so reuse is exact -- verified byte-for-byte
    // on CPU and CUDA. But this pinned llama.cpp's FLASH-ATTENTION kernel
    // (b10103 / c588c4f47) leaks non-zero K/V from those masked cells into the
    // result: a small numerical perturbation that flips the occasional greedy
    // argmax and silently corrupts generation (empirically: a Qwen2.5 tool call
    // comes back as `{{"name"..}}}}` instead of `{"name"..}` on the 2nd of two
    // identical requests -- reproduced on CPU *and* CUDA, so it is a flash-
    // attention bug, not a GPU one; it is not the position/KV bookkeeping,
    // which is provably correct here, nor CUDA graphs). llama_batch_get_one's
    // auto-assigned positions were ruled out too (explicit positions don't fix
    // it). The tiny fixture in test_prefix_cache doesn't trip it only because
    // its perturbation stays under the argmax-flip threshold.
    //
    // Fix: when flash attention may be active and stale cells would remain,
    // scrub the whole KV data buffer (llama_memory_clear(data=true) is the only
    // public API that zeroes data, not just metadata) and cold-decode. Reuse is
    // preserved for the common growing-conversation case (prompt_len >=
    // resident: every freed cell is overwritten by the suffix decode, so no
    // stale data survives) and whenever flash attention is explicitly off.
    const bool fa_maybe_on = ctx_.params().flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (fa_maybe_on && prompt_tokens.size() < resident) {
        llama_memory_clear(mem, /*data=*/true);
        resident_.invalidate(ctx_.epoch());
        n_reuse = 0;
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
        // Scrub the KV data (not just metadata) so a partially-written,
        // now-abandoned decode can't leak into the next request's flash-
        // attention pass -- then drop the tracked prefix to match.
        llama_memory_clear(mem, /*data=*/true);
        resident_.invalidate(ctx_.epoch());
        throw std::runtime_error("pleiades_engine: llama_decode failed on prompt");
    }

    // The resident sequence is now exactly this prompt's tokens.
    resident_.set(prompt_tokens, ctx_.epoch());

    auto t1 = std::chrono::steady_clock::now();
    double prompt_seconds = std::chrono::duration<double>(t1 - t0).count();

    return run_sampling_loop(n_predict, on_token, sampling, prompt_seconds,
                             static_cast<int>(prompt_tokens.size()), static_cast<int>(n_reuse),
                             /*track_resident_map=*/true);
}

GenerationResult Engine::complete_multimodal(const mtmd_input_chunks* chunks, int n_predict,
                                             const SamplingParams& sampling) {
    return generate_multimodal(chunks, n_predict, nullptr, sampling);
}

GenerationResult Engine::generate_multimodal(const mtmd_input_chunks* chunks, int n_predict,
                                             const std::function<bool(const std::string&)>& on_token,
                                             const SamplingParams& sampling) {
    if (!models_.is_loaded()) {
        throw std::runtime_error("pleiades_engine: no model loaded");
    }
    if (!models_.mtmd()) {
        throw std::runtime_error("pleiades_engine: no mmproj loaded (vision not enabled for this model -- "
                                 "start the server with --mmproj)");
    }
    llama_context* ctx = ctx_.ctx();
    if (!ctx) {
        throw std::runtime_error(
            "pleiades_engine: no live llama_context (a context resize failed and could not be "
            "rolled back; POST /resize to a smaller n_ctx to recover)");
    }
    llama_memory_t mem = llama_get_memory(ctx);

    auto t0 = std::chrono::steady_clock::now();

    // Always a cache-busting cold decode -- see this method's own header
    // comment (engine.h) for why ResidentMap cannot honestly represent an
    // image-spanned KV region. Both the incoming and outgoing cache state are
    // empty regardless of what the previous request left behind.
    llama_memory_clear(mem, /*data=*/true);
    resident_.invalidate(ctx_.epoch());

    llama_pos new_n_past = 0;
    int32_t rc = mtmd_helper_eval_chunks(models_.mtmd(), ctx, chunks, /*n_past=*/0, /*seq_id=*/0,
                                         ctx_.params().n_batch, /*logits_last=*/true, &new_n_past);
    if (rc != 0) {
        llama_memory_clear(mem, /*data=*/true);
        resident_.invalidate(ctx_.epoch());
        throw std::runtime_error("pleiades_engine: mtmd_helper_eval_chunks failed (rc=" + std::to_string(rc) + ")");
    }

    auto t1 = std::chrono::steady_clock::now();
    double prompt_seconds = std::chrono::duration<double>(t1 - t0).count();
    int n_prompt_tokens = static_cast<int>(mtmd_helper_get_n_tokens(chunks));

    return run_sampling_loop(n_predict, on_token, sampling, prompt_seconds, n_prompt_tokens,
                             /*n_prompt_cached=*/0, /*track_resident_map=*/false);
}

GenerationResult Engine::run_sampling_loop(int n_predict, const std::function<bool(const std::string&)>& on_token,
                                           const SamplingParams& sampling, double prompt_seconds,
                                           int n_prompt_tokens, int n_prompt_cached, bool track_resident_map) {
    llama_context* ctx = ctx_.ctx();
    const llama_vocab* vocab = llama_model_get_vocab(models_.model());
    llama_memory_t mem = llama_get_memory(ctx);

    GenerationResult result;
    result.n_prompt_tokens = n_prompt_tokens;
    result.n_prompt_cached = n_prompt_cached;
    result.prompt_seconds = prompt_seconds;
    auto t_gen_start = std::chrono::steady_clock::now();

    // Sampler chain per the request (greedy by default -- see
    // build_sampler_chain / SamplingParams). llama_sampler_sample() accepts
    // each sampled token into the chain, so the penalties sampler tracks the
    // generation window on its own; the prompt tokens are not fed to it (the
    // pre-sampling engine didn't either, and greedy needs no history).
    llama_sampler* chain = build_sampler_chain(sampling);

    const bool have_stops = !sampling.stop.empty();
    std::string text;      // full generated text (already stop-truncated once a stop hits)
    size_t emitted = 0;    // bytes of `text` handed to on_token so far (stop path only)
    int generated = 0;
    bool stopped_on_stop = false;
    bool client_cancelled = false;

    for (; generated < n_predict; ++generated) {
        llama_token next = llama_sampler_sample(chain, ctx, -1);
        if (llama_vocab_is_eog(vocab, next)) {
            break;
        }
        std::string piece = token_to_piece(vocab, next);
        text += piece;

        // Stop-string detection: if a complete stop string is now present,
        // trim the output at its first occurrence and halt (the stop text
        // itself is never surfaced -- matches llama-cpp-python).
        if (have_stops) {
            size_t pos = earliest_stop_pos(text, sampling.stop);
            if (pos != std::string::npos) {
                text.erase(pos);
                stopped_on_stop = true;
            }
        }

        // Emission. With no stop strings this is byte-for-byte the old
        // behavior: one on_token(piece) per token. With stop strings we hold
        // back any tail that could still grow into a stop match, so the stop
        // text is never leaked mid-stream.
        bool keep_going = true;
        if (on_token) {
            if (!have_stops) {
                keep_going = on_token(piece);
            } else {
                size_t safe = stopped_on_stop ? text.size()
                                              : text.size() - stop_overlap_suffix(text, sampling.stop);
                if (safe > emitted) {
                    keep_going = on_token(text.substr(emitted, safe - emitted));
                    emitted = safe;
                }
            }
        }

        if (stopped_on_stop) {
            ++generated;  // count the token that completed the stop match
            break;
        }
        if (!keep_going) {
            ++generated;  // count the token we just emitted before stopping
            client_cancelled = true;
            break;
        }

        llama_batch next_batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, next_batch) != 0) {
            llama_sampler_free(chain);
            llama_memory_clear(mem, /*data=*/true);  // scrub partial KV (see prompt-decode failure above)
            resident_.invalidate(ctx_.epoch());
            throw std::runtime_error("pleiades_engine: llama_decode failed during generation");
        }
        // This token is now resident in the KV. Text-path: keep the cache
        // mirroring the KV exactly so the next request can reuse it too.
        // Multimodal-path: never append -- see generate_multimodal's comment
        // on why resident_ stays empty for the entire duration of that call.
        if (track_resident_map) {
            resident_.append(next);
        }
    }

    // Flush any held-back (but safe) tail on a clean end -- EOG or n_predict
    // reached without a stop match. Skipped when a stop halted us (the tail
    // was the stop text, already trimmed) or the client cancelled.
    if (on_token && have_stops && !stopped_on_stop && !client_cancelled && emitted < text.size()) {
        on_token(text.substr(emitted));
    }

    auto t_gen_end = std::chrono::steady_clock::now();
    result.generate_seconds = std::chrono::duration<double>(t_gen_end - t_gen_start).count();
    result.n_generated_tokens = generated;
    result.stopped_on_stop_string = stopped_on_stop;
    result.text = std::move(text);

    llama_sampler_free(chain);
    return result;
}

}  // namespace pleiades_engine
