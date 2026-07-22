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

    GenerationResult result;

    // -- prompt processing -------------------------------------------------
    auto t0 = std::chrono::steady_clock::now();
    std::vector<llama_token> prompt_tokens = tokenize(vocab, prompt, /*add_special=*/true);
    result.n_prompt_tokens = static_cast<int>(prompt_tokens.size());

    llama_batch prompt_batch = llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
    if (llama_decode(ctx, prompt_batch) != 0) {
        throw std::runtime_error("pleiades_engine: llama_decode failed on prompt");
    }
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
            throw std::runtime_error("pleiades_engine: llama_decode failed during generation");
        }
    }
    auto t2 = std::chrono::steady_clock::now();
    result.generate_seconds = std::chrono::duration<double>(t2 - t1).count();
    result.n_generated_tokens = generated;
    result.text = std::move(text);

    llama_sampler_free(chain);
    return result;
}

}  // namespace pleiades_engine
