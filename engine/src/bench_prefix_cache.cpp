// Phase 6 benchmark: cold full-prompt decode vs. warm prefix-cache-hit
// decode. See docs/specs/2026-07-21-native-inference-engine-design.md,
// Phase 6. Mirrors bench_resize.cpp's shape: model load -> context create
// -> timed operations -> sanity output.
//
// Usage: pleiades-engine-bench-prefix <model.gguf> [n_gpu_layers] [prefix_repeat]
//
// Builds a long "system/persona" style prefix (the stable block Anamnesis
// resends every turn), then measures three things on the SAME long-lived
// Engine, exactly as the real server reuses one Engine across requests:
//
//   COLD  : first request on a fresh context -- whole prompt decoded.
//   WARM  : identical prompt again -- almost the entire prompt served from
//           the KV prefix cache (all but the final token), so prompt
//           processing should collapse to ~one token of decode.
//   TURN2 : the realistic case -- same long prefix + one appended user
//           message; the persona block is reused, only the new suffix is
//           decoded.
//
// It also asserts WARM/TURN2 output is byte-identical to a truly-cold decode
// of the same prompt (a second, throwaway Engine on a fresh context), since
// a benchmark that "wins" by producing different (wrong) output is worthless.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "llama.h"
#include "pleiades_engine/context_governor.h"
#include "pleiades_engine/engine.h"
#include "pleiades_engine/model_manager.h"

namespace {
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// A cold decode of `prompt` on its own fresh context, for the byte-identical
// correctness cross-check. Returns the generated text.
std::string cold_reference(pleiades_engine::ModelManager& models, int n_ctx, const std::string& prompt,
                            int n_predict) {
    pleiades_engine::ContextGovernor ctx;
    ctx.create(models.model(), n_ctx);
    pleiades_engine::Engine engine(models, ctx);
    return engine.complete(prompt, n_predict).text;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [n_gpu_layers] [prefix_repeat]\n", argv[0]);
        return 2;
    }
    std::string model_path = argv[1];
    int n_gpu_layers = argc > 2 ? std::atoi(argv[2]) : 0;
    int prefix_repeat = argc > 3 ? std::atoi(argv[3]) : 40;
    const int n_ctx = 8192;
    const int n_predict = 24;

    // A long, stable "persona/system" prefix -- the exact shape Anamnesis
    // resends verbatim every turn. Repeated sentences just pad it out to a
    // realistic multi-hundred-token block cheaply.
    std::string persona =
        "You are Mark, a careful and precise assistant with a long memory. "
        "You always ground your answers in the facts you have been given. ";
    std::string prefix;
    for (int i = 0; i < prefix_repeat; ++i) {
        prefix += persona;
    }
    const std::string cold_prompt = prefix + "\nUser: What is the capital of France?\nAssistant:";
    const std::string turn2_prompt =
        prefix + "\nUser: What is the capital of France?\nAssistant: Paris.\nUser: And of Japan?\nAssistant:";

    llama_backend_init();
    try {
        auto t_load0 = std::chrono::steady_clock::now();
        pleiades_engine::ModelManager models;
        models.load(model_path, n_gpu_layers);
        std::printf("model load:                %.1f ms (ngl=%d)\n", ms_since(t_load0), n_gpu_layers);

        pleiades_engine::ContextGovernor ctx;
        ctx.create(models.model(), n_ctx);
        pleiades_engine::Engine engine(models, ctx);

        // COLD ------------------------------------------------------------
        auto tc = std::chrono::steady_clock::now();
        auto cold = engine.complete(cold_prompt, n_predict);
        double cold_ms = ms_since(tc);
        std::printf("COLD  : prompt_tok=%d cached=%d prompt_ms=%.1f gen_ms=%.1f total=%.1f ms\n",
                    cold.n_prompt_tokens, cold.n_prompt_cached, cold.prompt_seconds * 1000.0,
                    cold.generate_seconds * 1000.0, cold_ms);

        // WARM (identical prompt) -----------------------------------------
        auto tw = std::chrono::steady_clock::now();
        auto warm = engine.complete(cold_prompt, n_predict);
        double warm_ms = ms_since(tw);
        std::printf("WARM  : prompt_tok=%d cached=%d prompt_ms=%.1f gen_ms=%.1f total=%.1f ms\n",
                    warm.n_prompt_tokens, warm.n_prompt_cached, warm.prompt_seconds * 1000.0,
                    warm.generate_seconds * 1000.0, warm_ms);

        // TURN2 (shared prefix + new suffix) ------------------------------
        auto tt = std::chrono::steady_clock::now();
        auto turn2 = engine.complete(turn2_prompt, n_predict);
        double turn2_ms = ms_since(tt);
        std::printf("TURN2 : prompt_tok=%d cached=%d prompt_ms=%.1f gen_ms=%.1f total=%.1f ms\n",
                    turn2.n_prompt_tokens, turn2.n_prompt_cached, turn2.prompt_seconds * 1000.0,
                    turn2.generate_seconds * 1000.0, turn2_ms);

        double prompt_speedup = warm.prompt_seconds > 0 ? cold.prompt_seconds / warm.prompt_seconds : 0.0;
        std::printf("\nprompt-processing speedup (cold/warm): %.1fx  (%.1f ms -> %.1f ms)\n",
                    prompt_speedup, cold.prompt_seconds * 1000.0, warm.prompt_seconds * 1000.0);

        // Correctness cross-check: warm output must equal a truly-cold
        // decode of the same prompt on a fresh context.
        std::string warm_ref = cold_reference(models, n_ctx, cold_prompt, n_predict);
        std::string turn2_ref = cold_reference(models, n_ctx, turn2_prompt, n_predict);
        bool warm_ok = (warm.text == warm_ref) && (warm.text == cold.text);
        bool turn2_ok = (turn2.text == turn2_ref);
        std::printf("byte-identical(warm  vs cold ref): %s\n", warm_ok ? "YES" : "NO");
        std::printf("byte-identical(turn2 vs cold ref): %s\n", turn2_ok ? "YES" : "NO");
        std::printf("--- warm text ---\n%s\n", warm.text.c_str());
        if (!warm_ok || !turn2_ok) {
            std::fprintf(stderr, "CORRECTNESS FAILURE: prefix-cache output diverged from cold decode\n");
            llama_backend_free();
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        llama_backend_free();
        return 1;
    }
    llama_backend_free();
    return 0;
}
