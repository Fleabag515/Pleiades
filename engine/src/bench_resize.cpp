// Phase 2 benchmark: ContextGovernor::resize() wall-clock latency, to
// compare against the Python elastic engine's EngineState.resize() and
// against native llama-server's only option today (full process restart).
// See docs/specs/2026-07-21-native-inference-engine-design.md, Phase 2.
//
// Usage: pleiades-engine-bench-resize <model.gguf> <n_ctx_from> <n_ctx_to>
//
// Runs a short sanity completion at n_ctx_from, times the resize to
// n_ctx_to, then runs the same completion again to confirm the resized
// context still produces sane output (basic regression, not a benchmark-
// grade eval per the Phase 2 anti-pattern note: no llama_state_* save/
// restore here, resize is a clean free+recreate).
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
}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <n_ctx_from> <n_ctx_to> [n_gpu_layers]\n", argv[0]);
        return 2;
    }
    std::string model_path = argv[1];
    int n_ctx_from = std::atoi(argv[2]);
    int n_ctx_to = std::atoi(argv[3]);
    int n_gpu_layers = argc > 4 ? std::atoi(argv[4]) : 0;
    const std::string prompt = "The capital of France is";

    llama_backend_init();

    try {
        auto t_load0 = std::chrono::steady_clock::now();
        pleiades_engine::ModelManager models;
        // n_gpu_layers is caller-supplied (default 0/CPU-only) so this can be
        // run either way depending on what VRAM headroom is available at the
        // time -- keep it consistent with whatever the other two legs of the
        // Phase 2 comparison (Python engine resize, llama-server restart) use.
        models.load(model_path, n_gpu_layers);
        std::printf("model load:        %.1f ms\n", ms_since(t_load0));

        pleiades_engine::ContextGovernor ctx;
        auto t_create0 = std::chrono::steady_clock::now();
        ctx.create(models.model(), n_ctx_from);
        std::printf("context create (n_ctx=%d): %.1f ms\n", n_ctx_from, ms_since(t_create0));

        pleiades_engine::Engine engine(models, ctx);
        auto pre = engine.complete(prompt, 24);
        std::printf("--- pre-resize completion (n_ctx=%d) ---\n%s\n", ctx.n_ctx(), pre.text.c_str());

        auto t_resize0 = std::chrono::steady_clock::now();
        int actual = ctx.resize(n_ctx_to);
        double resize_ms = ms_since(t_resize0);
        std::printf("RESIZE %d -> %d took %.1f ms\n", n_ctx_from, actual, resize_ms);

        auto post = engine.complete(prompt, 24);
        std::printf("--- post-resize completion (n_ctx=%d) ---\n%s\n", ctx.n_ctx(), post.text.c_str());

    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        llama_backend_free();
        return 1;
    }

    llama_backend_free();
    return 0;
}
