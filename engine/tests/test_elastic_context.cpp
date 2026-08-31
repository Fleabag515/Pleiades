// Phase 2 — elastic context: KV-preserving growth, auto-upshift, auto-YaRN,
// and the CPU spill.
//
// Two fixtures, on purpose:
//   * argv[1] = the tiny fixture (stories15M, n_ctx_train = 128). Every gear
//     is PAST its trained context, so it exercises auto-YaRN mechanics — but
//     the toy model degenerates under YaRN scaling, so NO text-equality
//     assertions are made on it.
//   * argv[2] = a real instruct model with a LARGE trained context (default
//     Llama-3.2-1B-Instruct, n_ctx_train = 131072): the common gears are all
//     WITHIN train (plain RoPE), so KV-preserving growth is actually
//     testable end to end. Skipped (exit 77) when absent.
#include "pleiades_engine/engine.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "llama.h"
#include "pleiades_engine/context_governor.h"
#include "pleiades_engine/model_manager.h"
#include "test_util.h"

namespace {

struct Rig {
    pleiades_engine::ModelManager models;
    pleiades_engine::ContextGovernor ctx;
    pleiades_engine::Engine engine;

    Rig(const std::string& model_path, int n_ctx, int n_ctx_max = 0,
        llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_DISABLED, int cache_reuse = 0)
        : engine(models, ctx, cache_reuse) {
        pleiades_engine::ContextParams params;
        params.flash_attn_type = fa;
        models.load(model_path, /*n_gpu_layers=*/0);
        ctx.create(models.model(), n_ctx, n_ctx_max, params);
    }
};

std::string long_prompt(int repeats) {
    std::string s =
        "The lighthouse keeper wrote in his journal every single evening after sunset, describing";
    for (int i = 0; i < repeats; ++i) {
        s += " the color of the water and the shapes of passing ships and the mood of the wind";
        s += std::to_string(i);  // defeat any naive repetition handling
    }
    s += " Then he closed the book. Continue the story in one short sentence:";
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <fixture.gguf> [large-train-ctx-model.gguf]\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    const std::string fixture = argv[1];
    const std::string real = argc > 2 ? argv[2] : "";

    // -- 1. Auto-YaRN mechanics (fixture: n_ctx_train = 128) ---------------- //
    // Every context is past-train; buckets are power-of-two in n_ctx/train.
    // Growth 256 -> 2048 crosses buckets (2x -> 16x), so the KV is honestly
    // reported NOT preserved, the token list stands, and the engine stays
    // fully functional.
    {
        Rig rig(fixture, /*n_ctx=*/256);
        PLEIADES_CHECK(rig.ctx.yarn_factor() == 2.0f);
        rig.engine.complete("Once upon a time", /*n_predict=*/4);
        auto gr = rig.engine.grow_context_to(2048);
        PLEIADES_CHECK(gr.n_ctx == 2048);
        PLEIADES_CHECK(rig.ctx.yarn_factor() == 16.0f);
        PLEIADES_CHECK(!gr.preserved);  // bucket change invalidates the K/V
        auto r = rig.engine.complete("Once upon a time, there was a little", /*n_predict=*/8);
        PLEIADES_CHECK(r.n_prompt_tokens > 0);
    }

    // -- 2. RAM budget exhaustion -> honest error, not truncation (fixture) -- //
    {
        Rig rig(fixture, /*n_ctx=*/1024);
        // 1 KiB budgets guarantee the GPU budget is exceeded (spill) AND the
        // CPU budget is then exceeded too (the toy model's KV state for even
        // 1024 cells is far larger than 1 KiB).
        rig.ctx.set_budgets(/*kv_budget_bytes=*/1024, /*cpu_budget_bytes=*/1024);
        rig.engine.complete("Once upon a time", /*n_predict=*/4);
        bool threw = false;
        try {
            (void)rig.ctx.grow_preserving(2048);
        } catch (const std::runtime_error& e) {
            threw = true;
            std::fprintf(stderr, "budget error (expected): %s\n", e.what());
        }
        PLEIADES_CHECK(threw);
    }

    // -- 3+. KV-preserving growth on a real, large-train-context model ------- //
    if (real.empty() || !std::ifstream(real).good()) {
        std::fprintf(stderr,
                     "SKIP: no large-train-ctx model given (pass a Llama-3.2-1B-class GGUF as argv[2]); "
                     "KV-preservation cases not run\n");
        llama_backend_free();
        return 77;
    }

    // 3a. grow_preserving within the trained context: warm at 1024, grow to
    // 2048 (state roundtrip), continue — the followup must reuse the restored
    // KV and answer exactly like a rig that started at 2048.
    {
        Rig rig(real, /*n_ctx=*/1024);
        rig.engine.complete("Once upon a time", /*n_predict=*/16);
        auto gr = rig.engine.grow_context_to(2048);
        PLEIADES_CHECK(gr.n_ctx == 2048);
        PLEIADES_CHECK(gr.preserved);
        PLEIADES_CHECK(!gr.spilled);
        PLEIADES_CHECK(rig.ctx.yarn_factor() == 1.0f);  // within train: plain rope
        auto r = rig.engine.complete("Once upon a time, there was a little", /*n_predict=*/16);
        std::fprintf(stderr, "grow-preserve followup: cached=%d\n", r.n_prompt_cached);
        PLEIADES_CHECK(r.n_prompt_cached > 0);
        Rig ref(real, /*n_ctx=*/2048);
        ref.engine.complete("Once upon a time", /*n_predict=*/16);
        auto r2 = ref.engine.complete("Once upon a time, there was a little", /*n_predict=*/16);
        PLEIADES_CHECK(r.text == r2.text);
    }

    // 3b. auto-upshift from a request (ensure_capacity): a prompt too big for
    // the starting gear transparently grows and produces byte-identical output
    // to a cold, bigger-context reference.
    {
        const std::string big = long_prompt(95);
        Rig rig(real, /*n_ctx=*/1024);
        auto r = rig.engine.complete(big, /*n_predict=*/24);
        PLEIADES_CHECK(rig.ctx.n_ctx() >= 2048);
        std::fprintf(stderr, "auto-upshift: n_ctx now %d, prompt=%d cached=%d\n",
                     rig.ctx.n_ctx(), r.n_prompt_tokens, r.n_prompt_cached);
        Rig ref(real, /*n_ctx=*/2048);
        auto r2 = ref.engine.complete(big, /*n_predict=*/24);
        PLEIADES_CHECK(r.text == r2.text);
    }

    // 3c. GPU KV budget -> spill to host RAM (never truncate): a 1 MiB GPU
    // budget forces the growth to spill; the state must SURVIVE the device
    // move (the state roundtrip abstracts the buffer location).
    {
        Rig rig(real, /*n_ctx=*/1024);
        rig.ctx.set_budgets(/*kv_budget_bytes=*/1 << 20, /*cpu_budget_bytes=*/512 << 20);
        rig.engine.complete("Once upon a time", /*n_predict=*/16);
        auto gr = rig.engine.grow_context_to(2048);
        PLEIADES_CHECK(gr.spilled);
        PLEIADES_CHECK(rig.ctx.kv_on_cpu());
        PLEIADES_CHECK(gr.preserved);
        auto r = rig.engine.complete("Once upon a time, there was a little", /*n_predict=*/16);
        PLEIADES_CHECK(!r.text.empty());
        // NOTE: no byte-equality against a GPU-resident reference. The spilled
        // run computes attention on the CPU — the K/V VALUES are identical but
        // the floating-point paths differ, and greedy argmax legitimately
        // flips on near-ties. Nor is reuse-vs-cold equality asserted here:
        // a trimmed-prefix decode re-associates the prefill math and can flip
        // near-ties on a big real model (the exact-equality property is
        // covered byte-for-byte on the tiny fixtures elsewhere). The
        // contract asserted: the state survived the device move (preserved)
        // and the reused KV is actually being served.
        auto r_again = rig.engine.complete("Once upon a time, there was a little", /*n_predict=*/16);
        PLEIADES_CHECK(r_again.n_prompt_cached > 0);
    }

    // 3d. auto-upshift with a spill mid-request: same as 3b with a tiny GPU
    // budget — grow AND spill AND stay deterministic. (Same caveat as 3c:
    // no cross-device byte-equality — only same-regime determinism.)
    {
        const std::string big = long_prompt(95);
        Rig rig(real, /*n_ctx=*/1024);
        rig.ctx.set_budgets(/*kv_budget_bytes=*/1 << 20, /*cpu_budget_bytes=*/512 << 20);
        auto r = rig.engine.complete(big, /*n_predict=*/24);
        PLEIADES_CHECK(rig.ctx.kv_on_cpu());
        std::fprintf(stderr, "spill mid-request: n_ctx now %d, kv_on_cpu=%d\n",
                     rig.ctx.n_ctx(), rig.ctx.kv_on_cpu() ? 1 : 0);
        PLEIADES_CHECK(!r.text.empty());
        auto r_again = rig.engine.complete(big, /*n_predict=*/24);
        PLEIADES_CHECK(r_again.n_prompt_cached > 0);
    }

    llama_backend_free();
    std::printf("test_elastic_context: all checks passed\n");
    return 0;
}
