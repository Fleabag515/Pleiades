// Requires the tiny fixture GGUF (see tests/CMakeLists.txt) as argv[1].
#include "pleiades_engine/context_governor.h"

#include <string>

#include "llama.h"
#include "pleiades_engine/model_manager.h"
#include "test_util.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    std::string model_path = argv[1];

    pleiades_engine::ModelManager models;
    models.load(model_path, /*n_gpu_layers=*/0);

    // -- create() ------------------------------------------------------- //
    pleiades_engine::ContextGovernor ctx;
    ctx.create(models.model(), /*n_ctx=*/1024, /*n_ctx_max=*/32768);
    PLEIADES_CHECK(ctx.ctx() != nullptr);
    PLEIADES_CHECK(ctx.n_ctx() == 1024);
    PLEIADES_CHECK(ctx.n_ctx_max() == 32768);

    // -- clamp_gear() ----------------------------------------------------- //
    // Below the floor clamps up to 1024 (ContextGovernor's own hard floor,
    // ported from hardware.py/server.py's clamp_gear()).
    PLEIADES_CHECK(ctx.clamp_gear(1) == 1024);
    // Within range passes through unchanged.
    PLEIADES_CHECK(ctx.clamp_gear(8192) == 8192);
    // Above the ceiling clamps down to n_ctx_max.
    PLEIADES_CHECK(ctx.clamp_gear(1000000) == 32768);

    // -- next_gear() -------------------------------------------------------- //
    // At n_ctx=1024, the next rung on the CONTEXT_GEARS ladder is 4096.
    auto next = ctx.next_gear();
    PLEIADES_CHECK(next.has_value());
    PLEIADES_CHECK(*next == 4096);

    // -- resize() ------------------------------------------------------- //
    // NOTE: not asserting ctx.ctx() != before-the-resize here. A pointer
    // that's freed and immediately reallocated (same size class, no other
    // allocations in between) very commonly gets the SAME address back
    // from the allocator -- that's a real, observed behavior, not a
    // reliable signal of "did a real free+recreate happen." The actual
    // free+recreate contract is verified functionally instead: n_ctx()
    // reflects the new size, and (in test_engine.cpp) generation still
    // works correctly after a resize.
    int actual = ctx.resize(4096);
    PLEIADES_CHECK(actual == 4096);
    PLEIADES_CHECK(ctx.n_ctx() == 4096);

    // Resizing to the same n_ctx is a no-op (matches EngineState.resize()'s
    // "already there" short-circuit in server.py). Verified via the same
    // pointer being returned (this direction IS a reliable signal: the
    // no-op path explicitly returns early without touching ctx_ at all,
    // so it's guaranteed to be the same object, not just possibly the
    // same address).
    llama_context* after_first_resize = ctx.ctx();
    int actual2 = ctx.resize(4096);
    PLEIADES_CHECK(actual2 == 4096);
    PLEIADES_CHECK(ctx.ctx() == after_first_resize);

    // A ceiling of 0 (constructor default) falls back to the model's own
    // trained context, not an unbounded/undefined clamp. Note: clamp_gear()
    // still applies its hard 1024 floor on top of that ceiling (ported
    // verbatim from server.py's `max(1024, min(requested, ceiling))`), so
    // for a model with a tiny trained context (like this fixture) the
    // floor can dominate -- that's inherited, expected behavior, not a
    // bug, so the expectation below mirrors the same formula rather than
    // assuming the ceiling always wins.
    pleiades_engine::ContextGovernor ctx2;
    ctx2.create(models.model(), 1024);  // n_ctx_max defaults to 0
    int train_ctx = llama_model_n_ctx_train(models.model());
    int expected = train_ctx > 1024 ? train_ctx : 1024;
    PLEIADES_CHECK(ctx2.clamp_gear(100000000) == expected);

    // -- ContextParams parity (flash-attn/KV-quant/n_ubatch/n_batch/threads) //
    // Context-param parity knobs must be retained verbatim across resize()
    // -- a resize that silently reverted to library defaults would be
    // exactly the regression the design doc's GPU/context-parity pass
    // flagged ("make sure ... n_ubatch is actually plumbed through, not
    // hardcoded"). llama_init_from_model() itself is llama.cpp's own
    // tested code (not re-verified here); what's under test is that THIS
    // governor actually threads non-default params through create() and
    // reapplies the SAME ones on every resize(), rather than only ever
    // touching n_ctx like the pre-parity-pass version did.
    //
    // Two cases, not one, because a real llama.cpp constraint was hit
    // while writing this test (src/llama-context.cpp: "V cache
    // quantization requires flash_attn" -- llama_init_from_model() rejects
    // a quantized type_v paired with flash_attn_type == DISABLED). That's
    // real upstream behavior, not a bug in this governor, so the two
    // orthogonal knob groups are exercised with combinations that are
    // actually valid together.
    {
        // Case A: flash attention explicitly disabled, KV cache left at
        // F16 (a disabled-flash-attn-compatible combination) -- exercises
        // n_ubatch/n_batch/n_threads/flash_attn_type plumbing.
        pleiades_engine::ContextGovernor ctx3;
        pleiades_engine::ContextParams cp;
        cp.n_ubatch = 256;
        cp.n_batch = 1024;
        cp.n_threads = 2;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx3.create(models.model(), /*n_ctx=*/1024, /*n_ctx_max=*/32768, cp);
        PLEIADES_CHECK(ctx3.ctx() != nullptr);
        PLEIADES_CHECK(ctx3.params().n_ubatch == 256);
        PLEIADES_CHECK(ctx3.params().n_batch == 1024);
        PLEIADES_CHECK(ctx3.params().n_threads == 2);
        PLEIADES_CHECK(ctx3.params().flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED);

        // Resize must succeed (llama_init_from_model accepted these
        // non-default params again on the recreate path) and params() must
        // be unchanged afterward -- resize() must not reset to defaults.
        int actual = ctx3.resize(4096);
        PLEIADES_CHECK(actual == 4096);
        PLEIADES_CHECK(ctx3.ctx() != nullptr);
        PLEIADES_CHECK(ctx3.params().n_ubatch == 256);
        PLEIADES_CHECK(ctx3.params().flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED);
    }
    {
        // Case B: KV cache type changed to BF16 (a real, non-default
        // type_k/type_v this fixture actually accepts) with flash
        // attention enabled -- exercises type_k/type_v plumbing.
        //
        // NOTE: this was originally written against GGML_TYPE_Q8_0, which
        // is what production callers actually want (real block-quantized
        // KV cache, matching llama-server's own -ctk/-ctv q8_0 usage).
        // Hit two real llama.cpp constraints running it against this
        // fixture, in order (src/llama-context.cpp):
        //   1. "V cache quantization requires flash_attn" -- a quantized
        //      type_v needs flash_attn_type != DISABLED (fixed by using
        //      ENABLED here, see Case A above for the DISABLED path).
        //   2. "K cache type q8_0 with block size 32 does not divide
        //      n_embd_head_k=48" -- this fixture's own head dimension (48)
        //      isn't evenly divisible by any block-quantized type's block
        //      size (32, for every quant type in kv_cache_types), so NO
        //      block-quantized KV type works against THIS SPECIFIC tiny
        //      fixture model. That's a property of the fixture (a small
        //      head_dim chosen for a tiny test model), not a bug in this
        //      governor or in llama.cpp -- production models (e.g. Qwen
        //      family, head_dim 64/128) don't hit it. GGML_TYPE_BF16 is
        //      used instead here specifically because ggml_is_quantized()
        //      returns false for it, so neither constraint applies --
        //      this still proves type_k/type_v aren't hardcoded to F16,
        //      which is what this test is actually checking; real q8_0 KV
        //      quantization against a production-shaped model was verified
        //      separately in the manual smoke test (see this pass's commit
        //      message / report, not unit-tested here since it needs a
        //      real multi-GB GGUF, not a CI-sized fixture).
        pleiades_engine::ContextGovernor ctx4;
        pleiades_engine::ContextParams cp;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = GGML_TYPE_BF16;
        cp.type_v = GGML_TYPE_BF16;
        ctx4.create(models.model(), /*n_ctx=*/1024, /*n_ctx_max=*/32768, cp);
        PLEIADES_CHECK(ctx4.ctx() != nullptr);
        PLEIADES_CHECK(ctx4.params().type_k == GGML_TYPE_BF16);
        PLEIADES_CHECK(ctx4.params().type_v == GGML_TYPE_BF16);

        int actual = ctx4.resize(4096);
        PLEIADES_CHECK(actual == 4096);
        PLEIADES_CHECK(ctx4.ctx() != nullptr);
        PLEIADES_CHECK(ctx4.params().type_k == GGML_TYPE_BF16);
    }

    // Default-constructed ContextParams must match
    // llama_context_default_params()'s own defaults exactly
    // (third_party/llama.cpp/src/llama-context.cpp) -- existing callers
    // that never pass a ContextParams (cli_main.cpp, the bench binaries)
    // must keep getting precisely today's pre-parity-pass behavior.
    {
        pleiades_engine::ContextParams defaults;
        PLEIADES_CHECK(defaults.n_batch == 2048);
        PLEIADES_CHECK(defaults.n_ubatch == 512);
        PLEIADES_CHECK(defaults.n_threads == 0);
        PLEIADES_CHECK(defaults.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO);
        PLEIADES_CHECK(defaults.type_k == GGML_TYPE_F16);
        PLEIADES_CHECK(defaults.type_v == GGML_TYPE_F16);
    }

    llama_backend_free();
    return 0;
}
