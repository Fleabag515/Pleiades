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

    llama_backend_free();
    return 0;
}
