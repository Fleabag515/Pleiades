// Slot-state save/restore round trip (Phase 1 parity: --slot-save-path +
// POST /slots/0?action=save|restore). Uses the tiny fixture GGUF; see
// tests/CMakeLists.txt.
#include "pleiades_engine/engine.h"
#include "pleiades_engine/slot_state.h"

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

    Rig(const std::string& model_path, int cache_reuse = 0)
        : engine(models, ctx, cache_reuse) {
        // Flash attention explicitly OFF: these tests rely on prefix reuse
        // across SHORTER follow-ups (the warm prompt's own generated tokens
        // make the resident sequence longer than the next prompt), which the
        // FA stale-cell guard correctly clears when FA may be on. The
        // explicit-mask path is the exactness gold standard (see engine.cpp).
        pleiades_engine::ContextParams params;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        models.load(model_path, /*n_gpu_layers=*/0);
        ctx.create(models.model(), /*n_ctx=*/512, /*n_ctx_max=*/0, params);
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    const std::string model_path = argv[1];
    const std::string save_path = model_path + ".slot-test.bin";

    const std::string warm = "Once upon a time";
    const std::string followup = "Once upon a time, there was a little";

    // Reference run: warm up, save nothing, just continue -- the answer the
    // restore path must reproduce exactly (greedy decoding is deterministic).
    std::string ref_followup;
    {
        Rig rig(model_path);
        rig.engine.complete(warm, /*n_predict=*/24);
        auto r = rig.engine.complete(followup, /*n_predict=*/24);
        ref_followup = r.text;
        PLEIADES_CHECK(!ref_followup.empty());
        PLEIADES_CHECK(r.n_prompt_cached > 0);  // warm prefix was resident
    }

    // Save from a live state, then restore into a FRESH context+engine --
    // the same thing models.py's stop()/start() handoff does across a process
    // boundary.
    {
        Rig rig(model_path);
        auto warmed = rig.engine.complete(warm, /*n_predict=*/24);
        PLEIADES_CHECK(warmed.n_generated_tokens > 0);

        PLEIADES_CHECK(pleiades_engine::save_slot_state(rig.ctx.ctx(),
                                                        rig.engine.prefix_cache().tokens(), save_path));

        // Corrupt-file handling: a garbage file must restore false and leave
        // the engine usable.
        {
            std::ofstream bad(save_path + ".bad", std::ios::binary);
            bad << "this is not a slot file";
        }
        bool restored_bad = pleiades_engine::restore_slot_state(rig.ctx.ctx(), rig.engine, save_path + ".bad");
        PLEIADES_CHECK(!restored_bad);
        auto after_bad = rig.engine.complete("A different prompt entirely", /*n_predict=*/4);
        PLEIADES_CHECK(!after_bad.text.empty());

        Rig fresh(model_path);
        PLEIADES_CHECK(fresh.engine.prefix_cache().size() == 0);
        PLEIADES_CHECK(pleiades_engine::restore_slot_state(fresh.ctx.ctx(), fresh.engine, save_path));
        // The restored engine must have adopted the saved prefix.
        PLEIADES_CHECK(fresh.engine.prefix_cache().size() > 0);

        // The restored engine answers a followup identically to the never-
        // restarted reference (state round trip + prefix adoption are exact).
        auto r = fresh.engine.complete(followup, /*n_predict=*/24);
        PLEIADES_CHECK(r.text == ref_followup);
        PLEIADES_CHECK(r.n_prompt_cached > 0);  // restored KV actually reused
    }

    // Missing file -> clean false, engine unaffected.
    {
        Rig rig(model_path);
        PLEIADES_CHECK(!pleiades_engine::restore_slot_state(rig.ctx.ctx(), rig.engine, "/nonexistent/x.bin"));
        auto r = rig.engine.complete(warm, /*n_predict=*/8);
        PLEIADES_CHECK(!r.text.empty());
    }

    std::remove(save_path.c_str());
    std::remove((save_path + ".bad").c_str());
    llama_backend_free();
    std::printf("test_slot_state: all checks passed\n");
    return 0;
}
