// Middle-of-prompt KV chunk reuse (Phase 1 parity: llama-server's
// --cache-reuse). Correctness bar: with reuse enabled, every shape must
// produce byte-identical greedy output to a cold engine that never reused
// anything, and deletion-realignment shapes must salvage chunks (cached
// tokens strictly beyond the common prefix).
//
// The shapes mirror what Anamnesis actually sends: [stable persona prefix]
// [rewritten memory block] [long stable tail]. A block that shrank or kept
// its length realigns the tail and gets salvaged; a block that GREW cannot
// be salvaged within a turn (llama.cpp requires contiguous alive positions;
// see engine.cpp) and simply decodes fresh -- correctness is still asserted.
#include "pleiades_engine/engine.h"

#include <cstdio>
#include <string>

#include "llama.h"
#include "pleiades_engine/context_governor.h"
#include "pleiades_engine/model_manager.h"
#include "test_util.h"

namespace {

struct Rig {
    pleiades_engine::ModelManager models;
    pleiades_engine::ContextGovernor ctx;
    pleiades_engine::Engine engine;

    Rig(const std::string& model_path, int cache_reuse = 0,
        llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_DISABLED)
        : engine(models, ctx, cache_reuse) {
        pleiades_engine::ContextParams params;
        params.flash_attn_type = fa;
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

    const std::string stable_head =
        "The old lighthouse keeper wrote in his journal every single evening after sunset, describing";
    const std::string middle_a = "the color of the water and the shapes of passing ships";
    const std::string middle_b = "the mood of the wind and the sound of the fog bell";
    const std::string stable_tail =
        ", and he always finished the entry the same way: the light was lit, the sea was calm, the ships"
        " were safe, and the keeper went to bed content, dreaming of tomorrow's sunrise over the harbor";
    const std::string filler =
        " Later the keeper added a lengthy appendix describing every storm of that winter, the dates,"
        " the wind speeds, the wave heights, and the repairs each one demanded of the tower";

    const std::string p1 = stable_head + middle_a + stable_tail;
    const std::string p2 = stable_head + middle_b + stable_tail;                  // rewritten, ~equal length
    const std::string p_deleted = stable_head + stable_tail;                      // block shrank -> tail realigns
    const std::string extra2 =
        " And so the winter passed, and the spring came, and the keeper kept writing every night";
    const std::string p_grown = stable_head + stable_tail + filler + extra2;      // pure deletion, total >= resident
    const int threshold = 8;

    // Cold references: fresh engine, no prior KV, reuse disabled.
    std::string ref_p2, ref_deleted, ref_grown;
    {
        Rig rig(model_path);
        ref_p2 = rig.engine.complete(p2, /*n_predict=*/24).text;
        ref_deleted = rig.engine.complete(p_deleted, /*n_predict=*/24).text;
        ref_grown = rig.engine.complete(p_grown, /*n_predict=*/24).text;
        PLEIADES_CHECK(!ref_p2.empty());
        PLEIADES_CHECK(!ref_deleted.empty());
        PLEIADES_CHECK(!ref_grown.empty());
    }

    // Deletion realignment, FA OFF, prompt SHORTER than resident: the shared
    // tail beyond the LCP is salvaged via seq_add moves; stale cells beyond
    // the (shrunken) prompt are exactly masked without flash attention.
    {
        Rig rig(model_path, threshold, LLAMA_FLASH_ATTN_TYPE_DISABLED);
        rig.engine.complete(p1, /*n_predict=*/8);
        auto r = rig.engine.complete(p_deleted, /*n_predict=*/24);
        std::fprintf(stderr, "deletion (fa off, shrink): prompt=%d cached=%d chunks=%d\n",
                     r.n_prompt_tokens, r.n_prompt_cached, r.n_chunks_reused);
        PLEIADES_CHECK(r.n_chunks_reused >= 1);
        PLEIADES_CHECK(r.n_prompt_cached > 0);
        PLEIADES_CHECK(r.text == ref_deleted);
    }

    // Deletion realignment, FA ON, prompt >= resident: chunk moves + the
    // suffix decode overwrite every cell, so no stale data survives and
    // reuse under flash attention must be exact.
    {
        Rig rig(model_path, threshold, LLAMA_FLASH_ATTN_TYPE_ENABLED);
        rig.engine.complete(p1 + filler, /*n_predict=*/1);
        auto r = rig.engine.complete(p_grown, /*n_predict=*/24);
        std::fprintf(stderr, "deletion (fa on, grow): prompt=%d cached=%d chunks=%d\n",
                     r.n_prompt_tokens, r.n_prompt_cached, r.n_chunks_reused);
        PLEIADES_CHECK(r.n_chunks_reused >= 1);
        PLEIADES_CHECK(r.text == ref_grown);
    }

    // Rewritten block under FA ON with a prompt SHORTER than resident: the
    // stale-cell guard forces the cold path -- no chunk credit, correct text.
    {
        Rig rig(model_path, threshold, LLAMA_FLASH_ATTN_TYPE_ENABLED);
        rig.engine.complete(p1, /*n_predict=*/8);
        auto r = rig.engine.complete(p2, /*n_predict=*/24);
        std::fprintf(stderr, "rewrite (fa on, shrink): prompt=%d cached=%d chunks=%d\n",
                     r.n_prompt_tokens, r.n_prompt_cached, r.n_chunks_reused);
        PLEIADES_CHECK(r.n_chunks_reused == 0);
        PLEIADES_CHECK(r.text == ref_p2);
    }

    // Rewritten block, FA OFF: reuse machinery runs; a same-length rewrite
    // has no deletable gap, so zero chunks is fine -- correctness is not.
    {
        Rig rig(model_path, threshold, LLAMA_FLASH_ATTN_TYPE_DISABLED);
        rig.engine.complete(p1, /*n_predict=*/8);
        auto r = rig.engine.complete(p2, /*n_predict=*/24);
        std::fprintf(stderr, "rewrite (fa off): prompt=%d cached=%d chunks=%d\n",
                     r.n_prompt_tokens, r.n_prompt_cached, r.n_chunks_reused);
        PLEIADES_CHECK(r.text == ref_p2);
    }

    // Disabled engine (cache_reuse=0): no chunk pass ever runs.
    {
        Rig rig(model_path);
        rig.engine.complete(p1, /*n_predict=*/8);
        auto r = rig.engine.complete(p2, /*n_predict=*/24);
        PLEIADES_CHECK(r.n_chunks_reused == 0);
        PLEIADES_CHECK(r.text == ref_p2);
    }

    llama_backend_free();
    std::printf("test_cache_reuse: all checks passed\n");
    return 0;
}
