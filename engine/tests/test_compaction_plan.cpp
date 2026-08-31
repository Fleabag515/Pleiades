// Phase 4 step 2 (docs/specs/2026-07-21-context-free-model-architecture-
// design.md, "bonus lane") -- the compaction MECHANISM proven correct in
// isolation, against a real dense model, before any policy exists to decide
// when to call it (see compaction_plan.h's own header comment).
//
// The load-bearing check: "byte-identical-to-truncation" -- if you decode
// A+B+C, then evict B via compact(), the resulting KV must be
// indistinguishable from having decoded A+C directly and never touched B at
// all. Requires the tiny fixture GGUF (see tests/CMakeLists.txt) as argv[1].
#include "pleiades_engine/compaction_plan.h"

#include <cstdio>
#include <string>
#include <vector>

#include "llama.h"
#include "pleiades_engine/context_governor.h"
#include "pleiades_engine/engine.h"
#include "pleiades_engine/model_manager.h"
#include "test_util.h"

using namespace pleiades_engine;

namespace {

std::vector<llama_token> tok(const llama_vocab* vocab, const std::string& text) {
    std::vector<llama_token> out(text.size() + 8);
    int n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), out.data(),
                           static_cast<int32_t>(out.size()), /*add_special=*/true, /*parse_special=*/true);
    if (n < 0) {
        out.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), out.data(),
                           static_cast<int32_t>(out.size()), /*add_special=*/true, /*parse_special=*/true);
    }
    out.resize(n);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    std::string model_path = argv[1];

    ModelManager models;
    models.load(model_path, /*n_gpu_layers=*/0);
    const llama_vocab* vocab = llama_model_get_vocab(models.model());

    // Flash attention off, same reasoning as test_resident_map.cpp: isolate
    // the compaction mechanism under test from the unrelated FA stale-KV
    // hazard that mechanism's own fix handles separately.
    ContextParams fa_off;
    fa_off.auto_yarn = false;  // toy fixture (n_ctx_train=128): plain rope; YaRN covered in test_elastic_context
    fa_off.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    const std::string a = "Once upon a time in a distant kingdom, ";
    const std::string b = "the weather that particular Tuesday was unremarkable, mild and "
                          "overcast with a light breeze, ";
    const std::string c = "there lived a";

    // -- (1) Determine the eviction span at the TOKEN level, not by hoping
    //         independently-tokenized "a" and "a+b" happen to land on the
    //         same boundaries a real tokenizer would pick inside "a+b+c"
    //         (BPE merges can differ at a seam depending on what follows).
    //         Instead: tokenize the full prompt and the target (post-
    //         eviction) prompt independently, then find the actual common
    //         prefix/suffix between them at the token level. This is
    //         self-verifying below, not just assumed correct. --------------
    std::vector<llama_token> full = tok(vocab, a + b + c);
    std::vector<llama_token> target = tok(vocab, a + c);

    size_t common_prefix = 0;
    while (common_prefix < full.size() && common_prefix < target.size() &&
          full[common_prefix] == target[common_prefix]) {
        ++common_prefix;
    }
    size_t common_suffix = 0;
    while (common_suffix < full.size() - common_prefix && common_suffix < target.size() - common_prefix &&
          full[full.size() - 1 - common_suffix] == target[target.size() - 1 - common_suffix]) {
        ++common_suffix;
    }
    const size_t evict_start = common_prefix;
    const size_t evict_end = full.size() - common_suffix;
    PLEIADES_CHECK(evict_end > evict_start);  // b must correspond to a real, nonempty token span

    // Self-check independent of the KV mechanism: removing [evict_start,
    // evict_end) from `full` must reconstruct `target` exactly at the
    // token level. If this fails, the span computation above is wrong --
    // not the compaction mechanism being tested below.
    {
        std::vector<llama_token> reconstructed(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(evict_start));
        reconstructed.insert(reconstructed.end(), full.begin() + static_cast<std::ptrdiff_t>(evict_end), full.end());
        PLEIADES_CHECK(reconstructed == target);
    }

    // -- (2) Truncation oracle: decode A+C directly, never touching B ---- //
    ContextGovernor ref_ctx;
    ref_ctx.create(models.model(), /*n_ctx=*/512, /*n_ctx_max=*/0, fa_off);
    Engine ref_engine(models, ref_ctx);
    auto reference = ref_engine.complete(a + c, /*n_predict=*/24);
    PLEIADES_CHECK(reference.n_generated_tokens > 0);

    // -- (3) Decode A+B+C for real, then evict B via the mechanism ------- //
    ContextGovernor ctx;
    ctx.create(models.model(), /*n_ctx=*/512, /*n_ctx_max=*/0, fa_off);
    Engine engine(models, ctx);
    engine.complete(a + b + c, /*n_predict=*/0);  // decode only, no generation
    PLEIADES_CHECK(engine.resident_map().size() == full.size());

    auto plan = engine.compact_resident_span(evict_start, evict_end);
    PLEIADES_CHECK(plan.applied);
    PLEIADES_CHECK(engine.resident_map().size() == target.size());
    PLEIADES_CHECK(engine.resident_map().tokens() == target);  // ResidentMap now mirrors A+C exactly

    // -- (4) Continue through the NORMAL request path (complete(a+c, ...))
    //         -- reusable_prefix() should find the compacted state almost
    //         entirely reusable, and the resulting text must be byte-
    //         identical to the truncation oracle from step (2). This is
    //         the actual "byte-identical-to-truncation" proof the design
    //         doc's sequencing note calls for. -------------------------- //
    auto after_compact = engine.complete(a + c, /*n_predict=*/24);
    PLEIADES_CHECK(after_compact.n_prompt_cached > 0);          // real reuse against the compacted KV
    PLEIADES_CHECK(after_compact.n_prompt_cached == static_cast<int>(target.size()) - 1);  // near-total reuse
    PLEIADES_CHECK(after_compact.text == reference.text);       // <- the actual correctness proof

    // -- (5) Refusal paths: must not crash or corrupt state -------------- //
    {
        auto bad_empty = engine.compact_resident_span(5, 5);       // empty span
        PLEIADES_CHECK(!bad_empty.applied);
        auto bad_range = engine.compact_resident_span(0, engine.resident_map().size() + 100);  // out of range
        PLEIADES_CHECK(!bad_range.applied);
        // Engine must still work normally after both refusals (no partial
        // mutation leaked through).
        auto still_ok = engine.complete(a + c, /*n_predict=*/4);
        PLEIADES_CHECK(still_ok.n_generated_tokens > 0);
    }

    // -- (6) Flash-attention: compact() must REFUSE, not silently corrupt --
    //
    //         This section originally attempted the same byte-identical-to-
    //         truncation proof as sections (2)-(4), but under flash
    //         attention, using the exact prompt shape test_resident_map.cpp's
    //         own section 5 already proved sensitive enough to flip a greedy
    //         argmax under this pinned llama.cpp build's known FA kernel
    //         leak (a short/bland prompt wouldn't perturb the argmax enough
    //         to expose a leak even if one exists). It found a REAL,
    //         previously-unknown bug: compact()'s seq_rm/seq_add never
    //         scrub the evicted span's cell DATA (only metadata/position),
    //         so the evicted span's old physical cells become genuinely
    //         orphaned in the MIDDLE of the live range (verified by reading
    //         build_graph_shift: the RoPE shift re-rotates each surviving
    //         cell's data IN PLACE at its OWN physical slot, it does not
    //         move data between slots) -- and generate()'s existing FA guard
    //         (fa_maybe_on && prompt_tokens.size() < resident) has no
    //         visibility into compact()'s history and never fires here (the
    //         post-compaction prompt is reused almost entirely, not shorter
    //         than resident). Generation genuinely diverged from the
    //         truncation oracle. Fix: compact() now refuses outright when
    //         flash attention may be active (compaction_plan.h/.cpp) --
    //         there is no cheap scrub-and-proceed option the way generate()
    //         has, since a full data clear would force a cold re-decode of
    //         the entire sequence, defeating compaction's purpose entirely.
    //         This section now verifies that refusal actually fires. ------ //
    {
        ContextParams fa_on;
        fa_on.auto_yarn = false;  // toy fixture: see test_elastic_context
        fa_on.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

        const std::string fa_a = "<|im_start|>system\nYou are a helpful assistant. Use tools when "
                                 "appropriate.<|im_end|>\n<|im_start|>user\n";
        const std::string fa_b = "By the way, completely unrelated: my favorite color is blue and I "
                                 "had cereal for breakfast this morning.\n";
        const std::string fa_c = "What is the weather in San Francisco? Use the get_weather "
                                 "tool.<|im_end|>\n<|im_start|>assistant\n";

        ContextGovernor fa_ctx;
        fa_ctx.create(models.model(), /*n_ctx=*/512, /*n_ctx_max=*/0, fa_on);
        Engine fa_engine(models, fa_ctx);
        fa_engine.complete(fa_a + fa_b + fa_c, /*n_predict=*/0);
        const size_t before_size = fa_engine.resident_map().size();

        // Any nonempty, in-range span refuses under FA -- exact boundaries
        // don't matter here, only that the FA guard fires before the KV/map
        // are ever touched.
        auto fa_plan = fa_engine.compact_resident_span(10, before_size - 5);
        PLEIADES_CHECK(!fa_plan.applied);
        // No partial mutation leaked through: resident_map must be
        // untouched, and the engine must still generate normally afterward.
        PLEIADES_CHECK(fa_engine.resident_map().size() == before_size);
        auto fa_still_ok = fa_engine.complete(fa_a + fa_b + fa_c, /*n_predict=*/4);
        PLEIADES_CHECK(fa_still_ok.n_generated_tokens > 0);
    }

    llama_backend_free();
    return 0;
}
