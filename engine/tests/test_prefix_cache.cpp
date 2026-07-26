// Phase 6 prefix-cache tests. Requires the tiny fixture GGUF (see
// tests/CMakeLists.txt) as argv[1].
//
// Two layers here:
//   1. Pure PrefixCache bookkeeping (no model) -- LCP math + the
//      "always leave one token" cap + epoch invalidation.
//   2. Engine-level behavior against the real (tiny) fixture model, with
//      flash attention OFF (see main()): a repeated prompt reuses the cache
//      and produces byte-identical output to a cold decode; a prompt that
//      diverges partway only re-decodes the diverged suffix; a resize drops
//      the cache (strategy 3b) and the engine still generates correctly.
//   3. Flash-attention stale-KV regression (section 5, FA ON): the fix that
//      makes reuse safe under flash attention -- a shorter/repeat request
//      cold-decodes (no stale-cell leak, byte-identical), a growing request
//      still reuses the shared prefix. On the unfixed engine the repeat
//      request's output diverged (the corruption this whole pass fixed).
//
// The determinism check (warm cache hit == cold decode, byte for byte) is
// the load-bearing one: a caching bug here would silently corrupt every
// future conversation. See docs/specs/2026-07-21-native-inference-engine-
// design.md, Phase 6.
#include "pleiades_engine/prefix_cache.h"

#include <string>
#include <vector>

#include "llama.h"
#include "pleiades_engine/context_governor.h"
#include "pleiades_engine/engine.h"
#include "pleiades_engine/model_manager.h"
#include "test_util.h"

using namespace pleiades_engine;

static int test_pure_bookkeeping() {
    // Empty cache -> nothing reusable.
    {
        PrefixCache pc;
        PLEIADES_CHECK(pc.reusable_prefix({1, 2, 3}) == 0);
        PLEIADES_CHECK(pc.empty());
    }
    // Full match still leaves the last token to decode fresh.
    {
        PrefixCache pc;
        pc.set({10, 20, 30, 40}, /*epoch=*/1);
        PLEIADES_CHECK(pc.reusable_prefix({10, 20, 30, 40}) == 3);  // 4 - 1
    }
    // Divergence partway: reuse only the common run.
    {
        PrefixCache pc;
        pc.set({10, 20, 30, 40}, 1);
        PLEIADES_CHECK(pc.reusable_prefix({10, 20, 99, 40}) == 2);
    }
    // New prompt shorter than resident: cap at prompt.size()-1.
    {
        PrefixCache pc;
        pc.set({10, 20, 30, 40, 50}, 1);
        PLEIADES_CHECK(pc.reusable_prefix({10, 20}) == 1);
    }
    // New prompt longer, shares whole resident run: reuse all of resident
    // (which is < prompt.size(), so the "leave one" cap doesn't bind).
    {
        PrefixCache pc;
        pc.set({10, 20}, 1);
        PLEIADES_CHECK(pc.reusable_prefix({10, 20, 30, 40}) == 2);
    }
    // 1-token prompt: nothing reusable (must decode the single token).
    {
        PrefixCache pc;
        pc.set({10}, 1);
        PLEIADES_CHECK(pc.reusable_prefix({10}) == 0);
    }
    // append() extends the resident run.
    {
        PrefixCache pc;
        pc.set({10, 20}, 1);
        pc.append(30);
        PLEIADES_CHECK(pc.size() == 3);
        PLEIADES_CHECK(pc.reusable_prefix({10, 20, 30, 40}) == 3);
    }
    // invalidate() clears and rebinds epoch.
    {
        PrefixCache pc;
        pc.set({10, 20}, 1);
        pc.invalidate(7);
        PLEIADES_CHECK(pc.empty());
        PLEIADES_CHECK(pc.epoch() == 7);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (int rc = test_pure_bookkeeping()) {
        return rc;
    }
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    std::string model_path = argv[1];

    ModelManager models;
    models.load(model_path, /*n_gpu_layers=*/0);

    // Reuse-math (sections 1-4) is verified with flash attention explicitly
    // DISABLED. FA-on prefix reuse has a separate, real llama.cpp stale-KV
    // hazard (fixed by engine.cpp's guard and regressed in section 5 below), so
    // isolating the reuse LOGIC here means pinning FA off -- otherwise the fix
    // would (correctly) cold-decode the shorter repeat prompts these sections
    // fire and the "reuse happened" assertions couldn't run at all. With FA off
    // the mask path makes reuse exact regardless of stale cells, so these
    // byte-identical checks exercise the reuse machinery directly.
    ContextParams fa_off;
    fa_off.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ContextGovernor ctx;
    ctx.create(models.model(), /*n_ctx=*/512, /*n_ctx_max=*/0, fa_off);
    Engine engine(models, ctx);

    // Keep prompts well within the fixture's own tiny trained context (128
    // tokens) so we're never in the extrapolation regime where FA/summation
    // order is allowed to differ between decodes (see test_engine.cpp's
    // resize note) -- that would confound the byte-identical check with an
    // unrelated numerical-order effect. These prompts are a handful of
    // tokens each.
    const std::string base = "Once upon a time there was a little";

    // -- (1) cold baseline: fresh context, no cache ---------------------- //
    engine.reset_prefix_cache();
    auto cold = engine.complete(base, /*n_predict=*/24);
    PLEIADES_CHECK(cold.n_prompt_tokens > 0);
    PLEIADES_CHECK(cold.n_prompt_cached == 0);       // nothing was cached
    PLEIADES_CHECK(cold.n_generated_tokens > 0);

    // -- (2) warm hit: identical prompt reuses all-but-one prompt token,
    //         and MUST produce byte-identical text ------------------------ //
    auto warm = engine.complete(base, /*n_predict=*/24);
    PLEIADES_CHECK(warm.n_prompt_tokens == cold.n_prompt_tokens);
    PLEIADES_CHECK(warm.n_prompt_cached == warm.n_prompt_tokens - 1);  // all but final
    PLEIADES_CHECK(warm.text == cold.text);           // determinism: byte-identical
    PLEIADES_CHECK(warm.n_generated_tokens == cold.n_generated_tokens);

    // -- (3) divergent prompt: shares a prefix, decodes only the suffix -- //
    // After (2), resident == tokens(base) + generated tokens from (2). A
    // prompt that shares base's leading words but diverges after them
    // should reuse the shared run and NOT re-decode it. We can't assert the
    // exact cached count without tokenizing here, but it must be > 0 (real
    // reuse happened) and < the full prompt (real suffix decode happened).
    const std::string divergent = "Once upon a time there was a big dragon";
    auto div = engine.complete(divergent, /*n_predict=*/16);
    PLEIADES_CHECK(div.n_prompt_cached > 0);
    PLEIADES_CHECK(div.n_prompt_cached < div.n_prompt_tokens);
    PLEIADES_CHECK(div.n_generated_tokens > 0);

    // The divergent result must ALSO equal a truly-cold decode of the same
    // prompt (correctness of partial reuse, not just of full reuse). Verify
    // against a fresh context.
    {
        ContextGovernor ctx2;
        ctx2.create(models.model(), /*n_ctx=*/512, /*n_ctx_max=*/0, fa_off);
        Engine engine2(models, ctx2);
        auto div_cold = engine2.complete(divergent, 16);
        PLEIADES_CHECK(div_cold.n_prompt_cached == 0);
        PLEIADES_CHECK(div.text == div_cold.text);    // partial-reuse determinism
    }

    // -- (4) resize invalidates the cache (strategy 3b) ------------------ //
    // A repeated prompt right after a resize must show n_prompt_cached == 0
    // (the resize threw the KV away, so nothing is reusable), and the engine
    // must still generate correctly.
    auto before = engine.complete(base, 8);
    PLEIADES_CHECK(before.n_prompt_cached > 0);        // cache was warm here
    ctx.resize(1024);
    auto after = engine.complete(base, 8);
    PLEIADES_CHECK(after.n_prompt_cached == 0);         // resize dropped it
    PLEIADES_CHECK(after.n_generated_tokens > 0);
    PLEIADES_CHECK(!after.text.empty());

    // -- (5) flash-attention stale-KV regression ------------------------- //
    // The real bug: with FLASH ATTENTION on, a second (shorter or diverged)
    // request that reuses a trimmed prefix read stale K/V from the freed cells
    // -- llama.cpp's FA kernel leaks masked-out cells' data -- and silently
    // corrupted generation. The fix cold-decodes (scrubbing KV data) when FA is
    // active and the prompt is shorter than the resident sequence, but keeps
    // reuse when the prompt GROWS. This section asserts both, against a real
    // FA-enabled context. On the UNFIXED engine the identical-repeat check
    // below diverges (that was the corruption). The prompt is deliberately
    // long enough that the FA perturbation would flip a greedy argmax -- the
    // handful-of-tokens prompts above stay under that threshold, which is why
    // the pre-fix engine passed sections 1-4 while still being broken here.
    {
        ContextParams fa_on;
        fa_on.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        // A ChatML-shaped prompt (the fixture tokenizes the markers as plain
        // text -- it has no tool template) that empirically flips this model's
        // greedy argmax on the unfixed engine: the identical-repeat below then
        // diverges from the cold decode. A shorter, blander prompt keeps the FA
        // perturbation under the flip threshold and would only exercise the
        // n_prompt_cached==0 behavioral assertion, not the byte-identical one.
        const std::string p =
            "<|im_start|>system\nYou are a helpful assistant. Use tools when appropriate.<|im_end|>\n"
            "<|im_start|>user\nWhat is the weather in San Francisco? Use the get_weather tool.<|im_end|>\n"
            "<|im_start|>assistant\n";

        // Cold reference in its own fresh FA-on context (no reuse possible).
        std::string cold_ref;
        {
            ContextGovernor c;
            c.create(models.model(), /*n_ctx=*/512, /*n_ctx_max=*/0, fa_on);
            Engine e(models, c);
            cold_ref = e.complete(p, /*n_predict=*/48).text;
        }

        ContextGovernor fctx;
        fctx.create(models.model(), /*n_ctx=*/512, /*n_ctx_max=*/0, fa_on);
        Engine fe(models, fctx);
        auto a1 = fe.complete(p, /*n_predict=*/48);        // cold on this engine
        auto a2 = fe.complete(p, /*n_predict=*/48);        // identical repeat
        PLEIADES_CHECK(a1.text == cold_ref);
        PLEIADES_CHECK(a2.text == a1.text);                 // <- corrupted pre-fix
        PLEIADES_CHECK(a2.n_prompt_cached == 0);            // safe cold path taken (prompt < resident)

        // Growth: a strictly LONGER prompt that shares p's prefix. A separate
        // engine seeds a small resident (short generation) then extends it, so
        // the second prompt genuinely exceeds the resident high-water -- the fix
        // keeps reuse (every freed cell is overwritten), so cached > 0, and the
        // output must still equal a cold decode of the same prompt.
        ContextGovernor gctx;
        gctx.create(models.model(), /*n_ctx=*/512, /*n_ctx_max=*/0, fa_on);
        Engine ge(models, gctx);
        ge.complete(p, /*n_predict=*/4);                    // small resident: p + 4 tokens
        const std::string pg =
            p + " walk through the ancient forest to deliver a woven basket of fresh warm bread and "
                "golden honey to her frail grandmother, taking great care never to stray from the "
                "narrow winding path between the tall dark trees.";
        std::string grow_cold;
        {
            ContextGovernor c;
            c.create(models.model(), /*n_ctx=*/512, /*n_ctx_max=*/0, fa_on);
            Engine e(models, c);
            grow_cold = e.complete(pg, /*n_predict=*/32).text;
        }
        auto grow = ge.complete(pg, /*n_predict=*/32);
        PLEIADES_CHECK(grow.n_prompt_cached > 0);           // reuse preserved under FA
        PLEIADES_CHECK(grow.n_prompt_cached < grow.n_prompt_tokens);
        PLEIADES_CHECK(grow.text == grow_cold);             // correct despite reuse
    }

    llama_backend_free();
    return 0;
}
