// Phase 6 prefix-cache tests. Requires the tiny fixture GGUF (see
// tests/CMakeLists.txt) as argv[1].
//
// Two layers here:
//   1. Pure PrefixCache bookkeeping (no model) -- LCP math + the
//      "always leave one token" cap + epoch invalidation.
//   2. Engine-level behavior against the real (tiny) fixture model:
//      a repeated prompt reuses the cache and produces byte-identical
//      output to a cold decode; a prompt that diverges partway only
//      re-decodes the diverged suffix; a resize drops the cache (strategy
//      3b) and the engine still generates correctly afterward.
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
    ContextGovernor ctx;
    ctx.create(models.model(), /*n_ctx=*/512);
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
        ctx2.create(models.model(), /*n_ctx=*/512);
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

    llama_backend_free();
    return 0;
}
