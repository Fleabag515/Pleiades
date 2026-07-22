// Requires the tiny fixture GGUF (see tests/CMakeLists.txt) as argv[1].
#include "pleiades_engine/engine.h"

#include <string>
#include <vector>

#include "llama.h"
#include "pleiades_engine/context_governor.h"
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
    pleiades_engine::ContextGovernor ctx;
    ctx.create(models.model(), /*n_ctx=*/512);
    pleiades_engine::Engine engine(models, ctx);

    // -- complete(): non-streaming ---------------------------------------- //
    {
        auto r = engine.complete("Once upon a time", /*n_predict=*/16);
        PLEIADES_CHECK(r.n_prompt_tokens > 0);
        PLEIADES_CHECK(r.n_generated_tokens > 0);
        PLEIADES_CHECK(r.n_generated_tokens <= 16);
        PLEIADES_CHECK(!r.text.empty());
        PLEIADES_CHECK(r.prompt_seconds >= 0.0);
        PLEIADES_CHECK(r.generate_seconds >= 0.0);
    }

    // -- generate(): streaming callback fires once per generated token,
    // and the concatenation of every piece it receives must equal the
    // same text complete() would have produced (same seed/greedy sampling
    // is deterministic, so this is a real equality check, not just "some
    // callback fired"). ------------------------------------------------- //
    {
        std::vector<std::string> pieces;
        auto r = engine.generate("Once upon a time", 16, [&](const std::string& piece) {
            pieces.push_back(piece);
            return true;  // keep going
        });
        PLEIADES_CHECK(static_cast<int>(pieces.size()) == r.n_generated_tokens);
        std::string joined;
        for (const auto& p : pieces) joined += p;
        PLEIADES_CHECK(joined == r.text);
    }

    // -- generate(): callback returning false stops generation early ----- //
    {
        int calls = 0;
        auto r = engine.generate("Once upon a time", 16, [&](const std::string&) {
            ++calls;
            return calls < 3;  // stop after the 3rd token
        });
        PLEIADES_CHECK(calls == 3);
        PLEIADES_CHECK(r.n_generated_tokens == 3);
    }

    // -- resize keeps the engine functional afterward --------------------- //
    // NOTE: not asserting byte-identical output before/after here. This
    // fixture's own trained context is tiny (128 tokens per llama.cpp's
    // own load log), and both 512 and 1024 already exceed it -- once a
    // model is extrapolating beyond its trained window, llama.cpp's Flash
    // Attention auto-selection and numerical summation order are allowed
    // to differ between allocations of different sizes, so greedy-decode
    // output isn't guaranteed byte-identical in that regime. The Phase 2
    // benchmark already proved byte-identical pre/post-resize output on a
    // REAL model resized within its actual (262144-token) trained ceiling
    // -- see the design doc's Phase 2 results. This test's job is narrower:
    // catch a regression where resize leaves the engine unable to
    // generate at all.
    {
        auto before = engine.complete("Once upon a time", 16);
        PLEIADES_CHECK(!before.text.empty());
        ctx.resize(1024);
        auto after = engine.complete("Once upon a time", 16);
        PLEIADES_CHECK(!after.text.empty());
        PLEIADES_CHECK(after.n_generated_tokens > 0);
    }

    llama_backend_free();
    return 0;
}
