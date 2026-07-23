// Requires the tiny fixture GGUF (see tests/CMakeLists.txt) as argv[1].
#include "pleiades_engine/model_manager.h"

#include <stdexcept>
#include <string>

#include "llama.h"
#include "test_util.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    std::string model_path = argv[1];

    {
        pleiades_engine::ModelManager m;
        PLEIADES_CHECK(!m.is_loaded());
        m.load(model_path, /*n_gpu_layers=*/0);
        PLEIADES_CHECK(m.is_loaded());
        PLEIADES_CHECK(m.model() != nullptr);
        PLEIADES_CHECK(m.path() == model_path);

        m.unload();
        PLEIADES_CHECK(!m.is_loaded());
        PLEIADES_CHECK(m.model() == nullptr);
        PLEIADES_CHECK(m.path().empty());
    }

    // Loading a second time (after unload, same object) works -- exercises
    // the "one resident model at a time" contract, not just first-load.
    {
        pleiades_engine::ModelManager m;
        m.load(model_path, 0);
        PLEIADES_CHECK(m.is_loaded());
        m.load(model_path, 0);  // reload while already loaded -- must not leak/crash
        PLEIADES_CHECK(m.is_loaded());
    }

    // A bad path must throw, not crash or silently report loaded.
    {
        pleiades_engine::ModelManager m;
        bool threw = false;
        try {
            m.load("/nonexistent/path/does-not-exist.gguf", 0);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        PLEIADES_CHECK(threw);
        PLEIADES_CHECK(!m.is_loaded());
    }

    // -- n_cpu_moe (MoE expert CPU offload) ------------------------------- //
    // Replicates llama.cpp's own "--n-cpu-moe" (common/arg.cpp) via
    // llama_model_params::tensor_buft_overrides (see model_manager.h/.cpp).
    // The tiny fixture used here (tinyllamas/stories15M) is NOT a MoE model,
    // so the per-layer "blk.N.\\.ffn_*_exps" regexes won't match any real
    // tensor -- that mirrors upstream's own behavior (--n-cpu-moe on a dense
    // model is a documented no-op, not an error, since the regex simply
    // matches nothing). This still exercises the real code path end-to-end:
    // pattern generation, tensor_buft_overrides construction/NULL-
    // termination, and passing it through llama_model_load_from_file
    // without the lifetime bug the council flagged (moe_override_patterns_/
    // moe_overrides_ must be member storage, not stack temporaries that go
    // out of scope before the load call reads them).
    {
        pleiades_engine::ModelManager m;
        m.load(model_path, /*n_gpu_layers=*/0, /*n_cpu_moe=*/8);
        PLEIADES_CHECK(m.is_loaded());
        m.unload();
        PLEIADES_CHECK(!m.is_loaded());
    }

    // n_cpu_moe larger than the model's real layer count must still be
    // safe -- upstream's own --n-cpu-moe has no such bound-check either;
    // extra per-layer regexes for nonexistent blk indices simply never
    // match anything.
    {
        pleiades_engine::ModelManager m;
        m.load(model_path, /*n_gpu_layers=*/0, /*n_cpu_moe=*/1000);
        PLEIADES_CHECK(m.is_loaded());
    }

    // use_mlock=true must not crash even if the OS denies the lock (e.g. a
    // low RLIMIT_MEMLOCK ulimit in a sandboxed/CI environment) --
    // llama.cpp's own llama_mlock only logs a warning on failure
    // (third_party/llama.cpp/src/llama-mmap.cpp), it never throws.
    {
        pleiades_engine::ModelManager m;
        m.load(model_path, /*n_gpu_layers=*/0, /*n_cpu_moe=*/0, /*use_mlock=*/true);
        PLEIADES_CHECK(m.is_loaded());
    }

    // Reloading the same object with a different n_cpu_moe must not leak or
    // retain stale override entries from the previous load (moe_overrides_/
    // moe_override_patterns_ are cleared at the top of every load()).
    {
        pleiades_engine::ModelManager m;
        m.load(model_path, 0, /*n_cpu_moe=*/4);
        PLEIADES_CHECK(m.is_loaded());
        m.load(model_path, 0, /*n_cpu_moe=*/0);
        PLEIADES_CHECK(m.is_loaded());
    }

    llama_backend_free();
    return 0;
}
