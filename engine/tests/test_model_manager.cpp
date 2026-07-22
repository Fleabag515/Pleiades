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

    llama_backend_free();
    return 0;
}
