#pragma once

#include <string>
#include <vector>

#include "llama.h"

namespace pleiades_engine {

// Wraps llama_model_load_from_file/llama_model_free. One model resident at
// a time, matching Pleiades' existing single-model-per-process design
// (pleiades/inference/server.py::EngineState). See
// docs/specs/2026-07-21-native-inference-engine-design.md, Phase 1.
class ModelManager {
public:
    ModelManager() = default;
    ~ModelManager();

    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;

    // Loads a GGUF from disk. n_gpu_layers follows llama.cpp's own
    // convention: negative = all layers on GPU, 0 = CPU only.
    //
    // n_cpu_moe > 0 keeps the MoE expert ("ffn_*_exps") weights of the
    // first n_cpu_moe transformer blocks on the CPU regardless of
    // n_gpu_layers -- this is llama.cpp's own `--n-cpu-moe` behavior
    // (common/arg.cpp, common/common.h::llm_ffn_exps_block_regex()/
    // llm_ffn_exps_cpu_override()), replicated here via the public
    // llama_model_params::tensor_buft_overrides field since the engine
    // deliberately doesn't build/link `common/` (see CMakeLists.txt).
    //
    // use_mlock mirrors llama_model_params::use_mlock (llama-server's
    // --mlock): force the OS to keep the model resident in RAM rather than
    // swapping/compressing it.
    //
    // Throws std::runtime_error on failure.
    void load(const std::string& path, int n_gpu_layers = 0, int n_cpu_moe = 0, bool use_mlock = false);

    void unload();

    bool is_loaded() const { return model_ != nullptr; }
    llama_model* model() const { return model_; }
    const std::string& path() const { return path_; }

private:
    llama_model* model_ = nullptr;
    std::string path_;

    // Storage for the tensor_buft_overrides regex patterns built by
    // load()'s n_cpu_moe handling. llama_model_params::tensor_buft_overrides
    // holds raw `const char*` pointers (include/llama.h) that must stay
    // alive for the duration of the llama_model_load_from_file() call --
    // these are member storage (not stack temporaries) for exactly that
    // reason, cleared and rebuilt on every load()/unload() call.
    std::vector<std::string> moe_override_patterns_;
    std::vector<llama_model_tensor_buft_override> moe_overrides_;
};

}  // namespace pleiades_engine
