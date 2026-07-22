#pragma once

#include <string>

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
    // Throws std::runtime_error on failure.
    void load(const std::string& path, int n_gpu_layers = 0);

    void unload();

    bool is_loaded() const { return model_ != nullptr; }
    llama_model* model() const { return model_; }
    const std::string& path() const { return path_; }

private:
    llama_model* model_ = nullptr;
    std::string path_;
};

}  // namespace pleiades_engine
