#include "pleiades_engine/model_manager.h"

#include <stdexcept>
#include <utility>

namespace pleiades_engine {

ModelManager::~ModelManager() { unload(); }

void ModelManager::load(const std::string& path, int n_gpu_layers) {
    unload();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    model_ = llama_model_load_from_file(path.c_str(), params);
    if (!model_) {
        throw std::runtime_error("pleiades_engine: failed to load model: " + path);
    }
    path_ = path;
}

void ModelManager::unload() {
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    path_.clear();
}

}  // namespace pleiades_engine
