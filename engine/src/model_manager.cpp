#include "pleiades_engine/model_manager.h"

#include <cstdio>
#include <stdexcept>
#include <utility>

namespace pleiades_engine {

namespace {

// Replicates third_party/llama.cpp's common/common.h::LLM_FFN_EXPS_REGEX /
// llm_ffn_exps_block_regex() verbatim (same regex string, same
// "blk\\.<N><regex>" pattern format) -- see common/arg.cpp's "--n-cpu-moe"
// (`-ncmoe`) option handler for the real upstream logic this mirrors. The
// engine deliberately doesn't build `common/` (engine/CMakeLists.txt), so
// this ~15-line piece is replicated directly here instead of linking that
// whole target in just for this.
constexpr const char* kFfnExpsRegex = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

std::string ffn_exps_block_regex(int layer_idx) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "blk\\.%d%s", layer_idx, kFfnExpsRegex);
    return std::string(buf);
}

}  // namespace

ModelManager::~ModelManager() { unload(); }

void ModelManager::load(const std::string& path, int n_gpu_layers, int n_cpu_moe, bool use_mlock) {
    unload();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    params.use_mlock = use_mlock;

    if (n_cpu_moe > 0) {
        // Build the per-layer override list BEFORE taking any c_str()
        // pointers below -- a std::vector<std::string> reallocation while
        // appending would invalidate any c_str() pointer taken from an
        // earlier element, which is exactly the lifetime bug the council
        // flagged for this field. moe_override_patterns_ is fully populated
        // first, then moe_overrides_ is built pointing at its (now stable)
        // elements.
        moe_override_patterns_.clear();
        moe_overrides_.clear();
        moe_override_patterns_.reserve(static_cast<size_t>(n_cpu_moe));
        for (int i = 0; i < n_cpu_moe; ++i) {
            moe_override_patterns_.push_back(ffn_exps_block_regex(i));
        }
        moe_overrides_.reserve(static_cast<size_t>(n_cpu_moe) + 1);
        for (const auto& pattern : moe_override_patterns_) {
            moe_overrides_.push_back({pattern.c_str(), ggml_backend_cpu_buffer_type()});
        }
        // NULL-terminated, per include/llama.h's llama_model_params doc
        // comment on tensor_buft_overrides.
        moe_overrides_.push_back({nullptr, nullptr});
        params.tensor_buft_overrides = moe_overrides_.data();
    } else {
        moe_override_patterns_.clear();
        moe_overrides_.clear();
    }

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
    moe_override_patterns_.clear();
    moe_overrides_.clear();
}

}  // namespace pleiades_engine
