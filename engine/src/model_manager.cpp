#include "pleiades_engine/model_manager.h"

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pleiades_engine {

namespace {

// Replicates third_party/llama.cpp's common/common.h::LLM_FFN_EXPS_REGEX /
// llm_ffn_exps_block_regex() verbatim (same regex string, same
// "blk\\.<N><regex>" pattern format) -- see common/arg.cpp's "--n-cpu-moe"
// (`-ncmoe`) option handler for the real upstream logic this mirrors. The
// engine deliberately doesn't build `common/` wholesale (engine/CMakeLists.txt),
// so this ~15-line piece is replicated directly here instead of linking that
// whole target in just for this.
constexpr const char* kFfnExpsRegex = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

std::string ffn_exps_block_regex(int layer_idx) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "blk\\.%d%s", layer_idx, kFfnExpsRegex);
    return std::string(buf);
}

}  // namespace

ModelManager::~ModelManager() { unload(); }

void ModelManager::load(const std::string& path, int n_gpu_layers, int n_cpu_moe, bool use_mlock,
                        const std::string& statewise_map, bool use_mmap, bool use_extra_bufts) {
    unload();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    params.use_mlock = use_mlock;
    params.use_mmap = use_mmap;
    // false = llama-server's --no-repack: keep quantized weights in their
    // on-disk layout under mmap instead of materializing a repacked copy
    // (which would copy them into anonymous RAM and defeat lazy paging --
    // see use_extra_bufts in model_manager.h, upstream issue #19578).
    params.use_extra_bufts = use_extra_bufts;

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

    // statewise expert cache: TEMPORARILY UNAVAILABLE as of the 2026-07-23
    // upstream rebase (chore/llamacpp-rebase). The static MoE expert-cache
    // mechanism ("Statewise v1", commit 17a6fbc72) lived on the Fleabag515
    // llama.cpp fork and provided llama_model_statewise_init(). That fork is
    // now off the submodule path (repointed to plain upstream ggerganov at
    // tag b10103), so the symbol no longer exists. Phase 1 (not yet started)
    // will rebuild a generalized, cross-architecture MoE expert-offload
    // mechanism against this new post-rewrite base -- not by re-porting the
    // old patch. Until then, statewise is a no-op if unrequested and a loud,
    // honest error if requested, rather than a silent forfeit of a speedup
    // the caller believes they are getting. See the rebase section of
    // docs/specs/2026-07-21-native-inference-engine-design.md.
    if (!statewise_map.empty()) {
        throw std::runtime_error(
            "pleiades_engine: statewise expert cache is temporarily unavailable after the "
            "2026-07-23 upstream llama.cpp rebase (the Fleabag515 fork's llama_model_statewise_init "
            "was intentionally dropped); Phase 1 will rebuild it generalized. Requested map: " +
            statewise_map);
    }

    // Build the model's real chat templates from its own GGUF jinja template.
    // A model that carries no usable template (or one whose jinja fails to
    // parse) must still load: llama.cpp's built-in ChatML default covers that
    // exactly the way the old hardcoded format_chatml() stopgap did, so this
    // never turns a template quirk into a failed model load. common_chat_*'s
    // "chatml" override is init()'s own documented alias for that built-in.
    try {
        chat_templates_ = ChatTemplates::create(model_, /*template_override=*/"");
    } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "[pleiades-engine] WARNING: model's own chat template failed to build (%s); falling back to "
                     "llama.cpp's built-in ChatML template. Tool-calling may be unavailable for this model.\n",
                     e.what());
        chat_templates_ = ChatTemplates::create(model_, /*template_override=*/"chatml");
    }
}

void ModelManager::unload() {
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    path_.clear();
    chat_templates_ = ChatTemplates{};
    moe_override_patterns_.clear();
    moe_overrides_.clear();
}

}  // namespace pleiades_engine
