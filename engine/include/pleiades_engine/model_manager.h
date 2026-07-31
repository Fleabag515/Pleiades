#pragma once

#include <string>
#include <vector>

#include "llama.h"
#include "pleiades_engine/chat_template.h"

// Opaque forward declaration -- callers that need the real mtmd API (e.g.
// http_server.cpp calling mtmd_tokenize()) #include "mtmd.h" themselves;
// this header only needs a pointer type. Safe regardless of mtmd.h's own
// `extern "C" { typedef struct mtmd_context mtmd_context; }` -- the struct
// tag `mtmd_context` names the same type either way, extern "C" affects
// symbol linkage, not type identity.
struct mtmd_context;

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
    // deliberately doesn't build/link `common/` wholesale (see CMakeLists.txt).
    //
    // use_mlock mirrors llama_model_params::use_mlock (llama-server's
    // --mlock): force the OS to keep the model resident in RAM rather than
    // swapping/compressing it.
    //
    // Also builds this model's real chat templates (see chat_templates()) from
    // its own GGUF-embedded jinja template -- once, here, not per request.
    //
    // Throws std::runtime_error on failure.
    // statewise_map (optional): path to a routing-profile file enabling the statewise
    // MoE expert cache (per-layer GPU hot-expert copies). Applied after load via
    // llama_model_statewise_init(). Only meaningful when the cached layers' experts are
    // CPU-offloaded (n_cpu_moe covering them) so the cold chain stays on CPU -- otherwise
    // the CUDA placement tripwire will abort. Empty = disabled. See docs Phase 7.
    // mmproj_path (optional, Phase 9.4.2): path to a multimodal-projector GGUF.
    // When set, initializes libmtmd (mtmd_init_from_file) alongside the text
    // model so vision-capable requests can be served -- mtmd() returns
    // non-null only when this succeeded. Throws if the file is given but
    // fails to load (an explicitly-requested capability that silently isn't
    // there is worse than a clean error at startup); a model with no mmproj
    // and no vision request needs it at all, so it's opt-in, not probed.
    void load(const std::string& path, int n_gpu_layers = 0, int n_cpu_moe = 0, bool use_mlock = false,
              const std::string& statewise_map = "", const std::string& mmproj_path = "");

    void unload();

    bool is_loaded() const { return model_ != nullptr; }
    llama_model* model() const { return model_; }
    const std::string& path() const { return path_; }

    // This model's real per-model chat-templating engine, built once at load()
    // from its own tokenizer.chat_template via llama.cpp's common_chat_*
    // machinery (see chat_template.h). Used by http_server.cpp to render every
    // request's prompt AND parse the model's completion back into structured
    // content/reasoning/tool_calls -- for ANY model family whose template
    // llama.cpp understands, not just the two Qwen dialects the engine
    // previously hand-coded. Always valid after a successful load() (a model
    // carrying no usable template falls back to llama.cpp's built-in ChatML).
    const ChatTemplates& chat_templates() const { return chat_templates_; }

    // Non-null only after a successful load() with mmproj_path set. Owned by
    // this ModelManager -- freed in unload()/~ModelManager, never by a caller.
    mtmd_context* mtmd() const { return mtmd_; }

private:
    llama_model* model_ = nullptr;
    std::string path_;
    ChatTemplates chat_templates_;
    mtmd_context* mtmd_ = nullptr;

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
