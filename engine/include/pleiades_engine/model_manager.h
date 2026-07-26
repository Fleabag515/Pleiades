#pragma once

#include <string>
#include <vector>

#include "llama.h"
#include "pleiades_engine/chat_template.h"

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
    // Also detects this model's own tool-calling dialect (see
    // tool_dialect()/open_thinking() below) by reading its
    // tokenizer.chat_template metadata -- once, here, not re-sniffed per
    // request.
    //
    // Throws std::runtime_error on failure.
    // statewise_map (optional): path to a routing-profile file enabling the statewise
    // MoE expert cache (per-layer GPU hot-expert copies). Applied after load via
    // llama_model_statewise_init(). Only meaningful when the cached layers' experts are
    // CPU-offloaded (n_cpu_moe covering them) so the cold chain stays on CPU -- otherwise
    // the CUDA placement tripwire will abort. Empty = disabled. See docs Phase 7.
    void load(const std::string& path, int n_gpu_layers = 0, int n_cpu_moe = 0, bool use_mlock = false,
              const std::string& statewise_map = "");

    void unload();

    bool is_loaded() const { return model_ != nullptr; }
    llama_model* model() const { return model_; }
    const std::string& path() const { return path_; }

    // Which tool-calling text dialect this model's own chat template
    // expects/emits (see chat_template.h::ToolDialect), detected once at
    // load() time by reading tokenizer.chat_template via
    // llama_model_meta_val_str() and sniffing it with detect_tool_dialect()
    // -- never guessed from the model's name/path. ToolDialect::NONE
    // (including before any model is loaded) means "no tool support for
    // this model" -- callers must fall back to plain completion behavior,
    // not attempt to parse tool calls that will never come.
    ToolDialect tool_dialect() const { return tool_dialect_; }

    // Whether this model's template defaults to an OPEN `<think>\n`
    // generation-prompt suffix (see detect_open_thinking()). Only
    // meaningful when tool_dialect() != NONE, but valid (false) regardless.
    bool open_thinking() const { return open_thinking_; }

    // This model's own raw tokenizer.chat_template metadata string (empty if
    // the GGUF carries none -- e.g. a base/non-instruct model). Read once at
    // load(). Used by the ToolDialect::NONE path to format via
    // llama_chat_apply_template() (its actual template family) instead of
    // assuming Qwen ChatML for every non-Qwen model -- see
    // apply_builtin_template() in chat_template.h and http_server.cpp's
    // handle_chat(). The Qwen XML/JSON dialects don't consult this (they have
    // their own hardcoded renderers), so it's only the cross-family fallback.
    const std::string& chat_template() const { return chat_template_; }

private:
    llama_model* model_ = nullptr;
    std::string path_;
    ToolDialect tool_dialect_ = ToolDialect::NONE;
    bool open_thinking_ = false;
    std::string chat_template_;

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
