#pragma once

#include <memory>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

// Forward-declared so this header never pulls in llama.cpp's chat.h. That
// matters for a concrete reason, not just hygiene: chat.h does
// `using json = nlohmann::ordered_json;` at GLOBAL scope, whereas the rest of
// this engine (http_server.cpp, tests) does `using json = nlohmann::json;`.
// Letting both aliases into one translation unit is a hard `::json`
// redefinition error. So the entire llama.cpp chat surface -- and the
// ordered_json alias it drags in -- is quarantined inside chat_template.cpp,
// and everything the rest of the engine touches is expressed here in plain
// std types + the engine's own (unordered) nlohmann::json.
struct llama_model;

namespace pleiades_engine {

// One parsed tool call, OpenAI-shape: `arguments` is the JSON-encoded string
// exactly as llama.cpp's parser emits it (ready to drop into a response's
// function.arguments field without re-encoding).
struct ChatToolCall {
    std::string id;
    std::string name;
    std::string arguments;  // JSON-encoded object, as a string
};

// The structured result of parsing a model's completion back out of whatever
// native format its own template specifies. `reasoning_content` is the
// extracted <think>-style reasoning (empty if the model/template has none);
// `tool_calls` is empty for an ordinary final answer.
struct ParsedChatMessage {
    std::string content;
    std::string reasoning_content;
    std::vector<ChatToolCall> tool_calls;
};

// The outcome of rendering ONE request through the model's real jinja
// template. Carries the prompt string AND the parser state (chat format +
// derived PEG parser) needed to turn the model's completion of that prompt
// back into structured fields -- so a single object round-trips render ->
// generate -> parse for one turn. Move-only; hides llama.cpp's common_chat_*
// types (and their ordered_json) behind an opaque pimpl.
class RenderedChat {
public:
    RenderedChat() noexcept;
    RenderedChat(RenderedChat&&) noexcept;
    RenderedChat& operator=(RenderedChat&&) noexcept;
    RenderedChat(const RenderedChat&) = delete;
    RenderedChat& operator=(const RenderedChat&) = delete;
    ~RenderedChat();

    // The fully-rendered prompt to feed the engine.
    const std::string& prompt() const;

    // Extra stop strings this template/format wants honored (e.g. a tool-call
    // closing marker). Callers merge these into their sampling stop list.
    const std::vector<std::string>& additional_stops() const;

    // Parse `generated_text` (the model's completion of prompt()) into
    // structured content/reasoning/tool_calls, using the exact format this
    // render resolved. `is_partial` tolerates a mid-generation prefix; pass
    // false for a finished generation.
    //
    // THROWS std::runtime_error when a finished (is_partial=false) generation
    // does not match the template's format -- e.g. a tool call truncated by
    // max_tokens, or malformed markup. That throw is the fail-loud signal the
    // HTTP layer maps to a distinct 422 (never a 200 that narrates broken
    // tool-call markup as prose) -- see http_server.cpp, and why it matters
    // when Pleiades runs exec_policy "allow".
    ParsedChatMessage parse(const std::string& generated_text, bool is_partial) const;

    struct Impl;
    explicit RenderedChat(Impl* impl) noexcept;  // internal ctor (chat_template.cpp)

private:
    std::unique_ptr<Impl> impl_;
};

// Real per-model chat templating, built once from a loaded GGUF's own
// tokenizer.chat_template via llama.cpp's common_chat_* machinery (the same
// code path real llama-server uses). Replaces the engine's old hardcoded
// Qwen ChatML/XML/JSON formatters and Qwen-only tool-call parser with genuine
// per-family jinja rendering + auto-derived tool-call parsing for ANY model
// whose template llama.cpp understands. One instance per loaded model; hold it
// on ModelManager. Move-only; opaque pimpl (see the llama_model note above).
class ChatTemplates {
public:
    ChatTemplates() noexcept;
    ChatTemplates(ChatTemplates&&) noexcept;
    ChatTemplates& operator=(ChatTemplates&&) noexcept;
    ChatTemplates(const ChatTemplates&) = delete;
    ChatTemplates& operator=(const ChatTemplates&) = delete;
    ~ChatTemplates();

    // Build from a loaded model. `template_override` forces a specific jinja
    // template string; empty (the normal case) reads the model's own GGUF
    // metadata template. A null `model` is allowed ONLY together with a
    // non-empty override (used by the offline unit tests to exercise a known
    // template with no model loaded). THROWS std::runtime_error if no usable
    // template can be built (e.g. the override is malformed jinja).
    static ChatTemplates create(const llama_model* model, const std::string& template_override = "");

    bool valid() const;

    // Which template source llama.cpp resolved (for load-time logging).
    std::string source() const;

    // Whether this model's template advertises tool-call support (jinja caps).
    // When false, tools offered on a request are ignored (see http_server.cpp).
    bool supports_tools() const;

    // Render an OpenAI-shape request. `messages` and `tools` are the raw
    // request arrays (unordered nlohmann::json, straight from the body). Pass
    // an empty/absent `tools` to render without tools. THROWS
    // std::runtime_error on a template evaluation error.
    RenderedChat apply(const nlohmann::json& messages, const nlohmann::json& tools, bool add_generation_prompt,
                       bool enable_thinking) const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace pleiades_engine
