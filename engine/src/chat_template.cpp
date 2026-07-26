// Real per-model chat templating, implemented over llama.cpp's own
// common_chat_* API (common/chat.cpp + the jinja engine + the auto-derived
// PEG tool-call parser, all compiled into the `pleiades_chat` static lib --
// see engine/CMakeLists.txt). This is the ONE translation unit in the engine
// that includes llama.cpp's chat.h, on purpose: chat.h does
// `using json = nlohmann::ordered_json;` at global scope, which collides with
// the `using json = nlohmann::json;` the rest of the engine uses. Keeping
// chat.h out of chat_template.h (opaque pimpl) confines that alias -- and the
// entire common_chat_* surface -- to this file.
//
// The flow this wraps mirrors real llama-server exactly (tools/server/
// server-common.cpp / server-task.cpp):
//   init : common_chat_templates_init(model)            -- once per model
//   apply: common_chat_templates_apply(tmpls, inputs)   -- once per request,
//          yields common_chat_params (prompt + resolved format + a serialized
//          PEG parser + additional stops + generation prompt)
//   parse: common_chat_parse(text, is_partial, params)  -- turns the model's
//          completion back into content / reasoning_content / tool_calls using
//          that same resolved format
//
// The OpenAI-shape messages/tools JSON the engine receives is unordered
// nlohmann::json; common_chat_*'s oaicompat parsers take ordered_json. The two
// are distinct nlohmann instantiations of the same vendored header, so they
// coexist fine; conversion is a dump()/parse() round-trip, done here.

#include "pleiades_engine/chat_template.h"

#include "chat.h"

#include <exception>
#include <utility>

namespace pleiades_engine {

namespace {

// Unordered engine json -> ordered_json (what common_chat_*'s oaicompat
// parsers consume). A missing/non-array value becomes an empty array so the
// parsers see a well-formed (empty) list rather than throwing.
nlohmann::ordered_json to_ordered_array(const nlohmann::json& j) {
    if (!j.is_array()) {
        return nlohmann::ordered_json::array();
    }
    return nlohmann::ordered_json::parse(j.dump());
}

// The reasoning format the engine renders + parses with. AUTO is llama.cpp's
// own recommended default: it extracts a model's <think>-style reasoning into
// the separate `reasoning_content` field (rather than inlining it in content),
// which is exactly the shape this engine's HTTP responses already expose.
constexpr common_reasoning_format kReasoningFormat = COMMON_REASONING_FORMAT_AUTO;

}  // namespace

// -- RenderedChat -----------------------------------------------------------

struct RenderedChat::Impl {
    common_chat_params params;  // prompt + format + serialized parser + stops
};

RenderedChat::RenderedChat() noexcept = default;
RenderedChat::RenderedChat(Impl* impl) noexcept : impl_(impl) {}
RenderedChat::RenderedChat(RenderedChat&&) noexcept = default;
RenderedChat& RenderedChat::operator=(RenderedChat&&) noexcept = default;
RenderedChat::~RenderedChat() = default;

const std::string& RenderedChat::prompt() const { return impl_->params.prompt; }

const std::vector<std::string>& RenderedChat::additional_stops() const { return impl_->params.additional_stops; }

ParsedChatMessage RenderedChat::parse(const std::string& generated_text, bool is_partial) const {
    common_chat_parser_params pp;
    pp.format = impl_->params.format;
    pp.generation_prompt = impl_->params.generation_prompt;
    pp.reasoning_format = kReasoningFormat;
    pp.parse_tool_calls = true;
    // apply() serializes the model-specific PEG parser into params.parser;
    // rehydrate it here (empty => the content-only default parser). This is
    // the exact string round-trip real llama-server does across its task queue
    // (server-common.cpp writes chat_params.parser; server-schema.cpp reloads
    // it via parser.load()); in-process it's a no-op-cheap re-parse.
    if (!impl_->params.parser.empty()) {
        pp.parser.load(impl_->params.parser);
    }

    // Throws std::runtime_error on a finished generation that doesn't match the
    // format (truncated/malformed tool call). Deliberately NOT swallowed here
    // -- http_server.cpp turns it into the fail-loud 422. See the header.
    common_chat_msg msg = common_chat_parse(generated_text, is_partial, pp);

    ParsedChatMessage out;
    out.content = msg.content;
    out.reasoning_content = msg.reasoning_content;
    out.tool_calls.reserve(msg.tool_calls.size());
    for (const auto& tc : msg.tool_calls) {
        out.tool_calls.push_back(ChatToolCall{tc.id, tc.name, tc.arguments});
    }
    return out;
}

// -- ChatTemplates ----------------------------------------------------------

struct ChatTemplates::Impl {
    common_chat_templates_ptr tmpls;
    bool supports_tools = false;
};

ChatTemplates::ChatTemplates() noexcept = default;
ChatTemplates::ChatTemplates(ChatTemplates&&) noexcept = default;
ChatTemplates& ChatTemplates::operator=(ChatTemplates&&) noexcept = default;
ChatTemplates::~ChatTemplates() = default;

ChatTemplates ChatTemplates::create(const llama_model* model, const std::string& template_override) {
    ChatTemplates ct;
    ct.impl_ = std::make_unique<Impl>();
    // May throw if the (override or GGUF) template is malformed jinja.
    ct.impl_->tmpls = common_chat_templates_init(model, template_override);

    const auto caps = common_chat_templates_get_caps(ct.impl_->tmpls.get());
    const auto it = caps.find("supports_tool_calls");
    ct.impl_->supports_tools = (it != caps.end() && it->second);
    return ct;
}

bool ChatTemplates::valid() const { return impl_ != nullptr && impl_->tmpls != nullptr; }

std::string ChatTemplates::source() const {
    if (!valid()) {
        return {};
    }
    return common_chat_templates_source(impl_->tmpls.get());
}

bool ChatTemplates::supports_tools() const { return valid() && impl_->supports_tools; }

RenderedChat ChatTemplates::apply(const nlohmann::json& messages, const nlohmann::json& tools,
                                  bool add_generation_prompt, bool enable_thinking) const {
    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(to_ordered_array(messages));
    if (tools.is_array() && !tools.empty()) {
        inputs.tools = common_chat_tools_parse_oaicompat(to_ordered_array(tools));
    }
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.use_jinja = true;
    inputs.enable_thinking = enable_thinking;
    inputs.reasoning_format = kReasoningFormat;

    auto impl = std::make_unique<RenderedChat::Impl>();
    // Throws std::runtime_error on a template evaluation error.
    impl->params = common_chat_templates_apply(impl_->tmpls.get(), inputs);
    return RenderedChat(impl.release());
}

}  // namespace pleiades_engine
