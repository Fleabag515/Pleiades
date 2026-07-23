#include "pleiades_engine/chat_template.h"

namespace pleiades_engine {

using json = nlohmann::json;

namespace {

std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return "";
    }
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// -- dialect instructional text -------------------------------------------
// Pulled byte-for-byte from real GGUFs' own tokenizer.chat_template (read
// via llama_model_meta_val_str, not paraphrased/guessed), 2026-07-23:
//   XML:  empero-ai/Qwythos-9B-Claude-Mythos-5-1M (arch qwen35) and
//         HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive (arch
//         qwen35moe) -- byte-identical between the two independent samples.
//   JSON: Qwen/Qwen2.5-0.5B-Instruct (arch qwen2). Rootkit7's
//         Ternary-Bonsai-4B (arch qwen3, dense) carries the same
//         instructional text except with single braces around
//         `{"name": ...}` where Qwen2.5's own template has doubled braces
//         (`{{"name": ...}}` -- confirmed via a hex dump of the raw stored
//         bytes, not a transcription slip; almost certainly a copy-paste
//         artifact from however Qwen's own team authored the template).
//         Qwen2.5's exact text is used here since it's the more literal
//         reading of "qwen2/qwen3" from the design review, and it's what
//         real unmodified servers (llama.cpp --jinja, vLLM, HF
//         apply_chat_template) actually send this model family today. This
//         is purely instructional/cosmetic text shown in the system prompt
//         -- the ACTUAL output contract both samples train the model to
//         produce (their assistant-turn rendering, not this instructional
//         blurb) uses correct single-brace JSON either way, which is all
//         tool_call_parser.cpp needs to parse.
constexpr const char* kXmlToolsHead = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
constexpr const char* kXmlToolsTail =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple "
    "lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be "
    "nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function "
    "call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge "
    "and do not tell the user about function calls\n</IMPORTANT>";

constexpr const char* kJsonToolsHead =
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
constexpr const char* kJsonToolsTail =
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": "
    "<args-json-object>}}\n</tool_call>";

std::string render_tools_block(const std::vector<json>& tools, const char* head, const char* tail) {
    std::string out = head;
    for (const auto& t : tools) {
        out += "\n";
        out += t.dump();  // compact JSON, matching the templates' own `tool | tojson`
    }
    out += tail;
    return out;
}

// Renders `calls` into the dialect's own native tool-call text, appended
// after `content` (already written by the caller) -- matching each real
// template's own assistant-turn jinja logic (see chat_template.h). `calls`
// come from history (ChatMessage::tool_calls), not from a fresh parse.
void render_assistant_tool_calls(std::string& out, const std::string& content, const std::vector<ToolCall>& calls,
                                  ToolDialect dialect) {
    for (size_t i = 0; i < calls.size(); ++i) {
        const ToolCall& call = calls[i];
        if (dialect == ToolDialect::QWEN_XML) {
            if (i == 0) {
                out += content.empty() ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=";
            } else {
                out += "\n<tool_call>\n<function=";
            }
            out += call.name;
            out += ">\n";
            if (call.arguments.is_object()) {
                for (auto it = call.arguments.begin(); it != call.arguments.end(); ++it) {
                    out += "<parameter=";
                    out += it.key();
                    out += ">\n";
                    // The template renders string-typed values raw and
                    // everything else via `| tojson` -- mirrored here by
                    // just checking the JSON value's own type, since a
                    // history tool_call's arguments were already parsed
                    // (by parse_chat_messages) into properly-typed JSON.
                    out += it.value().is_string() ? it.value().get<std::string>() : it.value().dump();
                    out += "\n</parameter>\n";
                }
            }
            out += "</function>\n</tool_call>";
        } else {  // QWEN_JSON -- every call (including the first) gets an
                   // unconditional leading '\n', matching Qwen2.5's own
                   // unconditional-per-iteration template logic exactly
                   // (unlike the XML dialect's content-conditional first
                   // call -- a real, checked difference, not an oversight).
            out += "\n<tool_call>\n{\"name\": \"";
            out += call.name;
            out += "\", \"arguments\": ";
            out += (call.arguments.is_object() ? call.arguments : json::object()).dump();
            out += "}\n</tool_call>";
        }
    }
}

void render_assistant_message(std::string& out, const ChatMessage& m, ToolDialect dialect) {
    bool is_xml = dialect == ToolDialect::QWEN_XML;
    std::string content = is_xml ? trim(m.content) : m.content;
    std::string reasoning = trim(m.reasoning_content);

    out += "<|im_start|>assistant";
    if (!reasoning.empty()) {
        out += "\n<think>\n";
        out += reasoning;
        out += "\n</think>\n\n";
        out += content;
    } else if (is_xml || !content.empty()) {
        // XML dialect always emits the newline+content pair (even when
        // content is empty); JSON dialect omits it entirely when content
        // is empty -- a real, checked difference between the two real
        // templates' assistant branches, not an oversight.
        out += "\n";
        out += content;
    }
    render_assistant_tool_calls(out, content, m.tool_calls, dialect);
    out += "<|im_end|>\n";
}

// Renders a maximal run of consecutive role=="tool" messages starting at
// `i` as one merged `<|im_start|>user ... <|im_end|>` turn -- both real
// dialects render tool-role messages identically (only `content`
// trimming differs, matching each dialect's own per-message convention).
// Returns the index just past the run.
size_t render_tool_run(std::string& out, const std::vector<ChatMessage>& messages, size_t i, ToolDialect dialect) {
    bool is_xml = dialect == ToolDialect::QWEN_XML;
    out += "<|im_start|>user";
    size_t j = i;
    while (j < messages.size() && messages[j].role == "tool") {
        out += "\n<tool_response>\n";
        out += is_xml ? trim(messages[j].content) : messages[j].content;
        out += "\n</tool_response>";
        ++j;
    }
    out += "<|im_end|>\n";
    return j;
}

// -- null-safe JSON field readers, shared by parse_chat_messages ----------
//
// nlohmann's `.value(key, default)` only substitutes `default` for a
// MISSING key -- a PRESENT key holding JSON `null` still reaches
// `.get<std::string>()`, which throws `type_error` 302. See
// chat_template.h's doc comment on parse_chat_messages() for why this is a
// real, live bug (not a hypothetical one) for this specific feature.
std::string json_str_or(const json& obj, const char* key, const std::string& fallback = "") {
    if (!obj.is_object() || !obj.contains(key) || obj[key].is_null()) {
        return fallback;
    }
    const json& v = obj[key];
    return v.is_string() ? v.get<std::string>() : fallback;
}

// `function.arguments` on an inbound tool_calls entry is, per the OpenAI
// spec (and what this engine itself emits -- see http_server.cpp), a JSON-
// encoded STRING. Some non-spec-compliant clients send a raw object
// instead; tolerate both, matching pleiades/harness/llm.py's own
// `isinstance(args, str)` leniency on the Python side.
json parse_arguments_field(const json& fn) {
    if (!fn.is_object() || !fn.contains("arguments")) {
        return json::object();
    }
    const json& a = fn["arguments"];
    if (a.is_object()) {
        return a;
    }
    if (a.is_string()) {
        try {
            return json::parse(a.get<std::string>());
        } catch (const std::exception&) {
            return json::object();
        }
    }
    return json::object();
}

}  // namespace

const char* tool_dialect_name(ToolDialect d) {
    switch (d) {
        case ToolDialect::QWEN_XML:
            return "QWEN_XML";
        case ToolDialect::QWEN_JSON:
            return "QWEN_JSON";
        default:
            return "NONE";
    }
}

ToolDialect detect_tool_dialect(const std::string& chat_template) {
    if (chat_template.find("<function=") != std::string::npos) {
        return ToolDialect::QWEN_XML;
    }
    if (chat_template.find("<tool_call>") != std::string::npos &&
        (chat_template.find("{\"name\"") != std::string::npos ||
         chat_template.find("{{\"name\"") != std::string::npos)) {
        return ToolDialect::QWEN_JSON;
    }
    return ToolDialect::NONE;
}

bool detect_open_thinking(const std::string& chat_template) {
    return chat_template.find("enable_thinking is defined and enable_thinking is false") != std::string::npos;
}

std::string format_chatml(const std::vector<ChatMessage>& messages) {
    std::string out;
    for (const auto& m : messages) {
        out += "<|im_start|>";
        out += m.role;
        out += '\n';
        out += m.content;
        out += "<|im_end|>\n";
    }
    out += "<|im_start|>assistant\n";
    return out;
}

std::string format_chat_prompt(const std::vector<ChatMessage>& messages, const std::vector<json>& tools,
                                ToolDialect dialect, bool open_thinking) {
    if (dialect == ToolDialect::NONE) {
        return format_chatml(messages);
    }

    bool is_xml = dialect == ToolDialect::QWEN_XML;
    bool have_tools = !tools.empty();
    bool first_is_system = !messages.empty() && messages[0].role == "system";

    std::string out;

    // -- system / tools preamble (rendered once, up front) -----------------
    if (have_tools) {
        out += "<|im_start|>system\n";
        if (is_xml) {
            out += render_tools_block(tools, kXmlToolsHead, kXmlToolsTail);
            if (first_is_system) {
                std::string sys = trim(messages[0].content);
                if (!sys.empty()) {
                    out += "\n\n";
                    out += sys;
                }
            }
        } else {
            // JSON dialect orders system content BEFORE the tools block --
            // the opposite order from XML -- a real, checked difference.
            if (first_is_system && !messages[0].content.empty()) {
                out += messages[0].content;
                out += "\n\n";
            }
            out += render_tools_block(tools, kJsonToolsHead, kJsonToolsTail);
        }
        out += "<|im_end|>\n";
    } else if (first_is_system) {
        out += "<|im_start|>system\n";
        out += is_xml ? trim(messages[0].content) : messages[0].content;
        out += "<|im_end|>\n";
    }

    // -- remaining messages --------------------------------------------------
    for (size_t i = first_is_system ? 1 : 0; i < messages.size();) {
        const ChatMessage& m = messages[i];
        if (m.role == "tool") {
            i = render_tool_run(out, messages, i, dialect);
            continue;
        }
        if (m.role == "assistant") {
            render_assistant_message(out, m, dialect);
        } else {
            out += "<|im_start|>";
            out += m.role;
            out += "\n";
            out += is_xml ? trim(m.content) : m.content;
            out += "<|im_end|>\n";
        }
        ++i;
    }

    // -- generation prompt ---------------------------------------------------
    out += "<|im_start|>assistant\n";
    if (open_thinking) {
        out += "<think>\n";
    }
    return out;
}

std::vector<ChatMessage> parse_chat_messages(const json& body) {
    std::vector<ChatMessage> out;
    if (!body.contains("messages") || !body["messages"].is_array()) {
        return out;
    }
    for (const auto& m : body["messages"]) {
        ChatMessage cm;
        cm.role = json_str_or(m, "role", "user");
        cm.content = json_str_or(m, "content", "");

        if (cm.role == "assistant") {
            cm.reasoning_content = json_str_or(m, "reasoning_content", "");
            if (m.is_object() && m.contains("tool_calls") && m["tool_calls"].is_array()) {
                for (const auto& tc : m["tool_calls"]) {
                    if (!tc.is_object() || !tc.contains("function") || !tc["function"].is_object()) {
                        continue;
                    }
                    ToolCall call;
                    call.name = json_str_or(tc["function"], "name", "");
                    call.arguments = parse_arguments_field(tc["function"]);
                    cm.tool_calls.push_back(std::move(call));
                }
            }
        } else if (cm.role == "tool") {
            cm.tool_call_id = json_str_or(m, "tool_call_id", "");
            cm.tool_name = json_str_or(m, "name", "");
        }

        out.push_back(std::move(cm));
    }
    return out;
}

}  // namespace pleiades_engine
