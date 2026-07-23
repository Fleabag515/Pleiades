#include "pleiades_engine/tool_call_parser.h"

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

// Strips exactly one structural newline (the one the template itself
// inserts immediately around a <parameter=...> value -- see
// chat_template.cpp's render_assistant_tool_calls, which inserts exactly
// this shape) from the front or back of `s`. Deliberately NOT a generic
// trim: a legitimate multi-line parameter value (e.g. a write_file tool's
// `content` spanning many lines) must keep its own internal newlines --
// and, in principle, its own meaningful leading/trailing whitespace --
// untouched. Only the ONE delimiter newline the wire format itself adds is
// removed.
std::string strip_one_edge_newline(const std::string& s, bool from_front) {
    if (from_front) {
        if (s.size() >= 2 && s[0] == '\r' && s[1] == '\n') return s.substr(2);
        if (!s.empty() && s[0] == '\n') return s.substr(1);
        return s;
    }
    if (s.size() >= 2 && s[s.size() - 2] == '\r' && s[s.size() - 1] == '\n') return s.substr(0, s.size() - 2);
    if (!s.empty() && s.back() == '\n') return s.substr(0, s.size() - 1);
    return s;
}

bool is_offered(const std::string& name, const std::vector<json>& offered_tools) {
    for (const auto& t : offered_tools) {
        if (t.is_object() && t.contains("function") && t["function"].is_object() &&
            t["function"].value("name", std::string()) == name) {
            return true;
        }
    }
    return false;
}

// The offered tool's own JSON-schema `properties` object for `name`, or an
// empty object if `name` isn't offered at all (in which case the anchor
// check in parse_tool_calls() rejects the whole call regardless -- this
// just has to not crash while that happens).
json schema_properties_for(const std::string& name, const std::vector<json>& offered_tools) {
    for (const auto& t : offered_tools) {
        if (!t.is_object() || !t.contains("function") || !t["function"].is_object()) {
            continue;
        }
        const json& fn = t["function"];
        if (fn.value("name", std::string()) != name) {
            continue;
        }
        if (fn.contains("parameters") && fn["parameters"].is_object() && fn["parameters"].contains("properties") &&
            fn["parameters"]["properties"].is_object()) {
            return fn["parameters"]["properties"];
        }
        return json::object();
    }
    return json::object();
}

bool param_is_string_typed(const json& properties, const std::string& key) {
    if (!properties.is_object() || !properties.contains(key) || !properties[key].is_object()) {
        return false;
    }
    return properties[key].value("type", std::string()) == "string";
}

// -- XML dialect: <function=NAME>(<parameter=KEY>\nVALUE\n</parameter>)*</function> --
//
// Explicit cursor/scanner walk, not a regex: nested multi-parameter,
// multi-block XML with unbounded, possibly '<'/'>'-containing parameter
// values (e.g. a write_file tool's `content` holding literal HTML) is
// exactly the shape a regex handles unreliably, and a scanner's failure
// points (which tag, which byte offset) stay debuggable when something
// doesn't match, unlike a failed/greedy regex match.
bool parse_xml_call_body(const std::string& body, const std::vector<json>& offered_tools, ToolCall& out,
                          std::string& err) {
    static const std::string kFnOpen = "<function=";
    static const std::string kFnClose = "</function>";
    static const std::string kParamOpen = "<parameter=";
    static const std::string kParamClose = "</parameter>";

    size_t fn_tag = body.find(kFnOpen);
    if (fn_tag == std::string::npos) {
        err = "missing <function=...> tag inside <tool_call> block";
        return false;
    }
    size_t name_start = fn_tag + kFnOpen.size();
    size_t name_end = body.find('>', name_start);
    if (name_end == std::string::npos) {
        err = "<function=...> tag never closed with '>'";
        return false;
    }
    std::string name = trim(body.substr(name_start, name_end - name_start));
    if (name.empty()) {
        err = "empty function name in <function=> tag";
        return false;
    }

    size_t fn_close = body.find(kFnClose, name_end);
    if (fn_close == std::string::npos) {
        err = "missing </function> closing tag for function '" + name + "'";
        return false;
    }

    json properties = schema_properties_for(name, offered_tools);
    json args = json::object();

    size_t cursor = name_end + 1;
    while (true) {
        size_t next_param = body.find(kParamOpen, cursor);
        if (next_param == std::string::npos || next_param > fn_close) {
            break;  // no more parameters before the function closes.
        }
        size_t key_start = next_param + kParamOpen.size();
        size_t key_end = body.find('>', key_start);
        if (key_end == std::string::npos || key_end > fn_close) {
            err = "<parameter=...> tag never closed with '>' (function '" + name + "')";
            return false;
        }
        std::string key = trim(body.substr(key_start, key_end - key_start));
        if (key.empty()) {
            err = "empty parameter name in function '" + name + "'";
            return false;
        }
        size_t value_start = key_end + 1;
        size_t close_tag = body.find(kParamClose, value_start);
        if (close_tag == std::string::npos) {
            err = "parameter '" + key + "' value never finds its closing </parameter> tag (function '" + name + "')";
            return false;
        }
        std::string raw_value = body.substr(value_start, close_tag - value_start);
        raw_value = strip_one_edge_newline(raw_value, /*from_front=*/true);
        raw_value = strip_one_edge_newline(raw_value, /*from_front=*/false);

        if (param_is_string_typed(properties, key)) {
            // Schema says string -- keep the literal raw text verbatim,
            // even if it happens to look like JSON (e.g. starts with '{').
            args[key] = raw_value;
        } else {
            // Non-string (or unknown/unoffered) -- the template JSON-
            // encodes these, so decode; fall back to the raw string if
            // that fails rather than dropping the value.
            try {
                args[key] = json::parse(raw_value);
            } catch (const std::exception&) {
                args[key] = raw_value;
            }
        }

        cursor = close_tag + kParamClose.size();
    }

    out.name = name;
    out.arguments = args;
    return true;
}

// -- JSON dialect: the block body is already a JSON object -----------------
bool parse_json_call_body(const std::string& body, ToolCall& out, std::string& err) {
    json parsed;
    try {
        parsed = json::parse(trim(body));
    } catch (const std::exception& e) {
        err = std::string("malformed JSON in <tool_call> block: ") + e.what();
        return false;
    }
    if (!parsed.is_object() || !parsed.contains("name") || !parsed["name"].is_string()) {
        err = "tool_call JSON missing a string 'name' field";
        return false;
    }
    out.name = parsed["name"].get<std::string>();

    if (!parsed.contains("arguments")) {
        out.arguments = json::object();
        return true;
    }
    const json& args = parsed["arguments"];
    if (args.is_object()) {
        out.arguments = args;
        return true;
    }
    if (args.is_string()) {
        try {
            out.arguments = json::parse(args.get<std::string>());
            return true;
        } catch (const std::exception&) {
            err = "tool_call 'arguments' string was not valid JSON";
            return false;
        }
    }
    err = "tool_call 'arguments' field must be a JSON object";
    return false;
}

}  // namespace

ParsedToolCalls parse_tool_calls(const std::string& raw_text, ToolDialect dialect,
                                  const std::vector<json>& offered_tools) {
    ParsedToolCalls result;

    if (dialect == ToolDialect::NONE) {
        result.content = raw_text;
        return result;
    }

    static const std::string kThinkOpen = "<think>";
    static const std::string kThinkClose = "</think>";
    static const std::string kCallOpen = "<tool_call>";
    static const std::string kCallClose = "</tool_call>";

    std::string text = raw_text;

    // -- 1. think-block splitting -------------------------------------------
    size_t think_close = text.find(kThinkClose);
    if (think_close != std::string::npos) {
        size_t reasoning_start = 0;
        size_t think_open = text.find(kThinkOpen);
        if (think_open != std::string::npos && think_open < think_close) {
            reasoning_start = think_open + kThinkOpen.size();
        }
        result.reasoning_content = trim(text.substr(reasoning_start, think_close - reasoning_start));
        text = text.substr(think_close + kThinkClose.size());
    } else {
        size_t think_open = text.find(kThinkOpen);
        if (think_open != std::string::npos) {
            // Unterminated think block (generation was cut off -- e.g. hit
            // max_tokens -- while still "reasoning"). Never scan for real
            // tool calls inside it; but if an opening <tool_call> tag is
            // visible in there, that's exactly the dangerous "can't verify
            // a complete call" case, not a clean final answer, so it must
            // fail loud rather than silently becoming plain content.
            std::string after_open = text.substr(think_open + kThinkOpen.size());
            if (after_open.find(kCallOpen) != std::string::npos) {
                result.ok = false;
                result.error =
                    "generation was truncated inside an unterminated <think> block that contains an opening "
                    "<tool_call> tag -- cannot verify a complete call";
                return result;
            }
            result.reasoning_content = trim(after_open);
            result.content = trim(text.substr(0, think_open));
            return result;
        }
        // No think markers at all -- fall through, scan `text` unchanged.
    }

    // -- 2. scan for <tool_call>...</tool_call> blocks -----------------------
    size_t first_call = text.find(kCallOpen);
    if (first_call == std::string::npos) {
        result.content = trim(text);
        return result;  // clean, ordinary final answer.
    }
    result.content = trim(text.substr(0, first_call));

    size_t cursor = first_call;
    while (true) {
        size_t start_tag = text.find(kCallOpen, cursor);
        if (start_tag == std::string::npos) {
            break;
        }
        size_t body_start = start_tag + kCallOpen.size();
        size_t end_tag = text.find(kCallClose, body_start);
        if (end_tag == std::string::npos) {
            result.ok = false;
            result.error =
                "unterminated <tool_call> block (no matching </tool_call> -- likely truncated by max_tokens)";
            result.calls.clear();
            return result;
        }
        std::string body = text.substr(body_start, end_tag - body_start);

        ToolCall call;
        std::string err;
        bool parsed_ok = (dialect == ToolDialect::QWEN_XML) ? parse_xml_call_body(body, offered_tools, call, err)
                                                             : parse_json_call_body(body, call, err);
        if (!parsed_ok) {
            result.ok = false;
            result.error = "tool call #" + std::to_string(result.calls.size()) + ": " + err;
            result.calls.clear();
            return result;
        }
        if (!is_offered(call.name, offered_tools)) {
            result.ok = false;
            result.error = "model called tool '" + call.name + "' which was not in the offered tools list";
            result.calls.clear();
            return result;
        }
        result.calls.push_back(std::move(call));
        cursor = end_tag + kCallClose.size();
    }

    return result;
}

}  // namespace pleiades_engine
