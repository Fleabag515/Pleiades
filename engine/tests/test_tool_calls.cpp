// Pure string/JSON logic tests -- no model needed, matching
// test_chat_template.cpp's fast/offline style. Covers chat_template.cpp's
// dialect detection / tool-aware prompt formatting / inbound message
// parsing, and tool_call_parser.cpp's output parsing.
//
// The dialect/instructional-text snippets below were reverse-engineered
// from real, currently-present GGUFs' own tokenizer.chat_template (dumped
// via llama_model_meta_val_str, 2026-07-23):
//   XML:  empero-ai/Qwythos-9B-Claude-Mythos-5-1M (arch qwen35) and
//         HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive (arch
//         qwen35moe).
//   JSON: Qwen/Qwen2.5-0.5B-Instruct (arch qwen2) and Rootkit7's
//         Ternary-Bonsai-4B-abliterated (arch qwen3, dense).
// See chat_template.cpp for the full byte-for-byte instructional text this
// feature injects, pulled from those same files.
#include "pleiades_engine/chat_template.h"
#include "pleiades_engine/tool_call_parser.h"

#include "nlohmann/json.hpp"
#include "test_util.h"

using json = nlohmann::json;
using namespace pleiades_engine;

namespace {

// Minimal real snippets of each dialect's own template, just enough to
// exercise detect_tool_dialect()/detect_open_thinking() without needing a
// whole multi-KB template string inline.
const char* kXmlTemplateSnippet =
    "...<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "</function>\n</tool_call>...\n{%- if enable_thinking is defined and enable_thinking is false %}\n"
    "{{- '<think>\n\n</think>\n\n' }}\n{%- else %}\n{{- '<think>\n' }}\n{%- endif %}";

const char* kJsonTemplateSnippet =
    "...<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>...";

const char* kPlainTemplateSnippet =
    "{%- if messages[0]['role'] == 'system' %}{{- '<|im_start|>system\n' + messages[0]['content'] }}{%- endif %}";

std::vector<json> weather_tools() {
    json get_weather = {
        {"type", "function"},
        {"function",
         {{"name", "get_weather"},
          {"description", "Get the weather for a location."},
          {"parameters",
           {{"type", "object"},
            {"properties", {{"location", {{"type", "string"}}}, {"unit", {{"type", "string"}}}}},
            {"required", json::array({"location"})}}}}},
    };
    json set_volume = {
        {"type", "function"},
        {"function",
         {{"name", "set_volume"},
          {"description", "Set the system volume."},
          {"parameters", {{"type", "object"}, {"properties", {{"level", {{"type", "integer"}}}}}}}}},
    };
    json write_file = {
        {"type", "function"},
        {"function",
         {{"name", "write_file"},
          {"description", "Write text to a file."},
          {"parameters",
           {{"type", "object"}, {"properties", {{"path", {{"type", "string"}}}, {"content", {{"type", "string"}}}}}}}}},
    };
    return {get_weather, set_volume, write_file};
}

}  // namespace

int main() {
    auto tools = weather_tools();

    // ---- dialect / open-thinking detection --------------------------------
    {
        PLEIADES_CHECK(detect_tool_dialect(kXmlTemplateSnippet) == ToolDialect::QWEN_XML);
        PLEIADES_CHECK(detect_tool_dialect(kJsonTemplateSnippet) == ToolDialect::QWEN_JSON);
        PLEIADES_CHECK(detect_tool_dialect(kPlainTemplateSnippet) == ToolDialect::NONE);
        PLEIADES_CHECK(std::string(tool_dialect_name(ToolDialect::QWEN_XML)) == "QWEN_XML");
        PLEIADES_CHECK(std::string(tool_dialect_name(ToolDialect::QWEN_JSON)) == "QWEN_JSON");
        PLEIADES_CHECK(std::string(tool_dialect_name(ToolDialect::NONE)) == "NONE");

        PLEIADES_CHECK(detect_open_thinking(kXmlTemplateSnippet) == true);
        PLEIADES_CHECK(detect_open_thinking(kJsonTemplateSnippet) == false);
    }

    // ---- format_chat_prompt: XML dialect, tools present -------------------
    {
        std::vector<ChatMessage> msgs = {{"system", "You are a helpful assistant."}, {"user", "weather in LA?"}};
        std::string prompt = format_chat_prompt(msgs, tools, ToolDialect::QWEN_XML, /*open_thinking=*/true);
        PLEIADES_CHECK(prompt.find("# Tools\n\nYou have access to the following functions:") != std::string::npos);
        PLEIADES_CHECK(prompt.find("\"get_weather\"") != std::string::npos);
        PLEIADES_CHECK(prompt.find("<function=example_function_name>") != std::string::npos);
        PLEIADES_CHECK(prompt.find("You are a helpful assistant.") != std::string::npos);
        // system content comes AFTER the tools block for the XML dialect.
        PLEIADES_CHECK(prompt.find("# Tools") < prompt.find("You are a helpful assistant."));
        PLEIADES_CHECK(prompt.find("<|im_start|>user\nweather in LA?<|im_end|>\n") != std::string::npos);
        // reasoning-open models get a bare, unclosed <think>\n generation prompt.
        std::string tail = prompt.substr(prompt.size() - std::string("<|im_start|>assistant\n<think>\n").size());
        PLEIADES_CHECK(tail == "<|im_start|>assistant\n<think>\n");
    }

    // ---- format_chat_prompt: JSON dialect, tools present, system-before-tools ----
    {
        std::vector<ChatMessage> msgs = {{"system", "Be terse."}, {"user", "hi"}};
        std::string prompt = format_chat_prompt(msgs, tools, ToolDialect::QWEN_JSON, /*open_thinking=*/false);
        PLEIADES_CHECK(prompt.find("# Tools\n\nYou may call one or more functions") != std::string::npos);
        // system content comes BEFORE the tools block for the JSON dialect
        // (opposite order from XML) -- a real, checked ground-truth difference.
        PLEIADES_CHECK(prompt.find("Be terse.") < prompt.find("# Tools"));
        PLEIADES_CHECK(prompt.find("{{\"name\": <function-name>") != std::string::npos);
        // no open_thinking -> plain assistant preamble, no <think> at all.
        PLEIADES_CHECK(prompt.substr(prompt.size() - std::string("<|im_start|>assistant\n").size()) ==
                       "<|im_start|>assistant\n");
        PLEIADES_CHECK(prompt.find("<think>") == std::string::npos);
    }

    // ---- dialect NONE: byte-identical to format_chatml, tools ignored -----
    {
        std::vector<ChatMessage> msgs = {{"user", "hi"}};
        PLEIADES_CHECK(format_chat_prompt(msgs, tools, ToolDialect::NONE, true) == format_chatml(msgs));
    }

    // ---- round trip: assistant tool_call + tool result, XML dialect ------
    {
        ChatMessage assistant;
        assistant.role = "assistant";
        assistant.content = "";
        assistant.tool_calls = {{"get_weather", json{{"location", "Los Angeles"}, {"unit", "f"}}}};
        ChatMessage tool_result;
        tool_result.role = "tool";
        tool_result.tool_call_id = "call_0";
        tool_result.tool_name = "get_weather";
        tool_result.content = "72 and sunny";

        std::vector<ChatMessage> msgs = {{"user", "weather?"}, assistant, tool_result};
        std::string prompt = format_chat_prompt(msgs, {}, ToolDialect::QWEN_XML, false);

        std::string expected_call =
            "<tool_call>\n<function=get_weather>\n<parameter=location>\nLos Angeles\n</parameter>\n"
            "<parameter=unit>\nf\n</parameter>\n</function>\n</tool_call>";
        PLEIADES_CHECK(prompt.find(expected_call) != std::string::npos);
        std::string expected_tool_turn =
            "<|im_start|>user\n<tool_response>\n72 and sunny\n</tool_response><|im_end|>\n";
        PLEIADES_CHECK(prompt.find(expected_tool_turn) != std::string::npos);
        PLEIADES_CHECK(prompt.find(expected_call) < prompt.find(expected_tool_turn));
    }

    // ---- round trip: two consecutive tool messages merge into one turn ---
    {
        ChatMessage t1{"tool", "result A"};
        ChatMessage t2{"tool", "result B"};
        std::vector<ChatMessage> msgs = {{"user", "go"}, t1, t2};
        std::string prompt = format_chat_prompt(msgs, {}, ToolDialect::QWEN_JSON, false);
        std::string expected =
            "<|im_start|>user\n<tool_response>\nresult A\n</tool_response>\n<tool_response>\nresult "
            "B\n</tool_response><|im_end|>\n";
        PLEIADES_CHECK(prompt.find(expected) != std::string::npos);
    }

    // ---- parse_tool_calls: XML dialect, single call -----------------------
    {
        std::string raw =
            "<tool_call>\n<function=get_weather>\n<parameter=location>\nLos Angeles\n</parameter>\n</function>\n"
            "</tool_call>";
        ParsedToolCalls r = parse_tool_calls(raw, ToolDialect::QWEN_XML, tools);
        PLEIADES_CHECK(r.ok);
        PLEIADES_CHECK(r.calls.size() == 1);
        PLEIADES_CHECK(r.calls[0].name == "get_weather");
        PLEIADES_CHECK(r.calls[0].arguments.at("location").get<std::string>() == "Los Angeles");
    }

    // ---- parse_tool_calls: JSON dialect, single call -----------------------
    {
        std::string raw =
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}\n</tool_call>";
        ParsedToolCalls r = parse_tool_calls(raw, ToolDialect::QWEN_JSON, tools);
        PLEIADES_CHECK(r.ok);
        PLEIADES_CHECK(r.calls.size() == 1);
        PLEIADES_CHECK(r.calls[0].name == "get_weather");
        PLEIADES_CHECK(r.calls[0].arguments.at("location").get<std::string>() == "Tokyo");
    }

    // ---- parse_tool_calls: prose before a call, multiple calls in one response, schema-typed int ----
    {
        std::string raw =
            "Sure, let me check both.<tool_call>\n<function=get_weather>\n<parameter=location>\nLA\n</parameter>\n"
            "</function>\n</tool_call>\n<tool_call>\n<function=set_volume>\n<parameter=level>\n11\n</parameter>\n"
            "</function>\n</tool_call>";
        ParsedToolCalls r = parse_tool_calls(raw, ToolDialect::QWEN_XML, tools);
        PLEIADES_CHECK(r.ok);
        PLEIADES_CHECK(r.content == "Sure, let me check both.");
        PLEIADES_CHECK(r.calls.size() == 2);
        PLEIADES_CHECK(r.calls[0].name == "get_weather");
        PLEIADES_CHECK(r.calls[1].name == "set_volume");
        // level's schema type is "integer" -> must be a real JSON number, not the string "11".
        PLEIADES_CHECK(r.calls[1].arguments.at("level").is_number());
        PLEIADES_CHECK(r.calls[1].arguments.at("level").get<int>() == 11);
    }

    // ---- parse_tool_calls: think-block splitting ---------------------------
    {
        std::string raw =
            "Let me think about this.\n</think>\n\nHere you go.<tool_call>\n{\"name\": \"get_weather\", "
            "\"arguments\": {\"location\": \"LA\"}}\n</tool_call>";
        ParsedToolCalls r = parse_tool_calls(raw, ToolDialect::QWEN_JSON, tools);
        PLEIADES_CHECK(r.ok);
        PLEIADES_CHECK(r.reasoning_content == "Let me think about this.");
        PLEIADES_CHECK(r.content == "Here you go.");
        PLEIADES_CHECK(r.calls.size() == 1);
    }

    // ---- parse_tool_calls: no tool call at all -> ordinary final answer ----
    {
        ParsedToolCalls r = parse_tool_calls("Just a normal answer, no tools needed.", ToolDialect::QWEN_XML, tools);
        PLEIADES_CHECK(r.ok);
        PLEIADES_CHECK(r.calls.empty());
        PLEIADES_CHECK(r.content == "Just a normal answer, no tools needed.");
    }

    // ---- parse_tool_calls: malformed call (missing </parameter>) must fail loud ----
    {
        std::string raw = "<tool_call>\n<function=get_weather>\n<parameter=location>\nLA\n</function>\n</tool_call>";
        ParsedToolCalls r = parse_tool_calls(raw, ToolDialect::QWEN_XML, tools);
        PLEIADES_CHECK(!r.ok);
        PLEIADES_CHECK(!r.error.empty());
        PLEIADES_CHECK(r.calls.empty());
    }

    // ---- parse_tool_calls: truncated call (no closing </tool_call>, e.g. hit max_tokens) ----
    {
        std::string raw = "<tool_call>\n<function=get_weather>\n<parameter=location>\nLA\n</parameter>\n</function>";
        ParsedToolCalls r = parse_tool_calls(raw, ToolDialect::QWEN_XML, tools);
        PLEIADES_CHECK(!r.ok);
        PLEIADES_CHECK(r.calls.empty());
    }

    // ---- parse_tool_calls: malformed JSON dialect body must fail loud, not become content ----
    {
        std::string raw = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \n</tool_call>";
        ParsedToolCalls r = parse_tool_calls(raw, ToolDialect::QWEN_JSON, tools);
        PLEIADES_CHECK(!r.ok);
    }

    // ---- parse_tool_calls: unterminated <think> hiding a <tool_call> must fail loud ----
    {
        std::string raw = "<think>\nhmm, let me call <tool_call>\n<function=get_weather>\n";
        ParsedToolCalls r = parse_tool_calls(raw, ToolDialect::QWEN_XML, tools);
        PLEIADES_CHECK(!r.ok);
    }

    // ---- parse_tool_calls: unterminated <think> with NO tool_call inside is just truncated reasoning, not an error ----
    {
        std::string raw = "<think>\nstill thinking about the weather and nothing else";
        ParsedToolCalls r = parse_tool_calls(raw, ToolDialect::QWEN_XML, tools);
        PLEIADES_CHECK(r.ok);
        PLEIADES_CHECK(r.calls.empty());
        PLEIADES_CHECK(r.reasoning_content == "still thinking about the weather and nothing else");
    }

    // ---- parse_tool_calls: schema-typed STRING parameter that looks like JSON must stay a raw string ----
    {
        std::string raw =
            "<tool_call>\n<function=write_file>\n<parameter=path>\nnote.txt\n</parameter>\n<parameter=content>\n"
            "{\"not\": \"real json, just text that starts with a brace\"}\n</parameter>\n</function>\n</tool_call>";
        ParsedToolCalls r = parse_tool_calls(raw, ToolDialect::QWEN_XML, tools);
        PLEIADES_CHECK(r.ok);
        PLEIADES_CHECK(r.calls.size() == 1);
        const json& content_val = r.calls[0].arguments.at("content");
        PLEIADES_CHECK(content_val.is_string());
        PLEIADES_CHECK(content_val.get<std::string>() ==
                       "{\"not\": \"real json, just text that starts with a brace\"}");
    }

    // ---- parse_tool_calls: anchor check -- a name that wasn't offered must fail loud ----
    {
        std::string raw = "<tool_call>\n<function=delete_everything>\n</function>\n</tool_call>";
        ParsedToolCalls r = parse_tool_calls(raw, ToolDialect::QWEN_XML, tools);
        PLEIADES_CHECK(!r.ok);
        PLEIADES_CHECK(r.calls.empty());
    }

    // ---- parse_tool_calls: dialect NONE -> tool support totally absent, always ok, never parses calls ----
    {
        std::string raw = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {}}\n</tool_call>";
        ParsedToolCalls r = parse_tool_calls(raw, ToolDialect::NONE, tools);
        PLEIADES_CHECK(r.ok);
        PLEIADES_CHECK(r.calls.empty());
        PLEIADES_CHECK(r.content == raw);
    }

    // ---- parse_chat_messages: null content on an inbound assistant tool-call turn must not throw ----
    {
        json body = {
            {"messages",
             json::array({
                 json{{"role", "user"}, {"content", "weather?"}},
                 json{{"role", "assistant"},
                      {"content", nullptr},
                      {"tool_calls", json::array({json{{"id", "call_0"},
                                                        {"type", "function"},
                                                        {"function",
                                                         json{{"name", "get_weather"},
                                                              {"arguments", "{\"location\": \"LA\"}"}}}}})}},
                 json{{"role", "tool"}, {"tool_call_id", "call_0"}, {"name", "get_weather"}, {"content", "72F"}},
             })},
        };
        std::vector<ChatMessage> msgs;
        bool threw = false;
        try {
            msgs = parse_chat_messages(body);
        } catch (const std::exception&) {
            threw = true;
        }
        PLEIADES_CHECK(!threw);
        PLEIADES_CHECK(msgs.size() == 3);
        PLEIADES_CHECK(msgs[1].role == "assistant");
        PLEIADES_CHECK(msgs[1].content.empty());  // null coerced to "", not left dangling/throwing.
        PLEIADES_CHECK(msgs[1].tool_calls.size() == 1);
        PLEIADES_CHECK(msgs[1].tool_calls[0].name == "get_weather");
        PLEIADES_CHECK(msgs[1].tool_calls[0].arguments.at("location").get<std::string>() == "LA");
        PLEIADES_CHECK(msgs[2].role == "tool");
        PLEIADES_CHECK(msgs[2].tool_call_id == "call_0");
        PLEIADES_CHECK(msgs[2].content == "72F");

        // And the whole round-tripped conversation renders without crashing.
        std::string prompt = format_chat_prompt(msgs, {}, ToolDialect::QWEN_JSON, false);
        PLEIADES_CHECK(prompt.find("\"name\": \"get_weather\"") != std::string::npos);
        PLEIADES_CHECK(prompt.find("\"location\"") != std::string::npos);
        PLEIADES_CHECK(prompt.find("<tool_response>\n72F\n</tool_response>") != std::string::npos);
    }

    // ---- parse_chat_messages: missing content key entirely must also not throw ----
    {
        json body = {{"messages", json::array({json{{"role", "user"}}})}};
        std::vector<ChatMessage> msgs = parse_chat_messages(body);
        PLEIADES_CHECK(msgs.size() == 1);
        PLEIADES_CHECK(msgs[0].content.empty());
    }

    return 0;
}
