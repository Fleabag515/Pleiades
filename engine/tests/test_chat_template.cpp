// Offline wrapper tests for the real jinja chat-templating layer
// (chat_template.h / chat_template.cpp), which fronts llama.cpp's common_chat_*
// engine. No GGUF/model is loaded: ChatTemplates::create(nullptr, override)
// drives everything from a jinja template string (llama.cpp's
// common_chat_templates_init supports a null model together with an explicit
// template), so these run fast and offline like the pre-Stage-B string tests.
//
// The real cross-family, tool-calling-with-a-loaded-model coverage lives in the
// end-to-end HTTP tests (test_http_tool_calls.cpp), which boot the server
// against real Qwen and non-Qwen GGUFs -- that's where the PEG tool-call parser
// derived from each model's own template is exercised for real.
#include "pleiades_engine/chat_template.h"

#include "nlohmann/json.hpp"
#include "test_util.h"

using json = nlohmann::json;
using namespace pleiades_engine;

int main() {
    // A minimal ChatML-style jinja template -- enough to exercise the render +
    // parse round-trip without a model. The exact rendered bytes are a property
    // of THIS template (not a hardcoded engine format), so they're assertable.
    const std::string chatml =
        "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    // ---- create from an override template, no model --------------------------
    {
        ChatTemplates t = ChatTemplates::create(nullptr, chatml);
        PLEIADES_CHECK(t.valid());
        PLEIADES_CHECK(!t.source().empty());
    }

    // ---- render a plain system+user turn through the real jinja engine -------
    {
        ChatTemplates t = ChatTemplates::create(nullptr, chatml);
        json messages = json::array({
            {{"role", "system"}, {"content", "You are helpful."}},
            {{"role", "user"}, {"content", "hi"}},
        });
        RenderedChat rc = t.apply(messages, json::array(), /*add_generation_prompt=*/true, /*enable_thinking=*/true);
        const std::string& p = rc.prompt();
        PLEIADES_CHECK(p.find("<|im_start|>system\nYou are helpful.<|im_end|>\n") != std::string::npos);
        PLEIADES_CHECK(p.find("<|im_start|>user\nhi<|im_end|>\n") != std::string::npos);
        // ends with the generation prompt
        const std::string tail = "<|im_start|>assistant\n";
        PLEIADES_CHECK(p.size() >= tail.size() && p.substr(p.size() - tail.size()) == tail);
    }

    // ---- parse a plain final answer -> content, no tool calls, no throw ------
    {
        ChatTemplates t = ChatTemplates::create(nullptr, chatml);
        RenderedChat rc = t.apply(json::array({{{"role", "user"}, {"content", "hi"}}}), json::array(), true, true);
        ParsedChatMessage pm = rc.parse("Just a normal answer.", /*is_partial=*/false);
        PLEIADES_CHECK(pm.tool_calls.empty());
        PLEIADES_CHECK(pm.content == "Just a normal answer.");
        PLEIADES_CHECK(pm.reasoning_content.empty());
    }

    // ---- null content on an assistant tool-call history turn must not throw --
    // llama.cpp's own oaicompat message parser tolerates content:null (the case
    // the engine previously hand-guarded against); prove the wrapper feeds it
    // through and renders the round-tripped conversation without crashing.
    {
        ChatTemplates t = ChatTemplates::create(nullptr, chatml);
        json convo = json::array({
            {{"role", "user"}, {"content", "weather?"}},
            {{"role", "assistant"},
             {"content", nullptr},
             {"tool_calls", json::array({{{"id", "call_0"},
                                          {"type", "function"},
                                          {"function", {{"name", "get_weather"},
                                                        {"arguments", "{\"location\":\"LA\"}"}}}}})}},
            {{"role", "tool"}, {"tool_call_id", "call_0"}, {"name", "get_weather"}, {"content", "72F"}},
        });
        bool threw = false;
        try {
            RenderedChat rc = t.apply(convo, json::array(), true, true);
            (void)rc.prompt();
        } catch (const std::exception&) {
            threw = true;
        }
        PLEIADES_CHECK(!threw);
    }

    // ---- a malformed override template fails loud at create() ----------------
    {
        bool threw = false;
        try {
            ChatTemplates t = ChatTemplates::create(nullptr, "{% for x in %}broken");
            (void)t;
        } catch (const std::exception&) {
            threw = true;
        }
        PLEIADES_CHECK(threw);
    }

    return 0;
}
