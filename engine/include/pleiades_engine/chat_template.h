#pragma once

#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace pleiades_engine {

// Which tool-calling text dialect a loaded model's own chat template
// expects/emits. There is no single "Qwen tool format" -- a council review
// (2026-07-23, cross-checked against real GGUF tokenizer.chat_template
// metadata pulled via llama_model_meta_val_str, not paraphrased) found at
// least two, both currently present on this machine:
//
//   QWEN_XML:  arch qwen35 / qwen35moe (e.g. the primary
//              Qwythos-9B-Claude-Mythos-5-1M model, and HauhauCS's
//              Qwen3.6-35B-A3B MoE):
//                <tool_call>
//                <function=NAME>
//                <parameter=KEY>
//                VALUE
//                </parameter>
//                </function>
//                </tool_call>
//
//   QWEN_JSON: arch qwen2 and dense qwen3 (e.g. Qwen2.5-0.5B-Instruct,
//              Rootkit7's Ternary-Bonsai-4B):
//                <tool_call>
//                {"name": NAME, "arguments": {...}}
//                </tool_call>
//
// A parser/formatter hardcoded to only one of these would silently produce
// zero tool calls against the other family. See ModelManager::tool_dialect()
// for how this is detected once, at model-load time, from the model's own
// template -- never guessed from a model name/path.
enum class ToolDialect {
    NONE,       // no tool-call dialect detected -- behave exactly as the
                // pre-tool-calling engine did (see format_chatml()).
    QWEN_XML,
    QWEN_JSON,
};

const char* tool_dialect_name(ToolDialect d);

// Sniffs a model's raw tokenizer.chat_template text for one of the two
// dialects above. Pure string function (no model/GGUF needed), so it's
// directly unit-testable against literal template snippets -- see
// test_tool_calls.cpp. XML is checked first: its `<function=` marker is
// unambiguous (pure XML-style tags, no JSON braces anywhere near it in
// either real sample), so checking it first keeps detection robust even if
// a future template's example text mixes markers from both families.
ToolDialect detect_tool_dialect(const std::string& chat_template);

// Whether this template's own `add_generation_prompt` tail defaults to an
// OPEN (model-completes-it) `<think>\n` block when `enable_thinking` isn't
// explicitly passed, vs. no think handling at all. Sniffed independently of
// dialect (not assumed from it) because real ground truth shows these are
// orthogonal: qwen35/qwen35moe (both XML-dialect samples checked) default
// to open thinking via this exact marker; qwen2 (JSON-dialect) has no think
// handling whatsoever; a dense-qwen3 JSON-dialect sample checked instead
// forces a pre-closed think block unconditionally (a third behavior this
// function deliberately does not try to distinguish from "no support" --
// see chat_template.cpp's doc comment for the full reasoning).
bool detect_open_thinking(const std::string& chat_template);

// One tool call, in the model's own {name, arguments} shape. Used both for
// an INBOUND assistant history turn's tool_calls (ChatMessage::tool_calls,
// round-tripped back into the dialect's native text by format_chat_prompt())
// and for an OUTBOUND freshly-parsed call from the model's own generated
// text (tool_call_parser.h). `arguments` is always a JSON object.
struct ToolCall {
    std::string name;
    nlohmann::json arguments;
};

struct ChatMessage {
    std::string role;                  // "system" | "user" | "assistant" | "tool"
    std::string content;
    std::string reasoning_content;     // assistant only; empty if absent
    std::vector<ToolCall> tool_calls;  // assistant only; empty if none
    std::string tool_call_id;          // role == "tool" only
    std::string tool_name;             // role == "tool" only (API fidelity; neither
                                        // dialect's template actually renders this
                                        // into the prompt text, only `content`)
};

// STOPGAP hardcoded ChatML formatter -- NOT real per-model jinja templating.
//
// Real templating (llama_model_chat_template() + llama.cpp's common/chat.cpp
// + minja) was deliberately deferred after a council review (qwen3.8-max +
// Kimi, 2026-07-22): chat.cpp is not self-contained (pulls in common.cpp,
// log.cpp, json-schema-to-grammar.cpp, the auto-parser/peg-parser files --
// confirmed by reading the actual pinned includes, not just taken on either
// model's word), so wiring it in is a real, separate unit of work, not a
// quick add. Since Pleiades serves mostly Qwen-family GGUFs today and this
// phase's goal is proving the HTTP plumbing (not perfecting templating),
// this hardcodes Qwen2/2.5/3.x's stable ChatML-style template
// (<|im_start|>role\ncontent<|im_end|>\n) and defers real multi-model
// template support to a later phase. See
// docs/specs/2026-07-21-native-inference-engine-design.md, Phase 3.
//
// Tool-calling (Phase 5, 2026-07-23) is layered on top via
// format_chat_prompt() below rather than by editing this function in place
// -- format_chatml() remains the exact byte-for-byte behavior for
// ToolDialect::NONE models, still covered by test_chat_template.cpp.
std::string format_chatml(const std::vector<ChatMessage>& messages);

// Tool-aware formatter. `tools` is the request's raw OpenAI-shape `tools`
// array (each element `{"type":"function","function":{"name",
// "description","parameters"}}`), passed straight through into the
// dialect's own `<tools>...</tools>` block via compact JSON, exactly like
// the real templates' own `tool | tojson`.
//
// When `dialect == NONE`, this is exactly `format_chatml(messages)` --
// tools/tool_calls/tool-role messages are ignored entirely, matching "no
// tool support, behave exactly as today" (see ModelManager::tool_dialect()).
//
// When `dialect != NONE`, assistant tool_calls and role=="tool" messages
// are re-rendered into the dialect's own native text form regardless of
// whether `tools` is non-empty on THIS specific request (history from an
// earlier round should still round-trip correctly even if, unusually, this
// exact call omitted `tools`) -- but the `# Tools...` instructional system
// block is only injected when `tools` is non-empty.
//
// Deliberate simplifications vs. the real per-model jinja (documented in
// chat_template.cpp, not hidden): a model-specific forced-identity string
// some finetunes splice into their own template (e.g. Qwythos's own "You
// are Qwythos..." override, which is designed to override conflicting
// identity instructions) is never reproduced -- Pleiades has its own
// per-character system-prompt identity, and forcibly injecting a
// finetune's own persona override here would actively fight that, not
// just omit a nicety. Likewise the vanilla Qwen2.5 template's hardcoded
// "You are Qwen, created by Alibaba Cloud" fallback (shown only when a
// request has NO system message at all) is not reproduced, since
// Pleiades' harness always sends one.
std::string format_chat_prompt(const std::vector<ChatMessage>& messages, const std::vector<nlohmann::json>& tools,
                                ToolDialect dialect, bool open_thinking);

// Cross-family fallback formatter (Stage A of real per-model templating).
// Renders `messages` through llama.cpp's OWN built-in template engine
// (llama_chat_apply_template) driven by `tmpl` -- the model's own
// tokenizer.chat_template string (ModelManager::chat_template()). This is NOT
// jinja: llama.cpp sniffs the template text against a fixed list of known
// families (Llama-3, Gemma, Phi-3, Mistral, Zephyr, ChatML, ...) and emits
// that family's correct role markup. It exists so a NON-Qwen instruct GGUF
// stops being silently mis-rendered as Qwen ChatML by format_chatml(): the
// ToolDialect::NONE path in http_server.cpp uses this whenever the model
// carries a template, and only falls back to format_chatml() when it doesn't
// (a base model) or llama.cpp doesn't recognize the family.
//
// Only role+content is rendered -- tool calls / tool-role messages are NOT
// supported (llama.cpp's builtin templater has no tool-calling; that is Stage
// B's real jinja work). That's correct for a NONE-dialect model, which has no
// tool support to begin with (see ModelManager::tool_dialect()). The Qwen
// XML/JSON dialects never reach here; they keep their own hardcoded renderers.
//
// Returns "" when `tmpl` is empty, `messages` is empty, or llama.cpp does not
// recognize the template family -- the caller then falls back to
// format_chatml(). `add_generation_prompt` appends the trailing
// assistant-turn opener, exactly like format_chatml()'s own tail.
std::string apply_builtin_template(const std::string& tmpl, const std::vector<ChatMessage>& messages,
                                   bool add_generation_prompt);

// Parses an OpenAI-shape request body's `messages` array into ChatMessage
// structs, round-tripping `tool_calls` (assistant turns) and
// `tool_call_id`/`name` (role=="tool" turns) -- see http_server.cpp's
// chat-completions handler, the only caller in this binary.
//
// Null-safe throughout: nlohmann's `.value(key, default)` only substitutes
// `default` for a MISSING key -- a PRESENT key holding JSON `null` still
// reaches `.get<std::string>()`, which throws `type_error` 302. An
// OpenAI-standard assistant tool-call turn commonly has `content: null`
// (no accompanying prose, just tool_calls), and Pleiades' own harness
// (pleiades/harness/agent.py::_assistant_turn) round-trips whatever this
// engine itself emits verbatim -- so if this engine ever emits `content:
// null` (which it does, matching the OpenAI convention, whenever a
// tool-call response has no leading prose), the very next turn of ANY
// multi-round tool-use conversation would feed it right back here. This
// was a real, live, latent 500 before this function existed; every
// optional string field below is read through the same null-safe path.
std::vector<ChatMessage> parse_chat_messages(const nlohmann::json& body);

}  // namespace pleiades_engine
