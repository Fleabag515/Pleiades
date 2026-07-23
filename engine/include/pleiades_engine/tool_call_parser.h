#pragma once

#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "pleiades_engine/chat_template.h"

namespace pleiades_engine {

// Outcome of parse_tool_calls(). Exactly one of three states, deliberately
// never ambiguous (see http_server.cpp's handling of `ok`):
//   ok==true,  calls.empty()  -> no <tool_call> found at all: a normal,
//                                 final answer (finish_reason "stop").
//   ok==true,  !calls.empty() -> one or more well-formed, schema-typed,
//                                 anchored calls (finish_reason
//                                 "tool_calls").
//   ok==false                 -> the text contains an opening <tool_call>
//                                 (or an unterminated <think> hiding one)
//                                 that could NOT be turned into a
//                                 complete, trustworthy call. The caller
//                                 MUST NOT treat this as a normal final
//                                 answer -- see `error` for why, and
//                                 http_server.cpp's handle_chat(), which
//                                 surfaces this as a distinct HTTP error
//                                 rather than a 200 with finish_reason
//                                 "stop". This is the failure mode that
//                                 matters most: Pleiades often runs with
//                                 exec_policy "allow", so a real intended
//                                 action (shell command, email send, ...)
//                                 must never silently fail to fire just
//                                 because its <tool_call> was malformed.
struct ParsedToolCalls {
    bool ok = true;
    std::string error;               // only set if !ok
    std::string content;             // prose before the first <tool_call> (or all text, if none/failure)
    std::string reasoning_content;   // <think>...</think> body, if present
    std::vector<ToolCall> calls;
};

// Parses a model's raw, COMPLETE (never streamed/partial -- the harness
// always requests stream:false for tool-capable calls, see
// pleiades/harness/llm.py::_chat_openai's `"stream": False`) generated text
// for the two known Qwen tool-calling dialects (see chat_template.h).
//
// `offered_tools` is the request's own `tools` array (OpenAI shape), used
// to:
//   (1) anchor every parsed call's name against what was actually offered
//       -- a name that wasn't offered makes the whole parse fail (ok=false),
//       guarding against both hallucinated tool names and prose that
//       happens to contain literal <tool_call>-like text; and
//   (2) schema-type XML-dialect parameter values -- a "string"-typed
//       parameter is kept as a literal raw string; anything else is
//       json::parse()'d, falling back to a raw string on parse failure.
//       This matters because the dialect's own template renders string
//       params raw and non-string params JSON-encoded, so a schema-blind
//       parser cannot tell a literal string "42" from the number 42, or a
//       string value that happens to start with '{'/'['.
//
// dialect == NONE short-circuits to "no tool support" (ok=true, all text
// as content, never scans for <tool_call> at all) -- matches "behave
// exactly as today" for a model with no detected dialect.
ParsedToolCalls parse_tool_calls(const std::string& raw_text, ToolDialect dialect,
                                  const std::vector<nlohmann::json>& offered_tools);

}  // namespace pleiades_engine
