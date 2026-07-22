#pragma once

#include <string>
#include <vector>

namespace pleiades_engine {

struct ChatMessage {
    std::string role;     // "system" | "user" | "assistant"
    std::string content;
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
// (<|im_start|>role\ncontent<|im_end|>\n) and defers real multi-model/
// tool-calling template support to a later phase. See
// docs/specs/2026-07-21-native-inference-engine-design.md, Phase 3.
std::string format_chatml(const std::vector<ChatMessage>& messages);

}  // namespace pleiades_engine
