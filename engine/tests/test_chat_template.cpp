// Pure string test -- no model, no GPU/CPU inference needed. Fast to run
// every time, unlike the fixture-model tests below.
#include "pleiades_engine/chat_template.h"

#include "test_util.h"

int main() {
    using pleiades_engine::apply_builtin_template;
    using pleiades_engine::ChatMessage;
    using pleiades_engine::format_chatml;

    // Single user turn.
    {
        std::string out = format_chatml({{"user", "hi"}});
        PLEIADES_CHECK(out == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    }

    // System + user, matching the shape a real chat request sends.
    {
        std::string out = format_chatml({{"system", "You are helpful."}, {"user", "hi"}});
        PLEIADES_CHECK(out ==
                        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
                        "<|im_start|>user\nhi<|im_end|>\n"
                        "<|im_start|>assistant\n");
    }

    // Empty message list -- still terminates with the assistant preamble,
    // doesn't crash. (The HTTP layer rejects empty `messages` before this
    // is ever called, but the formatter itself should still be well-defined.)
    {
        std::string out = format_chatml({});
        PLEIADES_CHECK(out == "<|im_start|>assistant\n");
    }

    // -- Stage A: cross-family builtin templating ------------------------- //
    // apply_builtin_template() routes a NON-Qwen model's own chat_template
    // through llama.cpp's builtin templater so it gets ITS family's markup,
    // not the Qwen ChatML the old stopgap forced on everything. Pure string
    // logic (no model/backend needed), so it's exercised here. The template
    // strings only need the substrings llama.cpp's family detector keys on.
    const std::vector<ChatMessage> sys_user = {{"system", "You are helpful."}, {"user", "hi"}};

    // A Llama-3 template renders Llama-3 header markup -- and, crucially, NOT
    // the <|im_start|> ChatML the pre-Stage-A path would have mis-rendered.
    {
        const std::string llama3_tmpl =
            "{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
            "{{ message['content'] }}<|eot_id|>{% endfor %}";
        std::string out = apply_builtin_template(llama3_tmpl, sys_user, /*add_generation_prompt=*/true);
        PLEIADES_CHECK(out ==
                        "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>"
                        "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"
                        "<|start_header_id|>assistant<|end_header_id|>\n\n");
        PLEIADES_CHECK(out.find("<|im_start|>") == std::string::npos);
    }

    // A Gemma template renders Gemma turns: no separate system turn (merged
    // into the first user turn) and "assistant" becomes "model".
    {
        const std::string gemma_tmpl =
            "{% for message in messages %}<start_of_turn>{{ message['role'] }}\n"
            "{{ message['content'] }}<end_of_turn>\n{% endfor %}";
        std::string out = apply_builtin_template(gemma_tmpl, sys_user, /*add_generation_prompt=*/true);
        PLEIADES_CHECK(out == "<start_of_turn>user\nYou are helpful.\n\nhi<end_of_turn>\n<start_of_turn>model\n");
    }

    // Longer content than the initial buffer estimate must still round-trip
    // (the grow-once path). A ~400-char user turn exceeds 2*chars only if the
    // estimate is wrong; assert the whole thing survives regardless.
    {
        const std::string big(400, 'x');
        const std::string llama3_tmpl =
            "{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
            "{{ message['content'] }}<|eot_id|>{% endfor %}";
        std::string out = apply_builtin_template(llama3_tmpl, {{"user", big}}, /*add_generation_prompt=*/true);
        PLEIADES_CHECK(out.find(big) != std::string::npos);
    }

    // No template (base model) and an unrecognized family both return "" so
    // the caller falls back to format_chatml() instead of a wrong-family render.
    {
        PLEIADES_CHECK(apply_builtin_template("", sys_user, true).empty());
        PLEIADES_CHECK(apply_builtin_template("not a known template at all", sys_user, true).empty());
    }

    return 0;
}
