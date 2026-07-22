// Pure string test -- no model, no GPU/CPU inference needed. Fast to run
// every time, unlike the fixture-model tests below.
#include "pleiades_engine/chat_template.h"

#include "test_util.h"

int main() {
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

    return 0;
}
