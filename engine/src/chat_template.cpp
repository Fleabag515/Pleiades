#include "pleiades_engine/chat_template.h"

namespace pleiades_engine {

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

}  // namespace pleiades_engine
