#include "pleiades_engine/prefix_cache.h"

#include <algorithm>

namespace pleiades_engine {

size_t PrefixCache::reusable_prefix(const std::vector<llama_token>& prompt) const {
    const size_t n = std::min(tokens_.size(), prompt.size());
    size_t lcp = 0;
    while (lcp < n && tokens_[lcp] == prompt[lcp]) {
        ++lcp;
    }
    // Never reuse the entire prompt -- always leave at least the final
    // token to decode fresh (see header). For a 0- or 1-token prompt this
    // is 0, i.e. decode the whole (tiny) thing cold.
    const size_t cap = prompt.empty() ? 0 : prompt.size() - 1;
    return std::min(lcp, cap);
}

void PrefixCache::set(std::vector<llama_token> tokens, uint64_t epoch) {
    tokens_ = std::move(tokens);
    epoch_ = epoch;
}

}  // namespace pleiades_engine
