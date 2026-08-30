#include "pleiades_engine/slot_state.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace pleiades_engine {

namespace {

constexpr char kMagic[4] = {'P', 'L', 'S', 'T'};
constexpr uint32_t kVersion = 1;

void write_u32(std::ofstream& out, uint32_t v) {
    unsigned char b[4] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8),
                          static_cast<unsigned char>(v >> 16), static_cast<unsigned char>(v >> 24)};
    out.write(reinterpret_cast<const char*>(b), 4);
}

void write_u64(std::ofstream& out, uint64_t v) {
    unsigned char b[8];
    for (int i = 0; i < 8; ++i) {
        b[i] = static_cast<unsigned char>(v >> (8 * i));
    }
    out.write(reinterpret_cast<const char*>(b), 8);
}

bool read_exact(std::ifstream& in, void* dst, size_t n) {
    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
    return static_cast<size_t>(in.gcount()) == n;
}

bool read_u32(std::ifstream& in, uint32_t& v) {
    unsigned char b[4];
    if (!read_exact(in, b, 4)) {
        return false;
    }
    v = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
        (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
    return true;
}

bool read_u64(std::ifstream& in, uint64_t& v) {
    unsigned char b[8];
    if (!read_exact(in, b, 8)) {
        return false;
    }
    v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(b[i]) << (8 * i);
    }
    return true;
}

// tmp-file + rename so a crash mid-save can never leave a truncated file that
// a later restore would trust (models.py's save-then-start handoff assumes a
// present file means a usable file).
bool atomic_write(const std::string& path, const std::vector<char>& bytes) {
    std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            return false;
        }
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        out.flush();
        if (!out) {
            return false;
        }
    }
    std::remove(path.c_str());
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

}  // namespace

bool save_slot_state(llama_context* ctx, const std::vector<llama_token>& prefix_tokens,
                     const std::string& path) {
    if (!ctx || path.empty()) {
        return false;
    }
    const size_t state_bytes = llama_state_seq_get_size(ctx, /*seq_id=*/0);
    if (state_bytes == 0) {
        // Nothing resident -- "nothing saved", not an error.
        return false;
    }
    std::vector<uint8_t> state(state_bytes);
    if (llama_state_seq_get_data(ctx, state.data(), state_bytes, /*seq_id=*/0) != state_bytes) {
        return false;
    }

    std::vector<char> file;
    file.reserve(sizeof(kMagic) + 4 + 4 + prefix_tokens.size() * sizeof(llama_token) + 8 +
                 state_bytes);
    file.insert(file.end(), kMagic, kMagic + 4);
    auto push_u32 = [&file](uint32_t v) {
        unsigned char b[4] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8),
                              static_cast<unsigned char>(v >> 16), static_cast<unsigned char>(v >> 24)};
        file.insert(file.end(), b, b + 4);
    };
    auto push_u64 = [&file](uint64_t v) {
        unsigned char b[8];
        for (int i = 0; i < 8; ++i) {
            b[i] = static_cast<unsigned char>(v >> (8 * i));
        }
        file.insert(file.end(), b, b + 8);
    };
    push_u32(kVersion);
    push_u32(static_cast<uint32_t>(prefix_tokens.size()));
    const auto* toks = reinterpret_cast<const char*>(prefix_tokens.data());
    file.insert(file.end(), toks, toks + prefix_tokens.size() * sizeof(llama_token));
    push_u64(static_cast<uint64_t>(state_bytes));
    const auto* st = reinterpret_cast<const char*>(state.data());
    file.insert(file.end(), st, st + state_bytes);

    return atomic_write(path, file);
}

bool restore_slot_state(llama_context* ctx, Engine& engine, const std::string& path) {
    if (!ctx || path.empty()) {
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    char magic[4];
    if (!read_exact(in, magic, 4) || std::memcmp(magic, kMagic, 4) != 0) {
        return false;
    }
    uint32_t version = 0;
    if (!read_u32(in, version) || version != kVersion) {
        return false;
    }
    uint32_t n_tokens = 0;
    if (!read_u32(in, n_tokens) || n_tokens > (1u << 26)) {  // sanity: 64M tokens is far past any real state
        return false;
    }
    std::vector<llama_token> tokens(n_tokens);
    if (n_tokens > 0 && !read_exact(in, tokens.data(), static_cast<size_t>(n_tokens) * sizeof(llama_token))) {
        return false;
    }
    uint64_t state_bytes = 0;
    if (!read_u64(in, state_bytes) || state_bytes == 0 || state_bytes > (1ull << 40)) {
        return false;
    }
    std::vector<uint8_t> state(static_cast<size_t>(state_bytes));
    if (!read_exact(in, state.data(), static_cast<size_t>(state_bytes))) {
        return false;
    }

    // A state saved from a larger context (or a different model) must never
    // partially load -- set_data returns the bytes it actually copied, so any
    // shortfall means "restore refused", and we leave the KV untouched-empty
    // (the caller's cold-start path, identical to a missing file).
    const size_t copied = llama_state_seq_set_data(ctx, state.data(), state_bytes, /*seq_id=*/0);
    if (copied != state_bytes) {
        return false;
    }
    engine.adopt_restored_state(std::move(tokens));
    return true;
}

}  // namespace pleiades_engine
