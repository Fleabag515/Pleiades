// Minimal extraction of the handful of `common/common.cpp` helpers that
// llama.cpp's chat-templating cluster (chat.cpp + the jinja/peg/auto-parser
// files, compiled here as the `pleiades_chat` static lib) actually calls --
// and NOTHING else from that 73 KB grab-bag translation unit.
//
// WHY this file exists instead of compiling common.cpp whole:
//   common.cpp is one translation unit whose object references symbols from
//   common/sampling.cpp (common_sampler_*) and common/fit.cpp (fit_params_*)
//   in its own function bodies (common_init_from_params and friends). Linking
//   common.o therefore forces sampling.o + fit.o into the binary, which in
//   turn drag in speculative.cpp and the CLI/arg machinery -- the exact
//   heavyweight closure this deliberately-minimal engine does not build (see
//   engine/CMakeLists.txt: LLAMA_BUILD_COMMON is OFF). common.cpp also needs a
//   generated build-info.h. An empirical undefined-symbol sweep of the whole
//   chat cluster (nm over every cluster .o, minus llama_*/ggml_*/libc/std)
//   showed its ONLY external references into common.cpp are the seven
//   functions below, each small and self-contained (calls only llama_*/stdlib).
//   So we vendor exactly those seven, verbatim, rather than the file that
//   defines them, and let libllama satisfy the llama_*/ggml_* rest.
//
// PROVENANCE: copied byte-for-byte from third_party/llama.cpp/common/common.cpp
// at submodule commit c588c4f47 (tag b10103). Kept verbatim on purpose: a
// definition here MUST match common.h's declaration to link, so a future
// rebase that changes any of these signatures fails the build loudly (at the
// cluster's own call sites) -- silent behavioral drift is not possible. If the
// set of common.cpp symbols the cluster needs ever grows, the build fails with
// an undefined-symbol error naming the new function; add it here to match.

#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#    include <io.h>
#else
#    include <unistd.h>
#endif

// -- string utilities (common.cpp) ------------------------------------------

void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

std::string string_repeat(const std::string & str, size_t n) {
    if (n == 0) {
        return "";
    }

    std::string result;
    result.reserve(str.length() * n);

    for (size_t i = 0; i < n; ++i) {
        result += str;
    }

    return result;
}

// -- TTY utils (common.cpp) -------------------------------------------------

bool tty_can_use_colors() {
    // Check NO_COLOR environment variable (https://no-color.org/)
    if (const char * no_color = std::getenv("NO_COLOR")) {
        if (no_color[0] != '\0') {
            return false;
        }
    }

    // Check TERM environment variable
    if (const char * term = std::getenv("TERM")) {
        if (std::strcmp(term, "dumb") == 0) {
            return false;
        }
    }

    // Check if stdout and stderr are connected to a terminal
    // We check both because log messages can go to either
    bool stdout_is_tty = isatty(fileno(stdout));
    bool stderr_is_tty = isatty(fileno(stderr));

    return stdout_is_tty || stderr_is_tty;
}

// -- vocab utils (common.cpp) -----------------------------------------------
// Only the llama_vocab overloads are provided: the chat cluster never calls the
// llama_context overloads (verified by the undefined-symbol sweep), so the
// context variants declared in common.h are intentionally left undefined.

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
    }
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}
