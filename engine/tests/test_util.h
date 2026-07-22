#pragma once
// Deliberately not assert() -- assert() is a no-op under NDEBUG (which
// Release builds define), and these tests must actually verify something
// even when built the same way the shipped binaries are. See
// docs/specs/2026-07-21-native-inference-engine-design.md, Phase 4.
#include <cstdio>

#define PLEIADES_CHECK(cond)                                                         \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::fprintf(stderr, "%s:%d: CHECK FAILED: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                                \
        }                                                                            \
    } while (0)
