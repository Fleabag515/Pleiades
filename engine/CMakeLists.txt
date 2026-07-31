# Pleiades native inference engine -- Phase 1 skeleton, GPU-enabled as of the
# CUDA/MoE-offload/context-param-parity pass (see
# docs/specs/2026-07-21-native-inference-engine-design.md).
#
# See that doc for the full plan. This phase: link libllama directly (no
# HTTP transport concerns here), expose ModelManager/ContextGovernor/Engine,
# and prove it with a CLI smoke-test binary against a real small GGUF.
#
# Build (auto-detect backend, default):
#   cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build engine/build -j
#   engine/build/pleiades-engine-cli <model.gguf> "<prompt>" [n_predict]
#
# Build (explicit backend -- auto-detect never silently falls back to CPU for
# an explicit request; it errors instead if the toolchain isn't found):
#   cmake -S engine -B engine/build -DPLEIADES_ENGINE_BACKEND=cuda|hip|vulkan|metal|cpu
cmake_minimum_required(VERSION 3.21)
project(pleiades_engine LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# MSVC reports __cplusplus as 199711L regardless of /std:c++20 unless this is
# passed. That silently breaks the vendored llama.cpp: llama-chat.cpp guards its
# LU8() macro on `#if __cplusplus >= 202000L` to cast u8"" literals back to
# const char* (they are const char8_t* in C++20). With the wrong __cplusplus the
# guard picks the C++17 branch and every LU8() use fails to compile (C2664/C2280).
# Only bites here because CMAKE_CXX_STANDARD 20 above propagates into llama.cpp's
# subdirectory -- its own `target_compile_features(llama PRIVATE cxx_std_17)`
# sets a floor and cannot lower it back down.
if(MSVC)
  add_compile_options(/Zc:__cplusplus)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PLEIADES_ENGINE_BUILD_TESTS "Build engine/tests (downloads a tiny fixture GGUF)" ON)
include(CTest)

# -- Compute backend selection -----------------------------------------------
# Phase 9.3 (docs/specs/2026-07-21-native-inference-engine-design.md): one
# backend, chosen explicitly or auto-detected, replacing the old CUDA-only
# boolean below (which defaulted ON and used CACHE...FORCE -- actively wrong
# on a Mac or an AMD box, and the FORCE made a user's own -DGGML_VULKAN=ON
# collide with it). Exactly one GGML_* flag is set before add_subdirectory();
# ggml's own per-backend CMakeLists.txt (ggml/src/ggml-{cuda,hip,vulkan}/
# CMakeLists.txt) does the real toolchain discovery and kernel compilation.
#
# auto-detect NEVER silently falls back to CPU for an EXPLICIT backend
# request -- that's the exact -DGGML_HIPBLAS=on-silently-ignored footgun
# install.sh already had to learn to avoid (see that file's resolve_gpu()).
# An explicit backend with no toolchain is a hard configure-time error.
set(PLEIADES_ENGINE_BACKEND "auto" CACHE STRING "auto|cuda|hip|vulkan|metal|cpu")
set_property(CACHE PLEIADES_ENGINE_BACKEND PROPERTY STRINGS auto cuda hip vulkan metal cpu)

# Deprecated alias, kept for one release so existing build scripts/CI configs
# using the old boolean don't silently break. Only takes effect if the caller
# set it AND left PLEIADES_ENGINE_BACKEND at its "auto" default -- an explicit
# PLEIADES_ENGINE_BACKEND always wins.
if (DEFINED PLEIADES_ENGINE_CUDA AND PLEIADES_ENGINE_BACKEND STREQUAL "auto")
  message(DEPRECATION "PLEIADES_ENGINE_CUDA is deprecated -- use "
                       "-DPLEIADES_ENGINE_BACKEND=cuda (or =cpu) instead.")
  if (PLEIADES_ENGINE_CUDA)
    set(PLEIADES_ENGINE_BACKEND "cuda")
  else()
    set(PLEIADES_ENGINE_BACKEND "cpu")
  endif()
endif()

if (PLEIADES_ENGINE_BACKEND STREQUAL "auto")
  # Apple silicon: Metal is the only sane default (ggml's own GGML_METAL_DEFAULT
  # already prefers it on macOS; mirrored here explicitly so /props can report
  # a resolved, non-"auto" backend name -- see 9.2's --caps).
  if (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(PLEIADES_ENGINE_BACKEND "metal")
  else()
    # Preference order matches pleiades/runtime.py's _backend_priority() for
    # the equivalent prebuilt-binary case: cuda > hip > vulkan > cpu. Each
    # find_package call here matches what the corresponding ggml-<backend>/
    # CMakeLists.txt itself requires (see that file), not a looser guess --
    # a QUIET pass here that ggml's own REQUIRED pass would fail is exactly
    # the kind of "compiles until it doesn't, three directories down" bug
    # this whole restructure exists to avoid.
    find_package(CUDAToolkit QUIET)
    find_package(hip QUIET)
    find_package(hipblas QUIET)
    find_package(Vulkan COMPONENTS glslc QUIET)
    if (CUDAToolkit_FOUND)
      set(PLEIADES_ENGINE_BACKEND "cuda")
    elseif (hip_FOUND AND hipblas_FOUND)
      set(PLEIADES_ENGINE_BACKEND "hip")
    elseif (Vulkan_FOUND)
      set(PLEIADES_ENGINE_BACKEND "vulkan")
    else()
      set(PLEIADES_ENGINE_BACKEND "cpu")
    endif()
  endif()
  message(STATUS "PLEIADES_ENGINE_BACKEND=auto resolved to '${PLEIADES_ENGINE_BACKEND}'")
endif()

if (PLEIADES_ENGINE_BACKEND STREQUAL "cuda")
  find_package(CUDAToolkit QUIET)
  if (NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "PLEIADES_ENGINE_BACKEND=cuda requested but no CUDA "
                         "toolkit was found. Install it, or build a different "
                         "backend explicitly (-DPLEIADES_ENGINE_BACKEND=cpu|"
                         "vulkan|hip|metal).")
  endif()
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
  # Escape hatch for cross-building on a box with no GPU attached (or a
  # different GPU generation) -- pin explicitly rather than relying on
  # ggml-cuda's own "native" auto-detection. Unset by default.
  if (DEFINED PLEIADES_ENGINE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${PLEIADES_ENGINE_CUDA_ARCHITECTURES} CACHE STRING "" FORCE)
  endif()

elseif (PLEIADES_ENGINE_BACKEND STREQUAL "hip")
  find_package(hip QUIET)
  find_package(hipblas QUIET)
  if (NOT (hip_FOUND AND hipblas_FOUND))
    message(FATAL_ERROR "PLEIADES_ENGINE_BACKEND=hip requested but ROCm's "
                         "hip/hipblas CMake packages were not found. Install "
                         "ROCm, or build a different backend explicitly.")
  endif()
  set(GGML_HIP ON CACHE BOOL "" FORCE)
  # Unlike CUDA's "native" default, HIP has no reliable native-arch detection
  # across the whole ROCm/gfx matrix -- a release build MUST pin an explicit,
  # possibly multi-arch list (e.g. "gfx1030;gfx1100;gfx1101;gfx90a"). Mirrors
  # PLEIADES_ENGINE_CUDA_ARCHITECTURES above; ggml-hip/CMakeLists.txt forwards
  # its own GPU_TARGETS into CMAKE_HIP_ARCHITECTURES only when the latter is
  # unset, so setting it directly here (when the caller opts in) takes
  # precedence over that fallback.
  if (DEFINED PLEIADES_ENGINE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES ${PLEIADES_ENGINE_HIP_ARCHITECTURES} CACHE STRING "" FORCE)
  endif()

elseif (PLEIADES_ENGINE_BACKEND STREQUAL "vulkan")
  find_package(Vulkan COMPONENTS glslc QUIET)
  if (NOT Vulkan_FOUND)
    message(FATAL_ERROR "PLEIADES_ENGINE_BACKEND=vulkan requested but the "
                         "Vulkan SDK (loader + headers + glslc shader "
                         "compiler) was not found. Install it, or build a "
                         "different backend explicitly.")
  endif()
  set(GGML_VULKAN ON CACHE BOOL "" FORCE)

elseif (PLEIADES_ENGINE_BACKEND STREQUAL "metal")
  if (NOT APPLE)
    message(FATAL_ERROR "PLEIADES_ENGINE_BACKEND=metal only makes sense on macOS.")
  endif()
  set(GGML_METAL ON CACHE BOOL "" FORCE)
  # Mandatory for a redistributable binary: without this, ggml-metal.metal has
  # to sit next to the executable at a fixed relative path, which breaks the
  # moment the binary is packaged (desktop app bundling) or installed to a
  # managed runtime directory (pleiades runtime install) instead of run from
  # the build tree it was compiled in.
  set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)

elseif (PLEIADES_ENGINE_BACKEND STREQUAL "cpu")
  # Nothing to set -- every ggml GPU backend option already defaults OFF.

else()
  message(FATAL_ERROR "Unknown PLEIADES_ENGINE_BACKEND '${PLEIADES_ENGINE_BACKEND}' "
                       "-- expected one of: auto cuda hip vulkan metal cpu")
endif()

# We build libllama itself from the submodule, and hand-pick the SMALL slice of
# llama.cpp's `common/` needed for real jinja chat templating into our own
# `pleiades_chat` static lib below -- rather than turning on LLAMA_BUILD_COMMON,
# which would compile the whole `llama-common` target (arg.cpp CLI parsing,
# download.cpp/hf-cache.cpp HuggingFace networking, imatrix, llguidance,
# ngram/speculative machinery) that this deliberately-minimal engine doesn't
# want. Also skip llama.cpp's own tests/tools/examples/server -- library only.
set(LLAMA_BUILD_COMMON  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS  OFF CACHE BOOL "" FORCE)

# Phase 9.4.2 (docs/specs/2026-07-21-native-inference-engine-design.md):
# vision/mmproj support. Upstream's tools/mtmd/CMakeLists.txt is explicitly
# designed as a standalone public library -- it links only ggml/llama/Threads
# and carries its own hard assertion (further down in that file) that it must
# NOT link llama-common, so turning this on does not pull in the
# arg.cpp/download.cpp/etc. bulk LLAMA_BUILD_COMMON=OFF above is avoiding.
# Requires LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS both OFF to take the
# standalone-library branch (see llama.cpp/CMakeLists.txt) -- both already are.
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp)

set(LLAMA_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp)

# -- pleiades_chat: real per-model jinja chat templating + tool-call parsing --
#
# The exact minimal slice of llama.cpp's common/ that implements
# common_chat_templates_init/apply + common_chat_parse (Stage B). This set was
# derived empirically, not guessed: every candidate translation unit was
# compiled to an object and an nm-based undefined-symbol sweep computed the
# closure. Compiling these 15 upstream TUs into one static archive and linking
# the three public entry points pulled in ALL 15 (none dead), and left only
# `llama_*`/`ggml_*` (satisfied by libllama) plus seven tiny common.cpp helpers
# unresolved -- those seven are vendored verbatim in src/llama_common_min.cpp
# (see that file's header for why common.cpp is NOT compiled whole). This is the
# same "compile a vendored source slice into its own small static lib" pattern
# pleiades_httplib already uses below for cpp-httplib.
#
# What each file provides (why the link breaks without it):
#   chat.cpp                      - the public common_chat_* API itself
#   chat-auto-parser-generator.cpp- autoparser::peg_generator::generate_parser,
#                                   which DERIVES a PEG tool-call parser from the
#                                   model's own jinja at apply() time (this is
#                                   how cross-family parsing works without a
#                                   hand-written parser per model)
#   chat-auto-parser-helpers.cpp  - helpers the generator/analyzer use
#   chat-peg-parser.cpp           - common_chat_peg_parse + AST->common_chat_msg
#   chat-diff-analyzer.cpp        - streaming-diff + id-part helpers chat.cpp uses
#   peg-parser.cpp                - the generic PEG engine (arena parse/load/save)
#   json-schema-to-grammar.cpp    - tool-argument grammar generation
#   log.cpp                       - the LOG_* facility the whole cluster calls
#   unicode.cpp                   - common_parse_utf8_codepoint (peg + jinja)
#   jinja/{lexer,parser,runtime,   - llama.cpp's own jinja template engine
#          value,string,caps}.cpp    (this repo's real templater; NOT "minja")
add_library(pleiades_chat STATIC
  ${LLAMA_CPP_DIR}/common/chat.cpp
  ${LLAMA_CPP_DIR}/common/chat-auto-parser-generator.cpp
  ${LLAMA_CPP_DIR}/common/chat-auto-parser-helpers.cpp
  ${LLAMA_CPP_DIR}/common/chat-peg-parser.cpp
  ${LLAMA_CPP_DIR}/common/chat-diff-analyzer.cpp
  ${LLAMA_CPP_DIR}/common/peg-parser.cpp
  ${LLAMA_CPP_DIR}/common/json-schema-to-grammar.cpp
  ${LLAMA_CPP_DIR}/common/log.cpp
  ${LLAMA_CPP_DIR}/common/unicode.cpp
  ${LLAMA_CPP_DIR}/common/jinja/lexer.cpp
  ${LLAMA_CPP_DIR}/common/jinja/parser.cpp
  ${LLAMA_CPP_DIR}/common/jinja/runtime.cpp
  ${LLAMA_CPP_DIR}/common/jinja/value.cpp
  ${LLAMA_CPP_DIR}/common/jinja/string.cpp
  ${LLAMA_CPP_DIR}/common/jinja/caps.cpp
  src/llama_common_min.cpp
)
# PUBLIC include dirs so these upstream TUs find their own headers (common/,
# nlohmann in vendor/, ggml/include, include/); when pleiades_engine links this
# PRIVATE (below), it inherits these to compile chat_template.cpp, but does NOT
# re-export them -- so http_server.cpp and the tests never see chat.h's
# global `using json = nlohmann::ordered_json;` alias, which would collide with
# the engine's own `using json = nlohmann::json;`.
target_include_directories(pleiades_chat PUBLIC
  ${LLAMA_CPP_DIR}/common
  ${LLAMA_CPP_DIR}/vendor
  ${LLAMA_CPP_DIR}/ggml/include
  ${LLAMA_CPP_DIR}/include
)
target_link_libraries(pleiades_chat PUBLIC llama)
# Upstream common/ targets C++17; these TUs also compile clean at the engine's
# global C++20 (verified) -- this only records the floor.
target_compile_features(pleiades_chat PUBLIC cxx_std_17)

add_library(pleiades_engine STATIC
  src/model_manager.cpp
  src/context_governor.cpp
  src/prefix_cache.cpp
  src/engine.cpp
  src/chat_template.cpp
)
target_include_directories(pleiades_engine PUBLIC include)
# nlohmann/json.hpp is vendored inside the llama.cpp submodule (see the
# pleiades_httplib/pleiades-engine-server comment below for the full rationale
# on why it's used directly rather than turning on LLAMA_BUILD_COMMON) --
# chat_template.cpp needs it too (rendering/parsing the `tools`/`tool_calls`
# JSON), so it's PUBLIC on pleiades_engine itself rather than only on the
# server binary.
target_include_directories(pleiades_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp/vendor)
# pleiades_chat is PRIVATE: chat_template.cpp (compiled here) needs its headers
# and symbols, but downstream targets linking pleiades_engine must NOT inherit
# common/ on their include path (see the include-dir note on pleiades_chat).
# CMake still propagates the static-lib link dependency to the final binaries so
# the common_chat_* symbols resolve.
#
# mtmd is PUBLIC (like llama): model_manager.h/engine.h expose mtmd_context*/
# mtmd_input_chunks* in their own public API (opaquely, via forward-declared
# pointers -- see those headers), and http_server.cpp (which links
# pleiades_engine, even PRIVATE) needs mtmd.h/mtmd-helper.h's real
# declarations to actually call mtmd_tokenize() etc. A PUBLIC dependency here
# still reaches http_server.cpp's own include path even though ITS link to
# pleiades_engine is PRIVATE -- PRIVATE only stops that propagating one hop
# further, which nothing here needs.
target_link_libraries(pleiades_engine PUBLIC llama mtmd PRIVATE pleiades_chat)

add_executable(pleiades-engine-cli src/cli_main.cpp)
target_link_libraries(pleiades-engine-cli PRIVATE pleiades_engine)

# Phase 2 benchmark: ContextGovernor::resize() wall-clock latency vs. the
# Python elastic engine and native llama-server's restart-only option.
add_executable(pleiades-engine-bench-resize src/bench_resize.cpp)
target_link_libraries(pleiades-engine-bench-resize PRIVATE pleiades_engine)

add_executable(pleiades-engine-bench-ladder src/bench_ladder.cpp)
target_link_libraries(pleiades-engine-bench-ladder PRIVATE pleiades_engine)

# Phase 6 benchmark: cold full-prompt decode vs. warm prefix-cache-hit
# decode of the same prompt + one appended message.
add_executable(pleiades-engine-bench-prefix src/bench_prefix_cache.cpp)
target_link_libraries(pleiades-engine-bench-prefix PRIVATE pleiades_engine)

# Phase 3: HTTP transport shim. cpp-httplib and nlohmann/json are both
# vendored inside the llama.cpp submodule already (vendor/cpp-httplib,
# vendor/nlohmann) -- compiled/included directly here rather than turning on
# LLAMA_BUILD_COMMON (real chat templating comes from the pleiades_chat slice
# above, not the whole llama-common target).
find_package(Threads REQUIRED)

add_library(pleiades_httplib STATIC ${LLAMA_CPP_DIR}/vendor/cpp-httplib/httplib.cpp)
target_include_directories(pleiades_httplib PUBLIC ${LLAMA_CPP_DIR}/vendor/cpp-httplib)
target_link_libraries(pleiades_httplib PUBLIC Threads::Threads)
target_compile_features(pleiades_httplib PUBLIC cxx_std_17)

add_executable(pleiades-engine-server src/http_server.cpp)
target_include_directories(pleiades-engine-server PRIVATE ${LLAMA_CPP_DIR}/vendor)
# Phase 9.4.2: base64.hpp only (public-domain, header-only, no compiled .cpp
# of its own) -- for decoding data: URI image payloads in handle_chat(). Not
# LLAMA_BUILD_COMMON; see the many comments above on why that stays off.
target_include_directories(pleiades-engine-server PRIVATE ${LLAMA_CPP_DIR}/common)
target_link_libraries(pleiades-engine-server PRIVATE pleiades_engine pleiades_httplib)
# Baked in at configure time so a --caps query (9.2) can report the resolved
# backend without probing hardware again at runtime -- it's a build-time fact.
target_compile_definitions(pleiades-engine-server PRIVATE
  PLEIADES_ENGINE_BACKEND="${PLEIADES_ENGINE_BACKEND}")

if (PLEIADES_ENGINE_BUILD_TESTS)
  add_subdirectory(tests)
endif()
