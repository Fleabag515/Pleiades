# Pleiades native inference engine -- Phase 1 skeleton, GPU-enabled as of the
# CUDA/MoE-offload/context-param-parity pass (see
# docs/specs/2026-07-21-native-inference-engine-design.md).
#
# See that doc for the full plan. This phase: link libllama directly (no
# HTTP transport concerns here), expose ModelManager/ContextGovernor/Engine,
# and prove it with a CLI smoke-test binary against a real small GGUF.
#
# Build (GPU, default):
#   cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build engine/build -j
#   engine/build/pleiades-engine-cli <model.gguf> "<prompt>" [n_predict]
#
# Build (CPU-only, e.g. a box with no CUDA toolkit):
#   cmake -S engine -B engine/build -DPLEIADES_ENGINE_CUDA=OFF
cmake_minimum_required(VERSION 3.21)
project(pleiades_engine LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PLEIADES_ENGINE_BUILD_TESTS "Build engine/tests (downloads a tiny fixture GGUF)" ON)
include(CTest)

# -- GPU (CUDA) backend -----------------------------------------------------
# Mirrors ~/.pleiades/runtime/cuda-main's own build exactly (confirmed via
# its build.log: cmake was invoked with `-DGGML_CUDA=ON` and nothing else
# CUDA-arch-related -- CMAKE_CUDA_ARCHITECTURES was left UNINITIALIZED and
# ggml/src/ggml-cuda/CMakeLists.txt's own logic auto-resolved it to "native",
# which on this box's RTX 2080 Ti resolves to 75-real; see that build.log's
# "Using CMAKE_CUDA_ARCHITECTURES=75 CMAKE_CUDA_ARCHITECTURES_NATIVE=75-real"
# line). We do the same here: set GGML_CUDA before add_subdirectory() and
# otherwise let ggml-cuda's own CMakeLists.txt auto-detect the architecture
# -- do NOT hardcode 75 in this file, since that's specific to whatever GPU
# happens to be attached at build time, not a property of the engine itself.
# CUDA is not declared in this file's own `project()` LANGUAGES list (cuda-
# main's top-level CMakeLists.txt doesn't either -- it's `project("llama.cpp"
# C CXX)`); ggml-cuda's CMakeLists.txt calls enable_language(CUDA) itself
# once GGML_CUDA is ON, so nothing else is needed at this level.
option(PLEIADES_ENGINE_CUDA "Build the engine with CUDA (GGML_CUDA) backend support" ON)
if (PLEIADES_ENGINE_CUDA)
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
  # Escape hatch for cross-building on a box with no GPU attached (or a
  # different GPU generation) -- pin explicitly rather than relying on
  # "native" auto-detection. Unset by default.
  if (DEFINED PLEIADES_ENGINE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${PLEIADES_ENGINE_CUDA_ARCHITECTURES} CACHE STRING "" FORCE)
  endif()
endif()

# -- GPU (Vulkan) backend -----------------------------------------------------
# ONE binary for NVIDIA + AMD + Intel (anything with a Vulkan driver). Slower
# than a native CUDA/HIP build on the big vendors but it is the cross-vendor
# delivery story and the only GPU path on boxes where neither toolkit is
# installed. Requires the Vulkan SDK headers + a shader compiler at BUILD time
# (libvulkan-dev + glslangValidator on Debian/Ubuntu); only a Vulkan ICD at
# runtime. Verify a build locally before wiring it into installers -- Vulkan
# shader compilation has the most moving parts of the three backends.
option(PLEIADES_ENGINE_VULKAN "Build the engine with Vulkan (GGML_VULKAN) backend support" OFF)
if (PLEIADES_ENGINE_VULKAN)
  set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

# -- GPU (HIP/ROCm) backend ---------------------------------------------------
# AMD's native path. Requires a ROCm toolchain (hipcc, ROCm CMake) at build
# time -- CI-only for now; no ROCm machine is available to build/verify on
# (the AMD verification box is Ion's, currently unavailable). GGML's HIP
# backend builds through GGML_HIP + GGML_HIPBLAS with CMAKE_HIP_PLATFORM.
option(PLEIADES_ENGINE_HIP "Build the engine with AMD HIP/ROCm (GGML_HIP) backend support" OFF)
if (PLEIADES_ENGINE_HIP)
  set(GGML_HIP ON CACHE BOOL "" FORCE)
  set(GGML_HIPBLAS ON CACHE BOOL "" FORCE)
  if (NOT DEFINED CMAKE_HIP_ARCHITECTURES AND DEFINED ENV{GPU_ARCHS})
    set(CMAKE_HIP_ARCHITECTURES "$ENV{GPU_ARCHS}" CACHE STRING "" FORCE)
  endif()
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
  src/slot_state.cpp
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
target_link_libraries(pleiades_engine PUBLIC llama PRIVATE pleiades_chat)

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
target_link_libraries(pleiades-engine-server PRIVATE pleiades_engine pleiades_httplib)

if (PLEIADES_ENGINE_BUILD_TESTS)
  add_subdirectory(tests)
endif()
