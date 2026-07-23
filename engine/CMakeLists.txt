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

# We only need libllama itself for this phase (llama_sampler_* and the
# context/batch/decode API all live directly in include/llama.h) -- not
# `common` (chat templates, json-schema-grammar, CLI arg parsing), so skip
# building it and its heavier optional deps for now. Also skip llama.cpp's
# own tests/tools/examples/server -- we're consuming it as a library only.
set(LLAMA_BUILD_COMMON  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS  OFF CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp)

add_library(pleiades_engine STATIC
  src/model_manager.cpp
  src/context_governor.cpp
  src/engine.cpp
  src/chat_template.cpp
  src/tool_call_parser.cpp
)
target_include_directories(pleiades_engine PUBLIC include)
# nlohmann/json.hpp is vendored inside the llama.cpp submodule (see the
# pleiades_httplib/pleiades-engine-server comment below for the full
# rationale on why it's used directly rather than turning on
# LLAMA_BUILD_COMMON) -- chat_template.cpp/tool_call_parser.cpp need it too
# now (Phase 5 tool-calling: rendering/parsing the `tools`/`tool_calls`
# JSON), so it's PUBLIC on pleiades_engine itself rather than only on the
# server binary.
target_include_directories(pleiades_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp/vendor)
target_link_libraries(pleiades_engine PUBLIC llama)

add_executable(pleiades-engine-cli src/cli_main.cpp)
target_link_libraries(pleiades-engine-cli PRIVATE pleiades_engine)

# Phase 2 benchmark: ContextGovernor::resize() wall-clock latency vs. the
# Python elastic engine and native llama-server's restart-only option.
add_executable(pleiades-engine-bench-resize src/bench_resize.cpp)
target_link_libraries(pleiades-engine-bench-resize PRIVATE pleiades_engine)

add_executable(pleiades-engine-bench-ladder src/bench_ladder.cpp)
target_link_libraries(pleiades-engine-bench-ladder PRIVATE pleiades_engine)

# Phase 3: HTTP transport shim. cpp-httplib and nlohmann/json are both
# vendored inside the llama.cpp submodule already (vendor/cpp-httplib,
# vendor/nlohmann) -- compiled/included directly here rather than turning on
# LLAMA_BUILD_COMMON (see chat_template.h for why common/chat.cpp + minja
# are deliberately not wired in yet).
find_package(Threads REQUIRED)
set(LLAMA_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp)

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
