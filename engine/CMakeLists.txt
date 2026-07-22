# Pleiades native inference engine -- Phase 1 skeleton.
#
# See docs/specs/2026-07-21-native-inference-engine-design.md for the full
# plan. This phase: link libllama directly (no server, no HTTP, no
# Anamnesis), expose ModelManager/ContextGovernor/Engine, and prove it with
# a CLI smoke-test binary against a real small GGUF.
#
# Build:
#   cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build engine/build -j
#   engine/build/pleiades-engine-cli <model.gguf> "<prompt>" [n_predict]
cmake_minimum_required(VERSION 3.21)
project(pleiades_engine LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PLEIADES_ENGINE_BUILD_TESTS "Build engine/tests (downloads a tiny fixture GGUF)" ON)
include(CTest)

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
)
target_include_directories(pleiades_engine PUBLIC include)
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
