# Phase 4 test suite. Mirrors llama.cpp's OWN test convention exactly
# (tests/CMakeLists.txt in the vendored submodule): plain C++ executables
# run via CTest, asserting and returning a nonzero exit code on failure --
# no external test framework, matching this repo's no-third-party-deps
# stance elsewhere (see pleiades/scheduler.py's hand-rolled cron).
#
# Fixture: a genuinely tiny, real, loadable GGUF is needed here (unlike
# tests/test_context_plan.py's Python make_gguf() helper, which writes
# metadata-only files with zero tensors -- fine for testing hparams
# parsing, but NOT loadable by llama_model_load_from_file). Rather than
# inventing a from-scratch synthetic-tensor generator, this reuses
# llama.cpp's own official CI fixture via its own download-models.cmake
# script: ggml-org/models' tinyllamas/stories15M-q4_0.gguf, hash-verified,
# a few MB. Same file llama.cpp's own test-thread-safety test uses.

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
  set(PLEIADES_TEST_MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
  set(PLEIADES_TEST_MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
  set(PLEIADES_TEST_MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
  set(PLEIADES_TEST_MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(PLEIADES_TEST_MODEL_DEST "${CMAKE_BINARY_DIR}/${PLEIADES_TEST_MODEL_NAME}")

add_test(NAME pleiades-test-download-model COMMAND ${CMAKE_COMMAND}
  -DDEST=${PLEIADES_TEST_MODEL_DEST}
  -DNAME=${PLEIADES_TEST_MODEL_NAME}
  -DHASH=${PLEIADES_TEST_MODEL_HASH}
  -P ${LLAMA_CPP_DIR}/cmake/download-models.cmake
)
set_tests_properties(pleiades-test-download-model PROPERTIES FIXTURES_SETUP pleiades-test-model)

function(pleiades_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE pleiades_engine)
  add_test(NAME ${name} COMMAND ${name} "${PLEIADES_TEST_MODEL_DEST}")
  set_tests_properties(${name} PROPERTIES FIXTURES_REQUIRED pleiades-test-model)
endfunction()

# Offline wrapper test for the real jinja chat-templating layer
# (chat_template.h/.cpp over llama.cpp's common_chat_*). No model needed:
# ChatTemplates::create(nullptr, override) renders/parses from a jinja string,
# so this runs even offline. Real cross-family, loaded-model tool-call parsing
# is proven end-to-end in test_http_tool_calls below.
add_executable(test_chat_template test_chat_template.cpp)
target_link_libraries(test_chat_template PRIVATE pleiades_engine)
add_test(NAME test_chat_template COMMAND test_chat_template)

pleiades_add_test(test_model_manager)
pleiades_add_test(test_context_governor)
pleiades_add_test(test_engine)
pleiades_add_test(test_resident_map)
pleiades_add_test(test_compaction_plan)

# Phase C: end-to-end HTTP tool-calling tests against REAL models.
#
# Unlike the offline wrapper test above, these boot the actual
# pleiades-engine-server binary as a subprocess and drive it over real HTTP.
# Two separate binaries prove real jinja tool-calling works cross-family (the
# whole point of Stage B):
#
#   test_http_tool_calls   - the Qwen family, with STRICT assertions tuned to
#                            Qwen2.5's exact tool output, plus the streaming +
#                            fail-loud-422 + prefix-reuse regressions.
#   test_http_cross_family - a NON-Qwen family (Llama-3.x by default), with
#                            family-agnostic assertions (the right tool fired
#                            with the requested city in valid-JSON arguments),
#                            proving the same machinery renders + parses a
#                            different family's OWN template + tool format.
#
# Each needs a GGUF whose template advertises tool-calling. Neither downloads a
# model: each points at the given path if it exists and SKIPs cleanly
# (SKIP_RETURN_CODE below) otherwise, so they run for real where a suitable
# model is resident and never fail a bare checkout.
#
# POSIX-only: both spawn the server binary via fork/execl and reap it via
# SIGTERM + waitpid (see ServerProcess in each file), none of which exists
# under MSVC/MinGW -- building them unconditionally would break `cmake
# --build` for the WHOLE engine on Windows (tests default ON in the parent
# CMakeLists). Gate the two targets, not the whole test suite: everything
# above still builds and runs on Windows. A Windows port would need a
# CreateProcess-based ServerProcess; until then these run only where the
# multi-hundred-MB dev-box fixtures they need actually live.
if (WIN32)
  message(STATUS "pleiades: skipping POSIX-only HTTP end-to-end tests (fork/execl-based) on Windows")
else()
set(PLEIADES_TEST_QWEN_GGUF "$ENV{HOME}/.anamnesis/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    CACHE FILEPATH "Qwen-family GGUF (with a tool chat_template) for the end-to-end HTTP tool-call test; SKIPs if absent")

# Non-Qwen cross-family model. Its OWN template must carry tool support (verify
# via the server's startup log: 'tool-calling: supported'). Default: the
# Llama-3.2-1B-Instruct GGUF resident on the dev box (its Llama-3.x template
# renders tools and the model emits a get_weather call). Override with
# -DPLEIADES_TEST_CROSS_GGUF=/path/to/model.gguf.
set(PLEIADES_TEST_CROSS_GGUF "$ENV{HOME}/.anamnesis/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    CACHE FILEPATH "Non-Qwen instruct GGUF (with a tool chat_template) for the cross-family HTTP tool-call test; SKIPs if absent")

add_executable(test_http_tool_calls test_http_tool_calls.cpp)
# pleiades_httplib gives httplib::Client (+ Threads); pleiades_engine
# propagates the vendored nlohmann/json include. Both targets are defined in
# the parent CMakeLists before add_subdirectory(tests), so they're in scope.
target_link_libraries(test_http_tool_calls PRIVATE pleiades_engine pleiades_httplib)
add_test(NAME test_http_tool_calls
  COMMAND test_http_tool_calls $<TARGET_FILE:pleiades-engine-server> "${PLEIADES_TEST_QWEN_GGUF}")
# 77 == the kSkip the test returns when the fixture model isn't present.
set_tests_properties(test_http_tool_calls PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_http_cross_family test_http_cross_family.cpp)
target_link_libraries(test_http_cross_family PRIVATE pleiades_engine pleiades_httplib)
add_test(NAME test_http_cross_family
  COMMAND test_http_cross_family $<TARGET_FILE:pleiades-engine-server> "${PLEIADES_TEST_CROSS_GGUF}")
set_tests_properties(test_http_cross_family PROPERTIES SKIP_RETURN_CODE 77)
endif()  # NOT WIN32 (POSIX-only HTTP e2e tests -- see the gate comment above)

# Phase E: elastic-context suite. argv[1] is the tiny fixture (auto-YaRN
# mechanics); argv[2] is a large-train-ctx model for the KV-preservation
# cases (defaults to the Llama-3.2-1B GGUF resident on the dev box; SKIPs
# cleanly when absent).
add_executable(test_elastic_context test_elastic_context.cpp)
target_link_libraries(test_elastic_context PRIVATE pleiades_engine)
set(ELASTIC_REAL "$ENV{HOME}/.anamnesis/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    CACHE FILEPATH "Large-train-ctx GGUF for KV-preservation tests; SKIPs if absent")
add_test(NAME test_elastic_context COMMAND test_elastic_context "${PLEIADES_TEST_MODEL_DEST}" "${ELASTIC_REAL}")
set_tests_properties(test_elastic_context PROPERTIES FIXTURES_REQUIRED pleiades-test-model SKIP_RETURN_CODE 77)
