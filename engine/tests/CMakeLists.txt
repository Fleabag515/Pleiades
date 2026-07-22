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

# Pure string test, no model needed -- still registered without the
# download fixture requirement so it runs even offline.
add_executable(test_chat_template test_chat_template.cpp)
target_link_libraries(test_chat_template PRIVATE pleiades_engine)
add_test(NAME test_chat_template COMMAND test_chat_template)

pleiades_add_test(test_model_manager)
pleiades_add_test(test_context_governor)
pleiades_add_test(test_engine)
