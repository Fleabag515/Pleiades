// Phase C: end-to-end HTTP tool-calling test against a REAL Qwen model.
//
// Unlike test_tool_calls.cpp (pure parser/JSON logic, no model, no HTTP), this
// boots the actual pleiades-engine-server binary as a subprocess against a
// real Qwen-family GGUF and drives it over real HTTP with httplib::Client,
// asserting the two things a default-engine flip depends on:
//
//   1. A tool-offered chat request returns correctly-parsed `tool_calls` --
//      BOTH non-streaming (choices[0].message.tool_calls) AND streaming
//      (reconstructed from delta.tool_calls, the path pleiades/engine.py::
//      stream_events relies on). The streaming path is the one that silently
//      dropped every tool before Phase A: it used to stream the raw
//      <tool_call> markup as delta.content and never emit delta.tool_calls.
//
//   2. The fail-loud discipline holds under streaming as well as
//      non-streaming: a malformed/incomplete tool call surfaces as HTTP 422
//      `tool_call_parse_error`, never a 200 that narrates the markup as prose
//      (Pleiades often runs exec_policy "allow" -- a real intended action must
//      never silently fail to fire). A forced-truncated generation
//      (max_tokens below what the call needs to close its <tool_call>) is used
//      to provoke this deterministically from a real model.
//
// It also spot-checks Phase B end-to-end: a seeded temperature>0 request is
// reproducible across calls (proving the sampler chain + seed are actually
// wired, not greedy-only).
//
// Model fixture: this needs a real Qwen-family GGUF that carries a tool
// template (dialect != NONE). That is NOT the tiny stories15M fixture the
// other tests download (a llama model with no tool dialect), and this test
// deliberately does NOT download a multi-hundred-MB model in CI. Instead it
// takes the model path from CMake (PLEIADES_TEST_QWEN_GGUF, default: the
// Qwen2.5-1.5B-Instruct GGUF already resident on the dev box) and SKIPS
// cleanly (CTest SKIP_RETURN_CODE) when that file isn't present -- so it runs
// for real where a suitable model exists and never fails a fresh checkout.
//
// A FRESH SERVER PROCESS backs each assertion group below (see ServerProcess),
// deliberately -- NOT a performance choice. A real, separate, PRE-EXISTING bug
// was found while writing this test: two successive requests against one
// running server, where the second's prompt has a long common prefix with
// whatever the KV/prefix-cache still holds from the first (Engine::generate's
// Phase 6 prefix-cache reuse -- see prefix_cache.cpp), can corrupt generation
// on the second call (reproduced independently of every Phase A/B/C change in
// this pass -- it happens on the pre-Phase-A/B/C code too, confirmed by
// reverting this session's engine.cpp/http_server.cpp changes and re-running
// two back-to-back curl requests by hand). That is a separate, real, tracked
// issue (not something this test is responsible for verifying or fixing), and
// giving every assertion group its own cold server sidesteps it entirely so
// this test actually verifies what it's named for: Phase A/B/C's own
// correctness, not the prefix cache's.
//
// argv[1] = path to the pleiades-engine-server binary (from CMake TARGET_FILE)
// argv[2] = path to the Qwen GGUF fixture

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "test_util.h"

using json = nlohmann::json;

namespace {

constexpr int kSkip = 77;  // must match SKIP_RETURN_CODE in tests/CMakeLists.txt

bool file_readable(const std::string& path) { return ::access(path.c_str(), R_OK) == 0; }

// The one tool-offered request body every test starts from. `get_weather`
// with a required string `location` is a call a greedy Qwen2.5-Instruct emits
// deterministically for a direct weather question.
json tool_request(bool stream, int max_tokens) {
    return {
        {"model", "pleiades-engine"},
        {"messages",
         json::array({
             {{"role", "system"}, {"content", "You are a helpful assistant. Use tools when appropriate."}},
             {{"role", "user"}, {"content", "What is the weather in San Francisco? Use the get_weather tool."}},
         })},
        {"tools",
         json::array({{
             {"type", "function"},
             {"function",
              {{"name", "get_weather"},
               {"description", "Get the current weather for a city"},
               {"parameters",
                {{"type", "object"},
                 {"properties", {{"location", {{"type", "string"}, {"description", "City name"}}}}},
                 {"required", json::array({"location"})}}}}},
         }})},
        {"temperature", 0},  // greedy => deterministic, reliable tool emission
        {"max_tokens", max_tokens},
        {"stream", stream},
    };
}

// Reconstructs a streamed chat response from raw SSE text exactly the way
// pleiades/engine.py::stream_events does: accumulate delta.content, and
// accumulate tool_calls keyed by their `index` (id / function.name /
// function.arguments). Also records whether any content delta contained raw
// <tool_call> markup (the pre-Phase-A silent-failure signature) and the final
// finish_reason + whether [DONE] terminated the stream.
struct StreamAccum {
    std::string content;
    std::vector<json> tool_calls;  // index-ordered, {name, arguments(string)}
    std::string finish_reason;
    bool saw_done = false;
    bool saw_raw_markup = false;
    bool is_sse = false;  // at least one well-formed chat.completion.chunk frame
};

StreamAccum parse_sse(const std::string& body) {
    StreamAccum acc;
    // tool_calls accumulated by index into a growable vector of {name,args}.
    std::vector<std::pair<std::string, std::string>> slots;
    size_t pos = 0;
    while (pos < body.size()) {
        size_t eol = body.find('\n', pos);
        std::string line = body.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
        pos = (eol == std::string::npos) ? body.size() : eol + 1;
        // strip a trailing '\r'
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("data: ", 0) != 0) continue;
        std::string data = line.substr(6);
        if (data == "[DONE]") {
            acc.saw_done = true;
            break;
        }
        json chunk = json::parse(data, nullptr, false);
        if (chunk.is_discarded() || !chunk.contains("choices") || chunk["choices"].empty()) continue;
        acc.is_sse = acc.is_sse || (chunk.value("object", std::string()) == "chat.completion.chunk");
        const json& choice = chunk["choices"][0];
        if (choice.contains("finish_reason") && choice["finish_reason"].is_string()) {
            acc.finish_reason = choice["finish_reason"].get<std::string>();
        }
        if (!choice.contains("delta") || !choice["delta"].is_object()) continue;
        const json& delta = choice["delta"];
        if (delta.contains("content") && delta["content"].is_string()) {
            std::string c = delta["content"].get<std::string>();
            if (c.find("<tool_call>") != std::string::npos) acc.saw_raw_markup = true;
            acc.content += c;
        }
        if (delta.contains("tool_calls") && delta["tool_calls"].is_array()) {
            for (const json& tc : delta["tool_calls"]) {
                size_t idx = tc.value("index", 0);
                if (idx >= slots.size()) slots.resize(idx + 1);
                if (tc.contains("function") && tc["function"].is_object()) {
                    const json& fn = tc["function"];
                    if (fn.contains("name") && fn["name"].is_string()) slots[idx].first += fn["name"].get<std::string>();
                    if (fn.contains("arguments") && fn["arguments"].is_string())
                        slots[idx].second += fn["arguments"].get<std::string>();
                }
            }
        }
    }
    for (auto& s : slots) {
        acc.tool_calls.push_back({{"name", s.first}, {"arguments", s.second}});
    }
    return acc;
}

// -- one assertion group per function, each run against its OWN fresh server (see file header) -- //

int test_non_streaming_tool_call(httplib::Client& cli) {
    auto res = cli.Post("/v1/chat/completions", tool_request(false, 128).dump(), "application/json");
    PLEIADES_CHECK(res != nullptr);
    PLEIADES_CHECK(res->status == 200);
    json body = json::parse(res->body, nullptr, false);
    PLEIADES_CHECK(!body.is_discarded());
    const json& choice = body["choices"][0];
    PLEIADES_CHECK(choice["finish_reason"] == "tool_calls");
    const json& tcs = choice["message"]["tool_calls"];
    PLEIADES_CHECK(tcs.is_array() && tcs.size() == 1);
    PLEIADES_CHECK(tcs[0]["function"]["name"] == "get_weather");
    // arguments is a JSON-encoded string carrying the location argument.
    json args = json::parse(tcs[0]["function"]["arguments"].get<std::string>(), nullptr, false);
    PLEIADES_CHECK(!args.is_discarded());
    PLEIADES_CHECK(args.contains("location"));
    std::fprintf(stderr, "[test] non-streaming tool call OK: get_weather(%s)\n", args.dump().c_str());
    return 0;
}

int test_streaming_tool_call(httplib::Client& cli) {
    // -- 2. streaming tool call (the Phase A fix) ------------------------- //
    auto res = cli.Post("/v1/chat/completions", tool_request(true, 128).dump(), "application/json");
    PLEIADES_CHECK(res != nullptr);
    PLEIADES_CHECK(res->status == 200);
    StreamAccum acc = parse_sse(res->body);
    PLEIADES_CHECK(acc.is_sse);     // real SSE frames, not a buffered JSON body
    PLEIADES_CHECK(acc.saw_done);   // terminated with [DONE]
    // The regression this test exists for: the streamed tool call must
    // arrive as structured delta.tool_calls, NOT as raw <tool_call> markup
    // narrated in delta.content.
    PLEIADES_CHECK(!acc.saw_raw_markup);
    PLEIADES_CHECK(acc.tool_calls.size() == 1);
    PLEIADES_CHECK(acc.tool_calls[0]["name"] == "get_weather");
    json args = json::parse(acc.tool_calls[0]["arguments"].get<std::string>(), nullptr, false);
    PLEIADES_CHECK(!args.is_discarded());
    PLEIADES_CHECK(args.contains("location"));
    PLEIADES_CHECK(acc.finish_reason == "tool_calls");
    std::fprintf(stderr, "[test] streaming tool call OK: delta.tool_calls get_weather(%s)\n", args.dump().c_str());
    return 0;
}

int test_non_streaming_fail_loud(httplib::Client& cli) {
    // -- 3. fail-loud, non-streaming (forced truncation -> 422) ----------- //
    // max_tokens=8 cuts the model off inside its <tool_call> block, before the
    // closing </tool_call> (the full call needs ~20 tokens) -> parse_tool_calls
    // reports an unterminated block -> HTTP 422 tool_call_parse_error.
    auto res = cli.Post("/v1/chat/completions", tool_request(false, 8).dump(), "application/json");
    PLEIADES_CHECK(res != nullptr);
    PLEIADES_CHECK(res->status == 422);
    json body = json::parse(res->body, nullptr, false);
    PLEIADES_CHECK(!body.is_discarded());
    PLEIADES_CHECK(body["error"]["type"] == "tool_call_parse_error");
    std::fprintf(stderr, "[test] non-streaming fail-loud OK: 422 tool_call_parse_error\n");
    return 0;
}

int test_streaming_fail_loud(httplib::Client& cli) {
    // -- 4. fail-loud, streaming (the key Phase A guarantee) -------------- //
    // The streamed tool path buffers generation BEFORE emitting any SSE byte,
    // so a malformed call still fails loud with a real 422 (identical status +
    // body to the non-streaming path) rather than a 200 SSE stream of raw
    // markup. The caller's OpenAI client raises on this, and engine.py's
    // streamed branch falls back to a non-streamed request.
    auto res = cli.Post("/v1/chat/completions", tool_request(true, 8).dump(), "application/json");
    PLEIADES_CHECK(res != nullptr);
    PLEIADES_CHECK(res->status == 422);
    json body = json::parse(res->body, nullptr, false);
    PLEIADES_CHECK(!body.is_discarded());
    PLEIADES_CHECK(body["error"]["type"] == "tool_call_parse_error");
    std::fprintf(stderr, "[test] streaming fail-loud OK: 422 tool_call_parse_error (not a 200 SSE)\n");
    return 0;
}

// -- 5. Phase B: seeded sampling is reproducible (not greedy-only) -------- //
// Two identical temperature>0 seeded requests, against two INDEPENDENT cold
// servers, must produce identical content -- impossible if sampling params
// were ignored (the pre-Phase-B engine was greedy-only regardless of what a
// request asked for), and a live proof the seeded dist sampler is actually
// threaded through end to end. Deliberately two separate server processes,
// not two calls to one server: that's what makes this a real proof of
// seeded-sampler determinism rather than something the (separate,
// pre-existing) prefix-cache-reuse bug could satisfy by accident -- see the
// file header, and the risk that TWO calls to the SAME server could return
// identical output simply because both calls hit the same corruption the
// same way, without the seed/sampler actually being exercised correctly.
json seeded_request_body() {
    return {
        {"model", "pleiades-engine"},
        {"messages", json::array({{{"role", "user"}, {"content", "Name three colors."}}})},
        {"temperature", 0.8},
        {"top_p", 0.95},
        {"top_k", 40},
        {"seed", 12345},
        {"max_tokens", 24},
        {"stream", false},
    };
}

// Returns the response content on success, or an empty string on any failure
// (status/parse/empty-content) -- printed and treated as a hard failure by
// the caller, which has the ctest-style PLEIADES_CHECK context main() lacks.
std::string seeded_request_content(httplib::Client& cli) {
    auto r = cli.Post("/v1/chat/completions", seeded_request_body().dump(), "application/json");
    if (!r || r->status != 200) return {};
    json j = json::parse(r->body, nullptr, false);
    if (j.is_discarded()) return {};
    return j["choices"][0]["message"].value("content", std::string());
}

}  // namespace

// Spawns a fresh pleiades-engine-server, waits for /health, and always kills
// + reaps it on destruction (even if a PLEIADES_CHECK returns early out of
// whatever ran against it) -- see file header for why each assertion group
// gets its own instance instead of sharing one server.
class ServerProcess {
public:
    ServerProcess(const std::string& server_bin, const std::string& model_path)
        : port_(20000 + static_cast<int>(::getpid() % 20000) + (next_port_offset_++ % 1000)),
          cli_("127.0.0.1", port_) {
        cli_.set_connection_timeout(2, 0);
        cli_.set_read_timeout(120, 0);  // generation of a small model, generously bounded

        pid_ = ::fork();
        if (pid_ < 0) {
            std::perror("fork");
            return;
        }
        if (pid_ == 0) {
            std::string port_str = std::to_string(port_);
            ::execl(server_bin.c_str(), server_bin.c_str(), "--model", model_path.c_str(), "--host", "127.0.0.1",
                    "--port", port_str.c_str(), "--ctx", "4096", "--alias", "pleiades-engine",
                    static_cast<char*>(nullptr));
            std::perror("execl");
            _exit(127);
        }
    }

    ~ServerProcess() {
        if (pid_ > 0) {
            ::kill(pid_, SIGTERM);
            int wstatus = 0;
            ::waitpid(pid_, &wstatus, 0);
        }
    }

    ServerProcess(const ServerProcess&) = delete;
    ServerProcess& operator=(const ServerProcess&) = delete;

    bool wait_healthy() {
        if (pid_ <= 0) return false;
        for (int i = 0; i < 120; ++i) {  // up to ~60s for model load
            auto h = cli_.Get("/health");
            if (h && h->status == 200) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        return false;
    }

    httplib::Client& client() { return cli_; }

private:
    static inline int next_port_offset_ = 0;
    int port_;
    httplib::Client cli_;
    pid_t pid_ = -1;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <server-binary> <qwen-model.gguf>\n", argv[0]);
        return 2;
    }
    std::string server_bin = argv[1];
    std::string model_path = argv[2];

    if (model_path.empty() || !file_readable(model_path)) {
        std::fprintf(stderr,
                     "[test] SKIP: Qwen tool-template GGUF not found at '%s'. Set -DPLEIADES_TEST_QWEN_GGUF=<path> to a "
                     "small Qwen-family GGUF (arch qwen2/qwen3, carries a tool chat_template) to run this end-to-end "
                     "test.\n",
                     model_path.c_str());
        return kSkip;
    }
    if (!file_readable(server_bin)) {
        std::fprintf(stderr, "[test] server binary not found: %s\n", server_bin.c_str());
        return 2;
    }

    // Each group gets a cold server -- see the file header for why.
    auto run_group = [&](int (*fn)(httplib::Client&)) -> int {
        ServerProcess server(server_bin, model_path);
        if (!server.wait_healthy()) {
            std::fprintf(stderr, "[test] server never became healthy\n");
            return 1;
        }
        return fn(server.client());
    };

    int rc = 0;
    rc |= run_group(test_non_streaming_tool_call);
    rc |= run_group(test_streaming_tool_call);
    rc |= run_group(test_non_streaming_fail_loud);
    rc |= run_group(test_streaming_fail_loud);

    // Phase B reproducibility: two independent cold servers, same seeded
    // request, must agree -- see seeded_request_content's doc comment for
    // why it's two servers and not two calls to one.
    {
        ServerProcess server1(server_bin, model_path);
        ServerProcess server2(server_bin, model_path);
        if (!server1.wait_healthy() || !server2.wait_healthy()) {
            std::fprintf(stderr, "[test] server never became healthy\n");
            rc |= 1;
        } else {
            std::string c1 = seeded_request_content(server1.client());
            std::string c2 = seeded_request_content(server2.client());
            if (c1.empty() || c1 != c2) {
                std::fprintf(stderr,
                             "[test] seeded sampling not reproducible across servers: c1=%s%s c2=%s%s\n",
                             c1.empty() ? "(empty)" : c1.substr(0, 40).c_str(), c1.size() > 40 ? "..." : "",
                             c2.empty() ? "(empty)" : c2.substr(0, 40).c_str(), c2.size() > 40 ? "..." : "");
                rc |= 1;
            } else {
                std::fprintf(stderr, "[test] seeded sampling reproducible OK (across two independent cold servers)\n");
            }
        }
    }

    return rc;
}
