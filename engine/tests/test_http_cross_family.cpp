// Stage B: end-to-end proof that real per-model jinja templating + auto-derived
// tool-call parsing work for a NON-Qwen model family, over real HTTP.
//
// test_http_tool_calls.cpp already proves the Qwen family end-to-end with strict
// assertions tailored to Qwen2.5's exact tool output ({"location": ...}). This
// test proves the SAME machinery (ModelManager::chat_templates() ->
// RenderedChat::apply/parse, backed by llama.cpp's common_chat_*) renders + parses
// a genuinely different family's OWN template + tool format -- the whole point of
// Stage B (cross-family, not two hand-coded Qwen dialects).
//
// It is deliberately family-AGNOSTIC in what it asserts: a small instruct model
// may nest or reshape the arguments object (a weak-model artifact, faithfully
// parsed), so rather than pin an exact key path this checks the durable,
// correct properties -- the tool call fired, with the right function name, valid
// JSON arguments, and the requested city present somewhere inside them. That is
// exactly "real cross-family tool-calling works", without over-fitting to one
// small model's argument shape.
//
// Model fixture: a non-Qwen instruct GGUF whose OWN template carries tool
// support (verify via the server's startup log: "tool-calling: supported").
// Default: the Llama-3.2-1B-Instruct GGUF resident on the dev box (its
// Llama-3.x template renders tools and the model emits a get_weather call). As
// with test_http_tool_calls, this SKIPs cleanly when the fixture is absent.
//
// argv[1] = path to the pleiades-engine-server binary (from CMake TARGET_FILE)
// argv[2] = path to the non-Qwen GGUF fixture

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

// Recursively: does any string value inside `j` contain `needle`? Family-
// agnostic way to assert "the tool call carries the requested city" regardless
// of whether the model put it at arguments.location or nested it somewhere.
bool json_has_substring(const json& j, const std::string& needle) {
    if (j.is_string()) {
        return j.get<std::string>().find(needle) != std::string::npos;
    }
    if (j.is_object() || j.is_array()) {
        for (const auto& v : j) {
            if (json_has_substring(v, needle)) {
                return true;
            }
        }
    }
    return false;
}

json weather_request(bool stream, int max_tokens) {
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
        {"temperature", 0},
        {"max_tokens", max_tokens},
        {"stream", stream},
    };
}

// -- assertion groups (each runs against its own fresh server) -------------- //

int test_plain_completion(httplib::Client& cli) {
    // No tools: proves the non-Qwen template renders a plain turn and the reply
    // comes back as ordinary content (not mis-parsed or errored).
    json req = {
        {"model", "pleiades-engine"},
        {"messages", json::array({{{"role", "user"}, {"content", "Name three colors."}}})},
        {"temperature", 0},
        {"max_tokens", 32},
        {"stream", false},
    };
    auto res = cli.Post("/v1/chat/completions", req.dump(), "application/json");
    PLEIADES_CHECK(res != nullptr);
    PLEIADES_CHECK(res->status == 200);
    json body = json::parse(res->body, nullptr, false);
    PLEIADES_CHECK(!body.is_discarded());
    const json& choice = body["choices"][0];
    PLEIADES_CHECK(choice["finish_reason"] == "stop");
    PLEIADES_CHECK(choice["message"]["content"].is_string());
    PLEIADES_CHECK(!choice["message"]["content"].get<std::string>().empty());
    std::fprintf(stderr, "[test] cross-family plain completion OK\n");
    return 0;
}

int test_non_streaming_tool_call(httplib::Client& cli) {
    auto res = cli.Post("/v1/chat/completions", weather_request(false, 128).dump(), "application/json");
    PLEIADES_CHECK(res != nullptr);
    PLEIADES_CHECK(res->status == 200);
    json body = json::parse(res->body, nullptr, false);
    PLEIADES_CHECK(!body.is_discarded());
    const json& choice = body["choices"][0];
    PLEIADES_CHECK(choice["finish_reason"] == "tool_calls");
    const json& tcs = choice["message"]["tool_calls"];
    PLEIADES_CHECK(tcs.is_array() && tcs.size() == 1);
    PLEIADES_CHECK(tcs[0]["function"]["name"] == "get_weather");
    json args = json::parse(tcs[0]["function"]["arguments"].get<std::string>(), nullptr, false);
    PLEIADES_CHECK(!args.is_discarded());
    PLEIADES_CHECK(json_has_substring(args, "San Francisco"));  // city carried through, family-agnostic
    std::fprintf(stderr, "[test] cross-family non-streaming tool call OK: get_weather(%s)\n", args.dump().c_str());
    return 0;
}

struct StreamAccum {
    std::vector<std::pair<std::string, std::string>> tool_slots;  // (name, arguments)
    std::string finish_reason;
    bool saw_done = false;
    bool saw_raw_markup = false;
    bool is_sse = false;
};

StreamAccum parse_sse(const std::string& body) {
    StreamAccum acc;
    size_t pos = 0;
    while (pos < body.size()) {
        size_t eol = body.find('\n', pos);
        std::string line = body.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
        pos = (eol == std::string::npos) ? body.size() : eol + 1;
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
            if (delta["content"].get<std::string>().find("<tool_call>") != std::string::npos) acc.saw_raw_markup = true;
        }
        if (delta.contains("tool_calls") && delta["tool_calls"].is_array()) {
            for (const json& tc : delta["tool_calls"]) {
                size_t idx = tc.value("index", 0);
                if (idx >= acc.tool_slots.size()) acc.tool_slots.resize(idx + 1);
                if (tc.contains("function") && tc["function"].is_object()) {
                    const json& fn = tc["function"];
                    if (fn.contains("name") && fn["name"].is_string()) acc.tool_slots[idx].first += fn["name"].get<std::string>();
                    if (fn.contains("arguments") && fn["arguments"].is_string())
                        acc.tool_slots[idx].second += fn["arguments"].get<std::string>();
                }
            }
        }
    }
    return acc;
}

int test_streaming_tool_call(httplib::Client& cli) {
    auto res = cli.Post("/v1/chat/completions", weather_request(true, 128).dump(), "application/json");
    PLEIADES_CHECK(res != nullptr);
    PLEIADES_CHECK(res->status == 200);
    StreamAccum acc = parse_sse(res->body);
    PLEIADES_CHECK(acc.is_sse);
    PLEIADES_CHECK(acc.saw_done);
    PLEIADES_CHECK(!acc.saw_raw_markup);  // structured delta.tool_calls, not raw markup as content
    PLEIADES_CHECK(acc.tool_slots.size() == 1);
    PLEIADES_CHECK(acc.tool_slots[0].first == "get_weather");
    json args = json::parse(acc.tool_slots[0].second, nullptr, false);
    PLEIADES_CHECK(!args.is_discarded());
    PLEIADES_CHECK(json_has_substring(args, "San Francisco"));
    PLEIADES_CHECK(acc.finish_reason == "tool_calls");
    std::fprintf(stderr, "[test] cross-family streaming tool call OK: delta.tool_calls get_weather(%s)\n",
                 args.dump().c_str());
    return 0;
}

}  // namespace

class ServerProcess {
public:
    ServerProcess(const std::string& server_bin, const std::string& model_path)
        : port_(21000 + static_cast<int>(::getpid() % 20000) + (next_port_offset_++ % 1000)), cli_("127.0.0.1", port_) {
        cli_.set_connection_timeout(2, 0);
        cli_.set_read_timeout(120, 0);
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
        for (int i = 0; i < 120; ++i) {
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
        std::fprintf(stderr, "usage: %s <server-binary> <non-qwen-model.gguf>\n", argv[0]);
        return 2;
    }
    std::string server_bin = argv[1];
    std::string model_path = argv[2];

    if (model_path.empty() || !file_readable(model_path)) {
        std::fprintf(stderr,
                     "[test] SKIP: non-Qwen tool-template GGUF not found at '%s'. Set "
                     "-DPLEIADES_TEST_CROSS_GGUF=<path> to a non-Qwen instruct GGUF whose template carries tool "
                     "support to run this cross-family end-to-end test.\n",
                     model_path.c_str());
        return kSkip;
    }
    if (!file_readable(server_bin)) {
        std::fprintf(stderr, "[test] server binary not found: %s\n", server_bin.c_str());
        return 2;
    }

    auto run_group = [&](int (*fn)(httplib::Client&)) -> int {
        ServerProcess server(server_bin, model_path);
        if (!server.wait_healthy()) {
            std::fprintf(stderr, "[test] server never became healthy\n");
            return 1;
        }
        return fn(server.client());
    };

    int rc = 0;
    rc |= run_group(test_plain_completion);
    rc |= run_group(test_non_streaming_tool_call);
    rc |= run_group(test_streaming_tool_call);
    return rc;
}
