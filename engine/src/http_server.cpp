// Phase 3: thin OpenAI-compatible HTTP shim over the Phase 1/2 core.
//
// Matches pleiades/inference/server.py's contract closely enough that
// pleiades/launch.py and Anamnesis's proxy don't need client-side changes:
//   POST /v1/chat/completions  (streaming + non-streaming)
//   POST /resize {"n_ctx": N}  -> {n_ctx, took_ms}
//   GET  /props                -> {n_ctx, n_ctx_train, n_ctx_max,
//                                   resizable, model_path, gears,
//                                   default_generation_settings}
//   GET  /health, /            -> {status:"ok", model}
//
// Not yet ported from server.py (out of Phase 3's explicit scope): /v1/models,
// /tokenize, /extras/tokenize/count, /metrics, prompt-overflow auto-upshift.
// kv_bytes_per_token is also omitted from /props -- that's hardware.py's KV-
// cost formula, not yet ported to C++; add it here once it is.
//
// Chat templating uses the hardcoded ChatML stopgap (chat_template.h/.cpp),
// not real per-model jinja -- see that file's header comment for why.
//
// One mutex serializes every request (chat AND resize), matching
// EngineState's own `self.lock` in server.py and llama.cpp's own "single
// stream per model" design -- a resize arriving mid-decode would otherwise
// free the llama_context out from under an in-flight llama_decode() call.
//
// See docs/specs/2026-07-21-native-inference-engine-design.md, Phase 3.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <mutex>
#include <string>

#include "httplib.h"
#include "llama.h"
#include "nlohmann/json.hpp"
#include "pleiades_engine/chat_template.h"
#include "pleiades_engine/context_governor.h"
#include "pleiades_engine/engine.h"
#include "pleiades_engine/model_manager.h"

using json = nlohmann::json;
using namespace pleiades_engine;

namespace {

struct ServerState {
    ModelManager& models;
    ContextGovernor& ctx;
    Engine& engine;
    std::string alias;
    std::mutex mu;  // serializes decode and resize -- see file header note.
};

json props_json(ServerState& s) {
    std::vector<int> gears;
    for (int g : CONTEXT_GEARS) {
        if (s.ctx.n_ctx_max() == 0 || g <= s.ctx.n_ctx_max()) {
            gears.push_back(g);
        }
    }
    return {
        {"n_ctx", s.ctx.n_ctx()},
        {"n_ctx_train", llama_model_n_ctx_train(s.models.model())},
        {"n_ctx_max", s.ctx.n_ctx_max()},
        {"resizable", true},
        {"model_path", s.models.path()},
        {"gears", gears},
        {"default_generation_settings", {{"n_ctx", s.ctx.n_ctx()}}},
    };
}

std::vector<ChatMessage> parse_messages(const json& body) {
    std::vector<ChatMessage> out;
    for (const auto& m : body.value("messages", json::array())) {
        out.push_back({m.value("role", "user"), m.value("content", "")});
    }
    return out;
}

std::string now_id() {
    static int counter = 0;
    return "chatcmpl-pleiades-" + std::to_string(++counter);
}

void handle_chat(ServerState& s, const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception&) {
        res.status = 400;
        res.set_content(json{{"error", {{"message", "invalid JSON body"}, {"type", "invalid_request_error"}}}}.dump(),
                         "application/json");
        return;
    }
    auto messages = parse_messages(body);
    if (messages.empty()) {
        res.status = 400;
        res.set_content(
            json{{"error", {{"message", "messages must not be empty"}, {"type", "invalid_request_error"}}}}.dump(),
            "application/json");
        return;
    }
    int n_predict = body.value("max_tokens", 512);
    bool stream = body.value("stream", false);
    std::string prompt = format_chatml(messages);
    long created = static_cast<long>(std::time(nullptr));
    std::string id = now_id();

    if (!stream) {
        std::lock_guard<std::mutex> lock(s.mu);
        GenerationResult r = s.engine.complete(prompt, n_predict);
        json resp = {
            {"id", id},
            {"object", "chat.completion"},
            {"created", created},
            {"model", s.alias},
            {"choices", json::array({{{"index", 0},
                                       {"message", {{"role", "assistant"}, {"content", r.text}}},
                                       {"finish_reason", "stop"}}})},
            {"usage",
             {{"prompt_tokens", r.n_prompt_tokens},
              {"completion_tokens", r.n_generated_tokens},
              {"total_tokens", r.n_prompt_tokens + r.n_generated_tokens}}},
        };
        res.set_content(resp.dump(), "application/json");
        return;
    }

    // Streaming: one provider call runs the whole generation, writing an
    // SSE chunk per token via the on_token callback, then [DONE].
    res.set_chunked_content_provider(
        "text/event-stream", [&s, messages, prompt, n_predict, id, created](size_t /*offset*/, httplib::DataSink& sink) {
            std::lock_guard<std::mutex> lock(s.mu);
            auto write_chunk = [&](const json& delta, const char* finish_reason) {
                json chunk = {
                    {"id", id},         {"object", "chat.completion.chunk"}, {"created", created},
                    {"model", s.alias}, {"choices", json::array({{{"index", 0}, {"delta", delta},
                                                                   {"finish_reason", finish_reason ? json(finish_reason) : json(nullptr)}}})},
                };
                std::string data = "data: " + chunk.dump() + "\n\n";
                return sink.write(data.data(), data.size());
            };
            write_chunk({{"role", "assistant"}}, nullptr);
            s.engine.generate(prompt, n_predict, [&](const std::string& piece) {
                return write_chunk({{"content", piece}}, nullptr);
            });
            write_chunk(json::object(), "stop");
            static const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();
            return true;
        });
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <host> <port> [n_ctx] [n_gpu_layers] [alias]\n", argv[0]);
        return 2;
    }
    std::string model_path = argv[1];
    std::string host = argv[2];
    int port = std::atoi(argv[3]);
    int n_ctx = argc > 4 ? std::atoi(argv[4]) : 4096;
    int n_gpu_layers = argc > 5 ? std::atoi(argv[5]) : 0;
    std::string alias = argc > 6 ? argv[6] : "pleiades-engine";

    llama_backend_init();

    try {
        ModelManager models;
        models.load(model_path, n_gpu_layers);
        ContextGovernor ctx;
        ctx.create(models.model(), n_ctx, /*n_ctx_max=*/llama_model_n_ctx_train(models.model()));
        Engine engine(models, ctx);
        ServerState state{models, ctx, engine, alias, {}};

        httplib::Server svr;

        svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
            res.set_content(json{{"status", "ok"}, {"model", state.alias}}.dump(), "application/json");
        });
        svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
            res.set_content(json{{"status", "ok"}, {"model", state.alias}}.dump(), "application/json");
        });
        svr.Get("/props", [&](const httplib::Request&, httplib::Response& res) {
            std::lock_guard<std::mutex> lock(state.mu);
            res.set_content(props_json(state).dump(), "application/json");
        });
        svr.Post("/resize", [&](const httplib::Request& req, httplib::Response& res) {
            json body;
            try {
                body = json::parse(req.body);
            } catch (const std::exception&) {
                res.status = 400;
                res.set_content(json{{"error", "n_ctx (positive int) required"}}.dump(), "application/json");
                return;
            }
            if (!body.contains("n_ctx") || !body["n_ctx"].is_number_integer() || body["n_ctx"].get<int>() <= 0) {
                res.status = 400;
                res.set_content(json{{"error", "n_ctx (positive int) required"}}.dump(), "application/json");
                return;
            }
            std::lock_guard<std::mutex> lock(state.mu);
            int requested = body["n_ctx"].get<int>();
            if (state.ctx.clamp_gear(requested) == state.ctx.n_ctx()) {
                res.set_content(json{{"n_ctx", state.ctx.n_ctx()}, {"took_ms", 0}, {"note", "already there"}}.dump(),
                                 "application/json");
                return;
            }
            auto t0 = std::chrono::steady_clock::now();
            int actual = state.ctx.resize(requested);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            res.set_content(json{{"n_ctx", actual}, {"took_ms", static_cast<int>(ms)}}.dump(), "application/json");
        });
        svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
            try {
                handle_chat(state, req, res);
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(json{{"error", {{"message", e.what()}, {"type", "server_error"}}}}.dump(),
                                 "application/json");
            }
        });

        std::printf("[pleiades-engine-server] listening on %s:%d (n_ctx=%d, ceiling=%d, alias=%s)\n",
                    host.c_str(), port, state.ctx.n_ctx(), state.ctx.n_ctx_max(), alias.c_str());
        svr.listen(host.c_str(), port);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        llama_backend_free();
        return 1;
    }

    llama_backend_free();
    return 0;
}
