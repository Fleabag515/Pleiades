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
// See docs/specs/2026-07-21-native-inference-engine-design.md, Phase 3, and
// the GPU/MoE-offload/context-param-parity pass for the flag-based CLI
// below (previously 6 positional args).
//
// -- CLI --------------------------------------------------------------------
// Named flags, matching llama-server's own flag-based convention (and the
// exact short-flag spelling pleiades/launch.py already uses when invoking
// native llama-server -- see build_command() in that file -- so a future
// pass can point that code at this binary with minimal translation):
//
//   --model PATH        (required)
//   --host HOST          (default 127.0.0.1)
//   --port PORT           (default 8080)
//   --ctx N               (default 4096)          llama_context n_ctx
//   --ngl N               (default 0)             n_gpu_layers (negative = all)
//   --n-cpu-moe N         (default 0)              see ModelManager::load()
//   --ub N                (default 512)           n_ubatch
//   --batch N             (default 2048)          n_batch
//   --fa on|off|auto      (default auto)          flash attention
//   --ctk TYPE            (default f16)           KV cache type for K
//   --ctv TYPE            (default f16)           KV cache type for V
//   --alias NAME          (default pleiades-engine)
//   --threads N           (default 0 = library default)
//   --mlock               (flag, default off)
//
// Legacy positional mode is also still accepted for one transition period:
// `<model> <host> <port> [n_ctx] [n_gpu_layers] [alias]`, matching this
// binary's Phase 3/5 argv shape exactly -- pleiades/launch.py's
// PLEIADES_ENGINE=pleiades_native branch still invokes it this way today
// (see that file's own comment on why it isn't being touched in this pass:
// wiring autofit/MoE-placement through launch.py is explicitly out of scope
// for this pass). Positional mode is detected by argv[1] not starting with
// "-"; it maps onto the same Args with n_cpu_moe/ub/batch/fa/ctk/ctv/threads/
// mlock left at their defaults above (i.e. unchanged from this binary's
// pre-this-pass behavior). New callers should use flags.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

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

// -- CLI argument parsing ---------------------------------------------------

struct Args {
    std::string model;
    std::string host = "127.0.0.1";
    int port = 8080;
    int n_ctx = 4096;
    int n_gpu_layers = 0;
    int n_cpu_moe = 0;
    int n_ubatch = 512;
    int n_batch = 2048;
    int n_threads = 0;
    std::string flash_attn = "auto";
    std::string cache_type_k = "f16";
    std::string cache_type_v = "f16";
    std::string alias = "pleiades-engine";
    bool use_mlock = false;
};

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --model PATH [--host HOST] [--port PORT] [--ctx N] [--ngl N]\n"
        "       [--n-cpu-moe N] [--ub N] [--batch N] [--fa on|off|auto]\n"
        "       [--ctk TYPE] [--ctv TYPE] [--alias NAME] [--threads N] [--mlock]\n"
        "\n"
        "legacy positional mode (still accepted): %s <model.gguf> <host> <port>"
        " [n_ctx] [n_gpu_layers] [alias]\n",
        prog, prog);
}

// Mirrors common/arg.cpp's common_arg_utils::is_truthy/is_falsey/is_autoy
// (same accepted spellings) for the --fa flag.
llama_flash_attn_type parse_flash_attn(const std::string& value) {
    if (value == "on" || value == "enabled" || value == "true" || value == "1") {
        return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    if (value == "off" || value == "disabled" || value == "false" || value == "0") {
        return LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    if (value == "auto" || value == "-1") {
        return LLAMA_FLASH_ATTN_TYPE_AUTO;
    }
    throw std::runtime_error("unknown value for --fa: '" + value + "' (expected on/off/auto)");
}

// Mirrors common/arg.cpp's file-local `kv_cache_types` list and
// kv_cache_type_from_str()/ggml_type_name() lookup for --ctk/--ctv.
ggml_type parse_cache_type(const std::string& value) {
    static const std::vector<ggml_type> kTypes = {
        GGML_TYPE_F32,  GGML_TYPE_F16,   GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    };
    for (ggml_type t : kTypes) {
        if (value == ggml_type_name(t)) {
            return t;
        }
    }
    throw std::runtime_error("unsupported cache type: '" + value + "'");
}

Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        print_usage(argv[0]);
        std::exit(2);
    }

    std::string first = argv[1];
    bool flag_mode = !first.empty() && first[0] == '-';

    if (!flag_mode) {
        // Legacy positional mode: <model> <host> <port> [n_ctx] [n_gpu_layers] [alias].
        if (argc < 4) {
            print_usage(argv[0]);
            std::exit(2);
        }
        a.model = argv[1];
        a.host = argv[2];
        a.port = std::atoi(argv[3]);
        if (argc > 4) a.n_ctx = std::atoi(argv[4]);
        if (argc > 5) a.n_gpu_layers = std::atoi(argv[5]);
        if (argc > 6) a.alias = argv[6];
        return a;
    }

    auto next = [&](int& i) -> std::string {
        if (i + 1 >= argc) {
            throw std::runtime_error(std::string("missing value for ") + argv[i]);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model") {
            a.model = next(i);
        } else if (arg == "--host") {
            a.host = next(i);
        } else if (arg == "--port") {
            a.port = std::stoi(next(i));
        } else if (arg == "--ctx") {
            a.n_ctx = std::stoi(next(i));
        } else if (arg == "--ngl") {
            a.n_gpu_layers = std::stoi(next(i));
        } else if (arg == "--n-cpu-moe") {
            a.n_cpu_moe = std::stoi(next(i));
        } else if (arg == "--ub") {
            a.n_ubatch = std::stoi(next(i));
        } else if (arg == "--batch") {
            a.n_batch = std::stoi(next(i));
        } else if (arg == "--fa") {
            a.flash_attn = next(i);
        } else if (arg == "--ctk") {
            a.cache_type_k = next(i);
        } else if (arg == "--ctv") {
            a.cache_type_v = next(i);
        } else if (arg == "--alias") {
            a.alias = next(i);
        } else if (arg == "--threads") {
            a.n_threads = std::stoi(next(i));
        } else if (arg == "--mlock") {
            a.use_mlock = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown flag: " + arg);
        }
    }
    if (a.model.empty()) {
        throw std::runtime_error("--model is required");
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage(argv[0]);
        return 2;
    }

    llama_backend_init();

    try {
        ContextParams cparams;
        cparams.n_ubatch = args.n_ubatch;
        cparams.n_batch = args.n_batch;
        cparams.n_threads = args.n_threads;
        cparams.flash_attn_type = parse_flash_attn(args.flash_attn);
        cparams.type_k = parse_cache_type(args.cache_type_k);
        cparams.type_v = parse_cache_type(args.cache_type_v);

        ModelManager models;
        models.load(args.model, args.n_gpu_layers, args.n_cpu_moe, args.use_mlock);
        ContextGovernor ctx;
        ctx.create(models.model(), args.n_ctx, /*n_ctx_max=*/llama_model_n_ctx_train(models.model()), cparams);
        Engine engine(models, ctx);
        ServerState state{models, ctx, engine, args.alias, {}};

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

        std::printf(
            "[pleiades-engine-server] listening on %s:%d (n_ctx=%d, ceiling=%d, alias=%s, ngl=%d, "
            "n_cpu_moe=%d, ub=%d, batch=%d, fa=%s, ctk=%s, ctv=%s, mlock=%s)\n",
            args.host.c_str(), args.port, state.ctx.n_ctx(), state.ctx.n_ctx_max(), args.alias.c_str(),
            args.n_gpu_layers, args.n_cpu_moe, args.n_ubatch, args.n_batch, args.flash_attn.c_str(),
            args.cache_type_k.c_str(), args.cache_type_v.c_str(), args.use_mlock ? "on" : "off");
        svr.listen(args.host.c_str(), args.port);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        llama_backend_free();
        return 1;
    }

    llama_backend_free();
    return 0;
}
