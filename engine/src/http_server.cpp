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
// not real per-model jinja -- see that file's header comment for why. Tool
// calling (Phase 5) is layered on top of that same stopgap: the request's
// `tools` array is rendered into the model's own detected dialect
// (ModelManager::tool_dialect(), see chat_template.h::ToolDialect) and its
// generated text is parsed back into structured `tool_calls`
// (tool_call_parser.h) for the non-streaming path -- see handle_chat().
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
//   --threads N           (default: auto-detected from hardware -- see
//                          detect_default_threads() below -- NOT ggml's
//                          hardcoded GGML_DEFAULT_N_THREADS of 4)
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
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "httplib.h"
#include "llama.h"
#include "nlohmann/json.hpp"
#include "pleiades_engine/chat_template.h"
#include "pleiades_engine/context_governor.h"
#include "pleiades_engine/engine.h"
#include "pleiades_engine/model_manager.h"
#include "pleiades_engine/tool_call_parser.h"

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

// Message parsing (role/content/tool_calls/tool_call_id round-trip, plus
// the JSON-null-content safety fix) lives in chat_template.cpp's
// parse_chat_messages() now -- shared with the unit tests in
// test_tool_calls.cpp, which a http_server.cpp-local static function
// couldn't be.

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
    auto messages = parse_chat_messages(body);
    if (messages.empty()) {
        res.status = 400;
        res.set_content(
            json{{"error", {{"message", "messages must not be empty"}, {"type", "invalid_request_error"}}}}.dump(),
            "application/json");
        return;
    }
    int n_predict = body.value("max_tokens", 512);
    bool stream = body.value("stream", false);

    // -- sampling parameters (parity with pleiades/inference/server.py, which
    // forwards these straight to llama-cpp-python's create_chat_completion).
    // Defaults below are that function's OWN defaults (llama-cpp-python
    // 0.3.x) so a request that omits a knob samples identically on both
    // engines. A present-but-null value is treated as "omitted" (the Python
    // server filters `None` out of the kwargs), and is read null-safely --
    // nlohmann's value() would throw type_error on a present JSON null. See
    // SamplingParams for how temperature <= 0 collapses to greedy.
    SamplingParams sampling;
    auto num = [&](const char* key, double fallback) -> double {
        return (body.contains(key) && body[key].is_number()) ? body[key].get<double>() : fallback;
    };
    sampling.temperature = static_cast<float>(num("temperature", 0.2));
    sampling.top_p = static_cast<float>(num("top_p", 0.95));
    sampling.top_k = static_cast<int>(num("top_k", 40));
    sampling.min_p = static_cast<float>(num("min_p", 0.05));
    sampling.typical_p = static_cast<float>(num("typical_p", 1.0));
    sampling.repeat_penalty = static_cast<float>(num("repeat_penalty", 1.0));
    sampling.presence_penalty = static_cast<float>(num("presence_penalty", 0.0));
    sampling.frequency_penalty = static_cast<float>(num("frequency_penalty", 0.0));
    if (body.contains("seed") && body["seed"].is_number_integer()) {
        sampling.seed = static_cast<uint32_t>(body["seed"].get<long long>());
    }
    // `stop` may be a single string or an array of strings (OpenAI allows both).
    if (body.contains("stop")) {
        const json& st = body["stop"];
        if (st.is_string()) {
            sampling.stop.push_back(st.get<std::string>());
        } else if (st.is_array()) {
            for (const auto& s : st) {
                if (s.is_string()) {
                    sampling.stop.push_back(s.get<std::string>());
                }
            }
        }
    }

    std::vector<json> tools;
    if (body.contains("tools") && body["tools"].is_array()) {
        for (const auto& t : body["tools"]) {
            tools.push_back(t);
        }
    }

    ToolDialect dialect = s.models.tool_dialect();
    if (!tools.empty() && dialect == ToolDialect::NONE) {
        std::fprintf(stderr,
                      "[pleiades-engine-server] WARNING: request offered %zu tool(s) but no tool-calling dialect "
                      "was detected for this model at load time -- ignoring tools, falling back to a plain "
                      "completion (no tool_calls will ever be parsed from this model's output; see "
                      "ModelManager::tool_dialect()).\n",
                      tools.size());
    }

    std::string prompt = format_chat_prompt(messages, tools, dialect, s.models.open_thinking());
    long created = static_cast<long>(std::time(nullptr));
    std::string id = now_id();

    if (!stream) {
        std::lock_guard<std::mutex> lock(s.mu);
        GenerationResult r = s.engine.complete(prompt, n_predict, sampling);
        // Prove-it-fires observability (Phase 6): report how much of the
        // prompt was served from the KV prefix cache vs. re-decoded. On a
        // pure-attention model a repeated persona/system prefix shows a high
        // `cached` here; on a hybrid-recurrent model (qwen35moe) it stays 0
        // by design (partial reuse is architecturally impossible there) and
        // the engine safely cold-decodes -- see the design doc's Phase 6.
        std::fprintf(stderr, "[pleiades-engine-server] chat: prompt_tokens=%d prefix_cached=%d decoded=%d\n",
                     r.n_prompt_tokens, r.n_prompt_cached, r.n_prompt_tokens - r.n_prompt_cached);

        json message;
        std::string finish_reason = "stop";

        if (dialect == ToolDialect::NONE) {
            message = {{"role", "assistant"}, {"content", r.text}};
        } else {
            ParsedToolCalls parsed = parse_tool_calls(r.text, dialect, tools);
            if (!parsed.ok) {
                // Fail loud and distinctly from a normal 200/"stop" response
                // -- see tool_call_parser.h's doc comment on
                // ParsedToolCalls. A 4xx/5xx here makes
                // pleiades/harness/llm.py::_post's urllib.request.urlopen()
                // raise HTTPError, which agent.py's loop already treats as
                // a retryable "model_error" (NOT as "no tool_calls -> final
                // answer") -- exactly the distinction that must never be
                // lost (Pleiades often runs with exec_policy "allow", so a
                // real intended action must never silently fail to fire).
                res.status = 422;
                res.set_content(
                    json{{"error",
                          {{"message", "model produced a malformed or incomplete tool call: " + parsed.error},
                           {"type", "tool_call_parse_error"}}}}
                        .dump(),
                    "application/json");
                return;
            }
            message = {{"role", "assistant"}};
            if (!parsed.calls.empty()) {
                // OpenAI convention: content is null (not "") when a turn
                // is ONLY tool calls with no leading prose -- and this
                // engine's own parse_chat_messages() is null-safe reading
                // it back on the next round (see chat_template.h).
                message["content"] = parsed.content.empty() ? json(nullptr) : json(parsed.content);
                json tc_array = json::array();
                for (size_t i = 0; i < parsed.calls.size(); ++i) {
                    tc_array.push_back({
                        {"id", "call_" + std::to_string(i)},
                        {"type", "function"},
                        {"function",
                         {{"name", parsed.calls[i].name}, {"arguments", parsed.calls[i].arguments.dump()}}},
                    });
                }
                message["tool_calls"] = tc_array;
                finish_reason = "tool_calls";
            } else {
                message["content"] = parsed.content;
            }
            if (!parsed.reasoning_content.empty()) {
                message["reasoning_content"] = parsed.reasoning_content;
            }
        }

        json resp = {
            {"id", id},
            {"object", "chat.completion"},
            {"created", created},
            {"model", s.alias},
            {"choices", json::array({{{"index", 0}, {"message", message}, {"finish_reason", finish_reason}}})},
            {"usage",
             {{"prompt_tokens", r.n_prompt_tokens},
              {"completion_tokens", r.n_generated_tokens},
              {"total_tokens", r.n_prompt_tokens + r.n_generated_tokens}}},
        };
        res.set_content(resp.dump(), "application/json");
        return;
    }

    // Formats one SSE `data: {chat.completion.chunk}\n\n` frame. Used only
    // synchronously below to PRE-BUILD the buffered-tool-stream frames before
    // any is written -- deliberately NOT reused inside the raw-stream
    // content-provider lambda, which runs after handle_chat() returns and
    // would capture this local by dangling reference.
    auto sse_chunk = [&](const json& delta, const char* finish_reason) -> std::string {
        json chunk = {
            {"id", id},         {"object", "chat.completion.chunk"}, {"created", created},
            {"model", s.alias}, {"choices", json::array({{{"index", 0}, {"delta", delta},
                                                           {"finish_reason", finish_reason ? json(finish_reason) : json(nullptr)}}})},
        };
        return "data: " + chunk.dump() + "\n\n";
    };

    // -- Phase A: buffered tool-capable streaming --------------------------- //
    //
    // The SSE token path streams delta.content only; it can't emit structured
    // delta.tool_calls. Streaming raw <tool_call> markup as visible content
    // is the specific silent failure this guards against: the streamed
    // interactive path (pleiades/engine.py::stream_events) reconstructs tool
    // calls from delta.tool_calls and never raises on plain content, so raw
    // markup there executes NO tool while looking like a normal answer.
    //
    // So when this request both offers tools AND the model has a known
    // dialect, run the whole turn buffered (reusing the exact non-streaming
    // complete() + parse_tool_calls() machinery), then replay the parsed
    // result as SSE: content first, then a SINGLE delta carrying the whole
    // tool_calls array (the OpenAI SDK the caller uses accumulates a
    // one-chunk tool_call fine via tc.index -- partial argument deltas are
    // not required). dialect == NONE or no tools offered falls through to the
    // unchanged raw token-streaming path below.
    if (!tools.empty() && dialect != ToolDialect::NONE) {
        GenerationResult r;
        {
            std::lock_guard<std::mutex> lock(s.mu);
            r = s.engine.complete(prompt, n_predict, sampling);
        }
        std::fprintf(stderr,
                     "[pleiades-engine-server] chat(stream,tools): prompt_tokens=%d prefix_cached=%d decoded=%d\n",
                     r.n_prompt_tokens, r.n_prompt_cached, r.n_prompt_tokens - r.n_prompt_cached);

        ParsedToolCalls parsed = parse_tool_calls(r.text, dialect, tools);
        if (!parsed.ok) {
            // Fail loud with the SAME 422/tool_call_parse_error as the
            // non-streaming path -- possible here precisely because generation
            // finished BEFORE any SSE byte was sent, so nothing is committed
            // yet. The caller's OpenAI client raises on the 422 and
            // engine.py's streamed branch falls back to a non-streamed
            // request (its existing `except Exception: streamed = False`).
            // A 200 that narrates the malformed markup as content is the one
            // outcome this must never produce (Pleiades often runs exec_policy
            // "allow" -- a real intended action must never silently not fire).
            res.status = 422;
            res.set_content(
                json{{"error",
                      {{"message", "model produced a malformed or incomplete tool call: " + parsed.error},
                       {"type", "tool_call_parse_error"}}}}
                    .dump(),
                "application/json");
            return;
        }

        std::vector<std::string> chunks;
        chunks.push_back(sse_chunk({{"role", "assistant"}}, nullptr));
        if (!parsed.reasoning_content.empty()) {
            chunks.push_back(sse_chunk({{"reasoning_content", parsed.reasoning_content}}, nullptr));
        }
        const char* finish_reason = "stop";
        if (!parsed.calls.empty()) {
            if (!parsed.content.empty()) {
                chunks.push_back(sse_chunk({{"content", parsed.content}}, nullptr));
            }
            json tc_array = json::array();
            for (size_t i = 0; i < parsed.calls.size(); ++i) {
                // Streaming delta tool_calls REQUIRE `index` (the field
                // engine.py accumulates on); the non-streaming message shape
                // does not. Otherwise identical to the non-streaming path.
                tc_array.push_back({
                    {"index", i},
                    {"id", "call_" + std::to_string(i)},
                    {"type", "function"},
                    {"function", {{"name", parsed.calls[i].name}, {"arguments", parsed.calls[i].arguments.dump()}}},
                });
            }
            chunks.push_back(sse_chunk({{"tool_calls", tc_array}}, nullptr));
            finish_reason = "tool_calls";
        } else if (!parsed.content.empty()) {
            chunks.push_back(sse_chunk({{"content", parsed.content}}, nullptr));
        }
        chunks.push_back(sse_chunk(json::object(), finish_reason));

        res.set_chunked_content_provider(
            "text/event-stream", [chunks = std::move(chunks)](size_t /*offset*/, httplib::DataSink& sink) {
                for (const std::string& c : chunks) {
                    if (!sink.write(c.data(), c.size())) {
                        return false;
                    }
                }
                static const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
        return;
    }

    // -- Raw token streaming (dialect == NONE or no tools offered) ---------- //
    //
    // Unchanged in spirit: one provider call runs the whole generation,
    // writing an SSE chunk per token (or per held-back chunk when `stop`
    // strings are set -- see Engine::generate) via the on_token callback,
    // then [DONE]. No tool_call parsing is needed on this path (no dialect or
    // no tools), and the PROMPT was still built tool-dialect-aware above, so
    // any assistant tool_calls history still round-trips for a streamed turn.
    res.set_chunked_content_provider(
        "text/event-stream", [&s, prompt, n_predict, id, created, sampling](size_t /*offset*/, httplib::DataSink& sink) {
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
            GenerationResult sr = s.engine.generate(
                prompt, n_predict, [&](const std::string& piece) { return write_chunk({{"content", piece}}, nullptr); },
                sampling);
            std::fprintf(stderr, "[pleiades-engine-server] chat(stream): prompt_tokens=%d prefix_cached=%d decoded=%d\n",
                         sr.n_prompt_tokens, sr.n_prompt_cached, sr.n_prompt_tokens - sr.n_prompt_cached);
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

// Sensible default thread count when --threads isn't passed explicitly.
// Without this, args.n_threads stays 0, ContextParams.n_threads stays 0 (its
// "leave llama_context_default_params() alone" sentinel -- see
// context_governor.h), and the context silently launches pinned at ggml's
// GGML_DEFAULT_N_THREADS (4 -- third_party/llama.cpp/ggml/include/ggml.h),
// regardless of the real machine's core count. That was confirmed to be the
// highest-ROI bug an engine review found: the daily-driver workload (a
// large MoE model with CPU-resident experts) is CPU-matmul-bound
// specifically on those 4 pinned threads.
//
// Mirrors pleiades/launch.py's own `threads = max((os.cpu_count() or 8) //
// 2, 4)` formula (its "physical cores beat SMT" comment) instead of
// inventing a new policy: std::thread::hardware_concurrency() is the C++
// analogue of os.cpu_count() (both report logical/SMT thread count, not
// physical cores, and both are allowed to report 0/None as "unknown" per
// their respective specs -- hence the "or 8"/"== 0 -> 8" fallback before
// halving). Using every logical thread for CPU matmuls oversubscribes real
// physical cores 2x on SMT/hyperthreaded hardware, which empirically loses
// throughput; half of the logical count approximates physical core count
// without a separate topology query. The floor of 4 keeps small/low-core
// boxes from being squeezed to 1-2 threads.
//
// This is resolved here, in the binary's arg handling, rather than inside
// ContextGovernor itself: ContextParams.n_threads == 0 is an intentional,
// tested sentinel ("leave library default alone") that cli_main.cpp and
// the bench binaries rely on to keep their pre-this-fix behavior byte for
// byte (see test_context_governor.cpp's defaults-match-upstream case).
// Only this binary -- the actual serving path real workloads run through --
// gets the auto-detected default; the CLI/bench tools are unaffected.
int detect_default_threads() {
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) {
        hc = 8;
    }
    return std::max<int>(static_cast<int>(hc) / 2, 4);
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
        // args.n_threads == 0 means --threads wasn't passed -- resolve a real
        // auto-detected default here rather than letting the 0 sentinel ride
        // through to ContextGovernor and silently pin at ggml's default of 4
        // (see detect_default_threads() above for the full rationale).
        cparams.n_threads = args.n_threads > 0 ? args.n_threads : detect_default_threads();
        std::fprintf(stderr, "[pleiades-engine] n_threads=%d n_threads_batch=%d (%s)\n",
                     cparams.n_threads, cparams.n_threads,
                     args.n_threads > 0 ? "explicit --threads" : "auto-detected");
        cparams.flash_attn_type = parse_flash_attn(args.flash_attn);
        cparams.type_k = parse_cache_type(args.cache_type_k);
        cparams.type_v = parse_cache_type(args.cache_type_v);

        ModelManager models;
        models.load(args.model, args.n_gpu_layers, args.n_cpu_moe, args.use_mlock);
        // Tool-calling dialect is sniffed once here (ModelManager::load(),
        // from the model's own tokenizer.chat_template) -- logged loudly
        // since a silent NONE means every /v1/chat/completions request
        // that offers `tools` will have them ignored entirely (see
        // handle_chat()'s WARNING log for that case).
        std::fprintf(stderr, "[pleiades-engine] tool-call dialect: %s%s\n",
                     tool_dialect_name(models.tool_dialect()),
                     models.open_thinking() ? " (open-thinking generation prompt)" : "");
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
