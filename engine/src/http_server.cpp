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
// Chat templating (Stage B) uses REAL per-model jinja: the model's own
// GGUF-embedded tokenizer.chat_template is compiled once at load
// (ModelManager::chat_templates(), backed by llama.cpp's common_chat_*
// machinery -- see chat_template.h), and EVERY request is rendered through it
// -- so tools, tool_calls history, and reasoning are emitted in whatever native
// format each model family's own template specifies, not the two hand-coded
// Qwen dialects the engine used before. The model's completion is parsed back
// into structured content/reasoning_content/tool_calls by that same resolved
// format (RenderedChat::parse) -- for any family llama.cpp understands, not
// just Qwen. A finished generation that can't be parsed into the template's
// format (e.g. a tool call truncated by max_tokens) throws, which handle_chat
// surfaces as a distinct HTTP 422 rather than a 200 that narrates broken markup
// as prose (Pleiades often runs exec_policy "allow" -- a real intended action
// must never silently fail to fire).
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

#include "ggml-backend.h"
#include "httplib.h"
#include "llama.h"
#include "nlohmann/json.hpp"
#include "pleiades_engine/chat_template.h"
#include "pleiades_engine/context_governor.h"
#include "pleiades_engine/engine.h"
#include "pleiades_engine/model_manager.h"
#include "pleiades_engine/slot_state.h"

using json = nlohmann::json;
using namespace pleiades_engine;

namespace {

struct ServerState {
    ModelManager& models;
    ContextGovernor& ctx;
    Engine& engine;
    std::string alias;
    std::mutex mu;  // serializes decode and resize -- see file header note.
    // -- single-slot queue (Phase 1 parity: request queue + busy reporting) --
    // One generation at a time (one character = one conversation through its
    // Anamnesis proxy), but concurrent requests now WAIT in a bounded queue
    // instead of piling up on the mutex and getting connection drops, and a
    // caller that can't be served gets an honest 429 + Retry-After instead of
    // an opaque timeout. `queued` counts waiters so the bound holds.
    int busy = 0;
    int queued = 0;
    int queue_depth = 8;
    int queue_timeout_sec = 30;
    std::condition_variable slot_cv;
    // -- slot persistence / chunk reuse reporting --
    std::string slot_save_path;  // empty = /slots endpoints return 501
    int cache_reuse = 0;
    bool has_state = false;      // anything resident since boot (for /props)
};

// RAII single-slot lease. acquire() waits up to queue_timeout_sec for the
// engine to go idle, refusing early once `queue_depth` waiters are already
// queued; the destructor releases and wakes the next waiter. Release happens
// even when a streaming provider exits early (client disconnect), because the
// lease is captured by the provider lambda and destroyed with it.
struct SlotLease {
    explicit SlotLease(ServerState& s) : s_(s) {}

    bool acquire() {
        std::unique_lock<std::mutex> lock(s_.mu);
        ++s_.queued;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(s_.queue_timeout_sec);
        while (s_.busy) {
            if (s_.queued > s_.queue_depth ||
                s_.slot_cv.wait_until(lock, deadline) == std::cv_status::timeout) {
                --s_.queued;
                return false;
            }
        }
        --s_.queued;
        s_.busy = 1;
        held_ = true;
        return true;
    }

    ~SlotLease() {
        if (held_) {
            {
                std::lock_guard<std::mutex> lock(s_.mu);
                s_.busy = 0;
            }
            s_.slot_cv.notify_all();
        }
    }

private:
    ServerState& s_;
    bool held_ = false;
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
        {"busy", s.busy != 0},
        {"queued", s.queued},
        {"queue_depth", s.queue_depth},
        {"cache_reuse", s.cache_reuse},
        {"slot_save_path", s.slot_save_path},
        {"ctx_pinned", s.ctx.n_ctx_max()},
        {"unbounded", s.ctx.n_ctx_max() == 0},
        {"yarn_factor", s.ctx.yarn_factor()},
        {"kv_on_cpu", s.ctx.kv_on_cpu()},
    };
}

std::string now_id() {
    static int counter = 0;
    return "chatcmpl-pleiades-" + std::to_string(++counter);
}

// Builds the OpenAI-shape `tool_calls` array for a chat.completion response
// message. `streaming` adds the `index` field the SSE delta path requires (and
// the non-streaming message shape omits) -- otherwise identical. Each call's id
// is the one llama.cpp's parser assigned, or a stable "call_<i>" fallback when
// the model/template produced none.
json tool_calls_json(const std::vector<ChatToolCall>& calls, bool streaming) {
    json arr = json::array();
    for (size_t i = 0; i < calls.size(); ++i) {
        json entry = {
            {"id", calls[i].id.empty() ? "call_" + std::to_string(i) : calls[i].id},
            {"type", "function"},
            {"function", {{"name", calls[i].name}, {"arguments", calls[i].arguments}}},
        };
        if (streaming) {
            entry["index"] = i;
        }
        arr.push_back(std::move(entry));
    }
    return arr;
}

void refuse_busy(ServerState& s, httplib::Response& res) {
    res.status = 429;
    res.set_header("Retry-After", "1");
    res.set_content(
        json{{"error",
              {{"message", "model is busy generating; retry after the in-flight turn completes"},
               {"type", "server_error"},
               {"code", "model_busy"}}}}
            .dump(),
        "application/json");
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
    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
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

    json tools = json::array();
    if (body.contains("tools") && body["tools"].is_array()) {
        tools = body["tools"];
    }
    bool tools_offered = !tools.empty();

    const ChatTemplates& tmpls = s.models.chat_templates();
    bool model_supports_tools = tmpls.supports_tools();
    // A tool-offered request against a template with no tool support is honored
    // exactly as before: the tools are dropped (rendered as a plain completion,
    // no tool_calls will ever be parsed), with a loud warning -- see the old
    // ToolDialect::NONE behavior this preserves.
    json effective_tools = (tools_offered && model_supports_tools) ? tools : json::array();
    if (tools_offered && !model_supports_tools) {
        std::fprintf(stderr,
                      "[pleiades-engine-server] WARNING: request offered %zu tool(s) but this model's chat template "
                      "advertises no tool-calling support (ChatTemplates::supports_tools()==false) -- ignoring tools, "
                      "falling back to a plain completion (no tool_calls will ever be parsed from this model's "
                      "output).\n",
                      tools.size());
    }

    // enable_thinking toggles the template's own thinking block (models that
    // have none ignore it). Default on; a request may override with a boolean.
    bool enable_thinking = true;
    if (body.contains("enable_thinking") && body["enable_thinking"].is_boolean()) {
        enable_thinking = body["enable_thinking"].get<bool>();
    }

    RenderedChat rendered;
    try {
        rendered = tmpls.apply(body["messages"], effective_tools, /*add_generation_prompt=*/true, enable_thinking);
    } catch (const std::exception& e) {
        // A jinja evaluation error is a bad request (e.g. a message shape the
        // template rejects), not a server fault -- surface it as 400.
        res.status = 400;
        res.set_content(json{{"error",
                              {{"message", std::string("chat template render failed: ") + e.what()},
                               {"type", "invalid_request_error"}}}}
                            .dump(),
                         "application/json");
        return;
    }
    std::string prompt = rendered.prompt();
    // The resolved format may want extra stop strings honored (e.g. a tool-call
    // closing marker); merge them into whatever the request already asked for.
    for (const auto& extra_stop : rendered.additional_stops()) {
        sampling.stop.push_back(extra_stop);
    }
    // Streaming can only carry structured tool_calls (not raw markup) if we
    // buffer the whole turn and parse it before emitting SSE. That's only
    // needed when a tool call could actually appear -- i.e. tools were offered
    // AND the template supports them. A plain (or tool-unsupported) request
    // keeps true token-by-token streaming. This mirrors the pre-Stage-B
    // `!tools.empty() && dialect != NONE` discriminator exactly.
    bool tool_mode = !effective_tools.empty();

    long created = static_cast<long>(std::time(nullptr));
    std::string id = now_id();

    if (!stream) {
        SlotLease lease(s);
        if (!lease.acquire()) {
            refuse_busy(s, res);
            return;
        }
        std::lock_guard<std::mutex> lock(s.mu);
        GenerationResult r = s.engine.complete(prompt, n_predict, sampling);
        // Prove-it-fires observability (Phase 6): report how much of the
        // prompt was served from the KV prefix cache vs. re-decoded. On a
        // pure-attention model a repeated persona/system prefix shows a high
        // `cached` here; on a hybrid-recurrent model (qwen35moe) it stays 0
        // by design (partial reuse is architecturally impossible there) and
        // the engine safely cold-decodes -- see the design doc's Phase 6.
        s.has_state = true;
        std::fprintf(stderr,
                     "[pleiades-engine-server] chat: prompt_tokens=%d prefix_cached=%d chunks_reused=%d decoded=%d\n",
                     r.n_prompt_tokens, r.n_prompt_cached, r.n_chunks_reused,
                     r.n_prompt_tokens - r.n_prompt_cached);

        json message = {{"role", "assistant"}};
        std::string finish_reason = "stop";
        try {
            ParsedChatMessage pm = rendered.parse(r.text, /*is_partial=*/false);
            // Fail-loud discipline (Pleiades-specific, not llama.cpp's): a
            // tool-capable turn whose parse yields NOTHING usable -- no tool
            // call, no content, no reasoning -- while the model DID generate
            // text is a tool call truncated (or mangled) mid-structure. The PEG
            // parser accepts such a fragment leniently as "empty" rather than
            // throwing, so we detect it here and 422 it exactly like a hard
            // parse error: with exec_policy "allow", a real intended action must
            // never silently vanish into an empty 200. (The old Qwen-only parser
            // caught this as an unterminated <tool_call>; this is the
            // format-agnostic equivalent.)
            if (tool_mode && pm.tool_calls.empty() && pm.content.empty() && pm.reasoning_content.empty() &&
                !r.text.empty()) {
                throw std::runtime_error(
                    "tool-capable generation produced no parseable tool call or content (likely truncated "
                    "mid tool-call by max_tokens)");
            }
            if (!pm.tool_calls.empty()) {
                // OpenAI convention: content is null (not "") when a turn is
                // ONLY tool calls with no leading prose.
                message["content"] = pm.content.empty() ? json(nullptr) : json(pm.content);
                message["tool_calls"] = tool_calls_json(pm.tool_calls, /*streaming=*/false);
                finish_reason = "tool_calls";
            } else {
                message["content"] = pm.content;
            }
            if (!pm.reasoning_content.empty()) {
                message["reasoning_content"] = pm.reasoning_content;
            }
        } catch (const std::exception& e) {
            // Fail loud and distinctly from a normal 200/"stop" response: the
            // model's output could NOT be parsed into a complete, trustworthy
            // reply in the template's format (truncated/malformed tool call).
            // A 4xx here makes pleiades/harness/llm.py::_post's
            // urllib.request.urlopen() raise HTTPError, which agent.py's loop
            // treats as a retryable "model_error" -- NOT as "no tool_calls ->
            // final answer", exactly the distinction that must never be lost
            // (Pleiades often runs exec_policy "allow").
            res.status = 422;
            res.set_content(
                json{{"error",
                      {{"message", std::string("model produced output that does not match the model's chat format: ") +
                                       e.what()},
                       {"type", "tool_call_parse_error"}}}}
                    .dump(),
                "application/json");
            return;
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
    // synchronously below to PRE-BUILD the buffered-structured-stream frames
    // before any is written -- deliberately NOT reused inside the raw-stream
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

    // -- Phase A: buffered structured streaming ----------------------------- //
    //
    // The SSE token path streams delta.content only; it can't emit structured
    // delta.tool_calls. Streaming raw tool-call markup as visible content is the
    // specific silent failure this guards against: the streamed interactive path
    // (pleiades/engine.py::stream_events) reconstructs tool calls from
    // delta.tool_calls and never raises on plain content, so raw markup there
    // executes NO tool while looking like a normal answer.
    //
    // So when a tool call could appear, run the whole turn buffered (reusing
    // the exact non-streaming complete() + parse machinery), then replay the
    // parsed result as SSE: content first, then a SINGLE delta carrying the
    // whole tool_calls array (the OpenAI SDK the caller uses accumulates a
    // one-chunk tool_call fine via tc.index). A plain request falls through to
    // the unchanged raw token-streaming path below.
    if (tool_mode) {
        SlotLease lease(s);
        if (!lease.acquire()) {
            refuse_busy(s, res);
            return;
        }
        GenerationResult r;
        {
            std::lock_guard<std::mutex> lock(s.mu);
            r = s.engine.complete(prompt, n_predict, sampling);
        }
        s.has_state = true;
        std::fprintf(stderr,
                     "[pleiades-engine-server] chat(stream,tools): prompt_tokens=%d prefix_cached=%d chunks_reused=%d decoded=%d\n",
                     r.n_prompt_tokens, r.n_prompt_cached, r.n_chunks_reused,
                     r.n_prompt_tokens - r.n_prompt_cached);

        std::vector<std::string> chunks;
        chunks.push_back(sse_chunk({{"role", "assistant"}}, nullptr));
        const char* finish_reason = "stop";
        try {
            ParsedChatMessage pm = rendered.parse(r.text, /*is_partial=*/false);
            // Same fail-loud guard as the non-streaming path (see there): a
            // tool-capable turn that parses to nothing usable is a truncated
            // tool call, not an empty final answer -- 422, never a 200 SSE.
            if (tool_mode && pm.tool_calls.empty() && pm.content.empty() && pm.reasoning_content.empty() &&
                !r.text.empty()) {
                throw std::runtime_error(
                    "tool-capable generation produced no parseable tool call or content (likely truncated "
                    "mid tool-call by max_tokens)");
            }
            if (!pm.reasoning_content.empty()) {
                chunks.push_back(sse_chunk({{"reasoning_content", pm.reasoning_content}}, nullptr));
            }
            if (!pm.tool_calls.empty()) {
                if (!pm.content.empty()) {
                    chunks.push_back(sse_chunk({{"content", pm.content}}, nullptr));
                }
                chunks.push_back(sse_chunk({{"tool_calls", tool_calls_json(pm.tool_calls, /*streaming=*/true)}}, nullptr));
                finish_reason = "tool_calls";
            } else if (!pm.content.empty()) {
                chunks.push_back(sse_chunk({{"content", pm.content}}, nullptr));
            }
        } catch (const std::exception& e) {
            // Fail loud with the SAME 422 as the non-streaming path -- possible
            // here precisely because generation finished BEFORE any SSE byte was
            // sent, so nothing is committed yet. The caller's OpenAI client
            // raises on the 422 and engine.py's streamed branch falls back to a
            // non-streamed request. A 200 that narrates malformed markup as
            // content is the one outcome this must never produce.
            res.status = 422;
            res.set_content(
                json{{"error",
                      {{"message", std::string("model produced output that does not match the model's chat format: ") +
                                       e.what()},
                       {"type", "tool_call_parse_error"}}}}
                    .dump(),
                "application/json");
            return;
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

    // -- Raw token streaming (content-only format) -------------------------- //
    //
    // Unchanged in spirit: one provider call runs the whole generation, writing
    // an SSE chunk per token (or per held-back chunk when `stop` strings are set
    // -- see Engine::generate) via the on_token callback, then [DONE]. No
    // tool-call parsing is needed on this path (a content-only format never
    // produces structured output), and the PROMPT was still built through the
    // model's real template above, so any assistant tool_calls history still
    // round-trips for a streamed turn.
    auto lease = std::make_shared<SlotLease>(s);
    if (!lease->acquire()) {
        refuse_busy(s, res);
        return;
    }
    res.set_chunked_content_provider(
        "text/event-stream", [&s, lease, prompt, n_predict, id, created, sampling](size_t /*offset*/, httplib::DataSink& sink) {
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
            try {
                GenerationResult sr = s.engine.generate(
                    prompt, n_predict,
                    [&](const std::string& piece) { return write_chunk({{"content", piece}}, nullptr); }, sampling);
                s.has_state = true;
                std::fprintf(stderr,
                             "[pleiades-engine-server] chat(stream): prompt_tokens=%d prefix_cached=%d chunks_reused=%d decoded=%d\n",
                             sr.n_prompt_tokens, sr.n_prompt_cached, sr.n_chunks_reused,
                             sr.n_prompt_tokens - sr.n_prompt_cached);
            } catch (const std::exception& e) {
                // A mid-stream failure (e.g. llama_decode OOM) must NOT escape
                // this content-provider lambda: httplib runs it during response
                // serialization, OUTSIDE the routing try/catch that guards the
                // handler, so an uncaught throw here tears down the connection
                // handler instead of just this request. The 200 + SSE preamble
                // is already committed, so we can't downgrade to an error
                // status -- log it and end the stream cleanly (the client sees
                // a truncated completion terminated by [DONE], never a hang).
                std::fprintf(stderr, "[pleiades-engine-server] chat(stream): generation failed mid-stream: %s\n",
                             e.what());
            }
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
    // --no-mmap / --no-repack (ModelManager::load's use_mmap/use_extra_bufts)
    bool no_mmap = false;
    bool no_repack = false;
    // llama-server's --cache-reuse: minimum token-run length eligible for
    // middle-of-prompt KV chunk salvage (0 = off, the upstream default).
    int cache_reuse = 0;
    // --slot-save-path directory: enables POST /slots/0?action=save|restore
    // (models.py drives these best-effort around stop/start).
    std::string slot_save_path;
    int queue_depth = 8;
    int queue_timeout_sec = 30;
    // Phase 2 — elastic context:
    int ctx_max = 0;              // hard pin (0 = unbounded; budgets govern growth)
    int kv_budget_mib = 0;        // GPU K/V budget; 0 = auto (free device VRAM at boot)
    int kv_cpu_budget_mib = 8192; // host-RAM K/V budget after a spill; 0 = unlimited
    bool no_auto_yarn = false;    // forbid growth past n_ctx_train
};

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --model PATH [--host HOST] [--port PORT] [--ctx N] [--ngl N]\n"
        "       [--n-cpu-moe N] [--ub N] [--batch N] [--fa on|off|auto]\n"
        "       [--ctk TYPE] [--ctv TYPE] [--alias NAME] [--threads N] [--mlock]\n"
        "       [--no-mmap] [--no-repack] [--cache-reuse N] [--slot-save-path DIR]\n"
        "       [--queue-depth N] [--queue-timeout-sec N]\n"
        "       [--ctx-max N] [--kv-budget-mib N] [--kv-cpu-budget-mib N] [--no-auto-yarn]\n"
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
        } else if (arg == "--no-mmap") {
            a.no_mmap = true;
        } else if (arg == "--no-repack") {
            a.no_repack = true;
        } else if (arg == "--cache-reuse") {
            a.cache_reuse = std::stoi(next(i));
        } else if (arg == "--slot-save-path") {
            a.slot_save_path = next(i);
        } else if (arg == "--queue-depth") {
            a.queue_depth = std::stoi(next(i));
        } else if (arg == "--queue-timeout-sec") {
            a.queue_timeout_sec = std::stoi(next(i));
        } else if (arg == "--ctx-max") {
            a.ctx_max = std::stoi(next(i));
        } else if (arg == "--kv-budget-mib") {
            a.kv_budget_mib = std::stoi(next(i));
        } else if (arg == "--kv-cpu-budget-mib") {
            a.kv_cpu_budget_mib = std::stoi(next(i));
        } else if (arg == "--no-auto-yarn") {
            a.no_auto_yarn = true;
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
        cparams.auto_yarn = !args.no_auto_yarn;

        ModelManager models;
        models.load(args.model, args.n_gpu_layers, args.n_cpu_moe, args.use_mlock,
                    /*statewise_map=*/"", /*use_mmap=*/!args.no_mmap,
                    /*use_extra_bufts=*/!args.no_repack);
        // Chat templating is built once here (ModelManager::load(), from the
        // model's own GGUF jinja template) -- logged loudly, including which
        // template llama.cpp resolved and whether it advertises tool-calling
        // support (a template without it means every /v1/chat/completions
        // request that offers `tools` will have them ignored -- see
        // handle_chat()'s WARNING log for that case).
        std::fprintf(stderr, "[pleiades-engine] chat template: %s (tool-calling: %s)\n",
                     models.chat_templates().source().c_str(),
                     models.chat_templates().supports_tools() ? "supported" : "not advertised");
        // Phase 2 — auto GPU KV budget: free device VRAM as of right after the
        // model load (the model's own footprint is already subtracted). An
        // explicit --kv-budget-mib wins.
        size_t gpu_free = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                size_t f = 0, t = 0;
                ggml_backend_dev_memory(dev, &f, &t);
                gpu_free += f;
            }
        }
        const size_t kv_budget = args.kv_budget_mib > 0
                                     ? static_cast<size_t>(args.kv_budget_mib) * 1024 * 1024
                                     : gpu_free;
        const size_t cpu_budget = args.kv_cpu_budget_mib > 0
                                      ? static_cast<size_t>(args.kv_cpu_budget_mib) * 1024 * 1024
                                      : 0;
        ContextGovernor ctx;
        // n_ctx_max > 0 is a HARD pin (--ctx-max); 0 = unbounded, growth is
        // governed by the KV budgets. This replaces the old implicit
        // n_ctx_train ceiling: growth past the trained context engages YaRN
        // (opt out with --no-auto-yarn).
        ctx.create(models.model(), args.n_ctx, args.ctx_max, cparams);
        ctx.set_budgets(kv_budget, cpu_budget);
        Engine engine(models, ctx, args.cache_reuse);
        ServerState state{models, ctx, engine, args.alias, {}};
        state.slot_save_path = args.slot_save_path;
        state.cache_reuse = args.cache_reuse;
        state.queue_depth = args.queue_depth > 0 ? args.queue_depth : 1;
        state.queue_timeout_sec = args.queue_timeout_sec;
        if (!args.slot_save_path.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(args.slot_save_path, ec);
            if (ec) {
                std::fprintf(stderr, "[pleiades-engine-server] WARNING: cannot create --slot-save-path '%s': %s\n",
                             args.slot_save_path.c_str(), ec.message().c_str());
                state.slot_save_path.clear();
            }
        }

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
        // Slot-0 state persistence -- the same HTTP contract llama-server
        // serves with --slot-save-path and models.py already drives
        // best-effort around stop()/start(): POST /slots/0?action=save|restore
        // with {"filename": "<basename>"}. 200 on success; any failure is a
        // 4xx that models.py treats as "nothing saved / cold start", which is
        // exactly the pre-existing semantics.
        svr.Post(R"(/slots/\d+)", [&](const httplib::Request& req, httplib::Response& res) {
            if (state.slot_save_path.empty()) {
                res.status = 501;
                res.set_content(json{{"error", "server was not started with --slot-save-path"}}.dump(),
                                 "application/json");
                return;
            }
            auto action_it = req.params.find("action");
            std::string action = action_it != req.params.end() ? action_it->second : "";
            if (action != "save" && action != "restore") {
                res.status = 400;
                res.set_content(json{{"error", "action must be save or restore"}}.dump(), "application/json");
                return;
            }
            std::string filename;
            try {
                json body = json::parse(req.body);
                if (body.contains("filename") && body["filename"].is_string()) {
                    filename = body["filename"].get<std::string>();
                }
            } catch (const std::exception&) {
                // body optional for this endpoint; filename may also arrive via
                // query param in some callers.
                auto filename_it = req.params.find("filename");
                if (filename_it != req.params.end()) {
                    filename = filename_it->second;
                }
            }
            if (filename.empty() || filename.find('/') != std::string::npos ||
                filename.find('\\') != std::string::npos || filename.find("..") != std::string::npos) {
                res.status = 400;
                res.set_content(json{{"error", "filename must be a plain basename"}}.dump(), "application/json");
                return;
            }
            const std::string path = state.slot_save_path + "/" + filename;

            std::lock_guard<std::mutex> lock(state.mu);
            if (action == "save") {
                // PrefixCache's view of sequence 0 rides along in the file so a
                // restore can rebind the cache and keep reusing the KV.
                std::vector<llama_token> tokens = state.engine.prefix_cache().tokens();
                if (save_slot_state(state.ctx.ctx(), tokens, path)) {
                    std::fprintf(stderr, "[pleiades-engine-server] slot saved: %s (%zu tokens)\n", path.c_str(),
                                 tokens.size());
                    res.set_content(json{{"saved", true}, {"file", filename}}.dump(), "application/json");
                } else {
                    res.status = 409;
                    res.set_content(json{{"error", "nothing to save (no resident state) or write failed"}}.dump(),
                                     "application/json");
                }
            } else {
                if (restore_slot_state(state.ctx.ctx(), state.engine, path)) {
                    std::fprintf(stderr, "[pleiades-engine-server] slot restored: %s\n", path.c_str());
                    res.set_content(json{{"restored", true}, {"file", filename}}.dump(), "application/json");
                } else {
                    res.status = 404;
                    res.set_content(json{{"error", "no usable save file (missing, corrupt, or state mismatch)"}}.dump(),
                                     "application/json");
                }
            }
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
            "n_cpu_moe=%d, ub=%d, batch=%d, fa=%s, ctk=%s, ctv=%s, mlock=%s, mmap=%s, repack=%s, "
            "cache_reuse=%d, slot_save_path=%s)\n"
            "[pleiades-engine-server] elastic: ctx_max=%s, auto_yarn=%s, kv_budget=%.1f MiB (GPU), "
            "kv_cpu_budget=%s\n",
            args.host.c_str(), args.port, state.ctx.n_ctx(), state.ctx.n_ctx_max(), args.alias.c_str(),
            args.n_gpu_layers, args.n_cpu_moe, args.n_ubatch, args.n_batch, args.flash_attn.c_str(),
            args.cache_type_k.c_str(), args.cache_type_v.c_str(), args.use_mlock ? "on" : "off",
            args.no_mmap ? "off" : "on", args.no_repack ? "off" : "on", args.cache_reuse,
            state.slot_save_path.empty() ? "none" : state.slot_save_path.c_str(),
            args.ctx_max > 0 ? std::to_string(args.ctx_max).c_str() : "unbounded",
            args.no_auto_yarn ? "off" : "on", kv_budget / (1024.0 * 1024.0),
            cpu_budget ? std::to_string(args.kv_cpu_budget_mib).c_str() : "unlimited");
        svr.listen(args.host.c_str(), args.port);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        llama_backend_free();
        return 1;
    }

    llama_backend_free();
    return 0;
}
