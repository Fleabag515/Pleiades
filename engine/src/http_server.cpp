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
// Not yet ported from server.py (out of Phase 3's explicit scope): /tokenize,
// /extras/tokenize/count, /metrics, prompt-overflow auto-upshift. Since
// ported (docs/specs/2026-07-21-native-inference-engine-design.md, Phase 9):
// GET /v1/models, kv_bytes_per_token in /props, POST /slots/0?action=
// save|restore, --caps, --mmproj/vision (POST /v1/chat/completions accepts
// OpenAI content-parts image_url as data: URIs).
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
#include <atomic>
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

#include "base64.hpp"     // public domain, header-only -- see CMakeLists.txt's include-dir comment
#include "httplib.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
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

// Port of pleiades/hardware.py::kv_bytes_per_token() -- the one field Phase 3
// left out of /props vs. the Python server (server.py:189). Pure-recurrent
// architectures genuinely have zero per-token KV growth (llama_model_is_recurrent),
// matching hardware.py's memory_arch=="recurrent" branch exactly. For hybrid
// models, hardware.py discounts recurrent layers when its GGUF-metadata-parsed
// recurrent_layer_count is known, "falling back to the original conservative
// all-layers estimate otherwise" (its own docstring) -- that fallback is what
// this ports: llama_model_n_head_kv()/n_head()/n_layer() are model-wide public
// API calls (no per-layer hybrid breakdown exposed there), so this always takes
// hardware.py's conservative branch. Never UNDER-estimates; may over-estimate
// KV cost for a hybrid model relative to Python's more precise metadata-aware
// path -- same direction of error hardware.py itself accepts as the fallback.
int64_t kv_bytes_per_token(const llama_model* model, int bytes_per_elem = 2) {
    if (llama_model_is_recurrent(model)) {
        return 0;
    }
    int32_t n_layer = llama_model_n_layer(model);
    int32_t n_embd = llama_model_n_embd(model);
    int32_t n_head = llama_model_n_head(model);
    int32_t n_head_kv = llama_model_n_head_kv(model);
    if (n_layer <= 0 || n_embd <= 0 || n_head <= 0) {
        return static_cast<int64_t>(n_layer > 0 ? n_layer : 32) * 1024;  // order-of-magnitude fallback, matches hardware.py
    }
    int64_t head_dim = n_embd / n_head;
    return static_cast<int64_t>(n_layer) * n_head_kv * head_dim * 2 * bytes_per_elem;
}

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
        {"kv_bytes_per_token", kv_bytes_per_token(s.models.model())},
        {"modalities", {{"vision", s.models.mtmd() && mtmd_support_vision(s.models.mtmd())}}},
        {"default_generation_settings", {{"n_ctx", s.ctx.n_ctx()}}},
    };
}

std::string now_id() {
    // httplib dispatches concurrent connections across a worker-thread pool
    // (see file header), and this is called for every request BEFORE s.mu is
    // taken -- so the counter itself must be safe under concurrent increment,
    // independent of the generation-serializing lock.
    static std::atomic<uint64_t> counter{0};
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

// Phase 9.4.2 (docs/specs/2026-07-21-native-inference-engine-design.md):
// rewrites `messages` IN PLACE, extracting any OpenAI content-parts
// image_url entries into bitmaps and replacing each with a plain "text" part
// containing the model's media marker -- so the chat template (which never
// learns about images at all) renders a normal string prompt with the
// marker(s) sitting exactly where each image was, ready for mtmd_tokenize().
// This mirrors upstream llama-server's own approach (server-common.cpp's
// image_url -> media_marker substitution before template rendering), done
// independently here since that file is server/-scoped and never compiled
// into this engine (see CMakeLists.txt's LLAMA_BUILD_COMMON=OFF rationale).
//
// Only data: URIs are supported; remote http(s):// fetching is deliberately
// refused -- this is a locally-bound server, and fetching an
// attacker-controlled URL server-side is a real SSRF surface neither Python
// fallback this engine replaces implements either (see the design doc's own
// recommendation on this point).
//
// Throws on: an image part with no --mmproj loaded, a non-data: URL, invalid
// base64, or a bitmap the vision encoder can't decode (unsupported format /
// corrupt data / video input, which is out of scope here).
//
// Returns a plain std::vector<mtmd::bitmap>, NOT mtmd.h's own `mtmd::bitmaps`
// wrapper struct -- that struct declares `~bitmaps() = default`, which (a
// real C++ rule-of-five gotcha, not a bug in our code) silently SUPPRESSES
// the implicitly-generated move assignment operator, leaving only a copy
// assignment that's itself deleted (bitmap owns a unique_ptr). Assigning a
// `bitmaps` value -- exactly what `image_bitmaps = extract_and_replace_images(...)`
// at the call site needs to do -- fails to compile as a result. A bare
// std::vector<bitmap> has no such problem: vector's own move-assignment
// swaps ownership of its internal buffer directly (for the default
// std::allocator, always) and never needs bitmap's own assignment operator
// at all, only its move CONSTRUCTOR (used solely for reallocation during
// emplace_back, which mtmd::bitmap does declare, noexcept).
std::vector<mtmd::bitmap> extract_and_replace_images(json& messages, mtmd_context* mtmd_ctx) {
    std::vector<mtmd::bitmap> bitmaps;
    for (auto& message : messages) {
        if (!message.is_object() || !message.contains("content") || !message["content"].is_array()) {
            continue;
        }
        for (auto& part : message["content"]) {
            if (!part.is_object() || part.value("type", "") != "image_url") {
                continue;
            }
            if (!mtmd_ctx) {
                throw std::runtime_error("this model has no --mmproj loaded -- cannot accept image input");
            }
            std::string url = part.value("image_url", json::object()).value("url", "");
            size_t comma = url.find(',');
            if (url.rfind("data:", 0) != 0 || comma == std::string::npos) {
                throw std::runtime_error(
                    "image_url must be a data: URI (data:<mime>;base64,<data>) -- fetching remote URLs is disabled");
            }
            std::string raw;
            try {
                raw = base64::decode(url.substr(comma + 1));
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("invalid base64 image data: ") + e.what());
            }
            mtmd_helper_bitmap_wrapper wrapped = mtmd_helper_bitmap_init_from_buf(
                mtmd_ctx, reinterpret_cast<const unsigned char*>(raw.data()), raw.size(), /*placeholder=*/false);
            if (wrapped.video_ctx) {
                mtmd_helper_video_free(wrapped.video_ctx);
                if (wrapped.bitmap) {
                    mtmd_bitmap_free(wrapped.bitmap);
                }
                throw std::runtime_error("video input is not supported here, only static images");
            }
            if (!wrapped.bitmap) {
                throw std::runtime_error("failed to decode image (unsupported format or corrupt data)");
            }
            bitmaps.emplace_back(wrapped.bitmap);
            part = json{{"type", "text"}, {"text", mtmd_get_marker(mtmd_ctx)}};
        }
    }
    return bitmaps;
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

    // Phase 9.4.2: pull any image_url content-parts out into bitmaps and
    // replace them with plain text markers BEFORE the chat template ever
    // sees `messages` -- see extract_and_replace_images's own comment. Must
    // run before tmpls.apply() below, since that's what renders `prompt`.
    std::vector<mtmd::bitmap> image_bitmaps;
    try {
        image_bitmaps = extract_and_replace_images(body["messages"], s.models.mtmd());
    } catch (const std::exception& e) {
        res.status = 400;
        res.set_content(json{{"error", {{"message", e.what()}, {"type", "invalid_request_error"}}}}.dump(),
                        "application/json");
        return;
    }
    const bool is_multimodal = !image_bitmaps.empty();

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

    // Phase 9.4.2: `prompt` now contains the media marker(s) exactly where
    // each image was (the template rendered our text-part substitution like
    // any other text). Turn it + the extracted bitmaps into mixed chunks.
    // Resolved once here, synchronously, so any failure is a normal 400 --
    // none of the three call sites below (including the async raw-streaming
    // one) need their own tokenize-failure handling.
    // shared_ptr, not mtmd.h's own unique_ptr alias: the raw-token-streaming
    // call site further down captures this into a lambda handed to
    // httplib::Server::set_chunked_content_provider(), whose signature is a
    // std::function -- and std::function requires its target to be
    // COPY-constructible (a real standard-library constraint, not a cpp-
    // httplib choice), which a unique_ptr-capturing lambda is not, even when
    // never actually copied in practice. shared_ptr's copy is cheap (just a
    // refcount bump) and gives every call site the same *.get() access
    // pattern a unique_ptr would, so this is the standard workaround rather
    // than a compromise.
    std::shared_ptr<mtmd_input_chunks> mm_chunks;
    if (is_multimodal) {
        mtmd_input_chunks* raw_chunks = mtmd_input_chunks_init();
        mtmd_input_text input_text{prompt.c_str(), prompt.size(), /*add_special=*/true, /*parse_special=*/true};
        std::vector<const mtmd_bitmap*> bptrs;
        bptrs.reserve(image_bitmaps.size());
        for (const mtmd::bitmap& b : image_bitmaps) {
            bptrs.push_back(b.ptr.get());
        }
        int32_t tok_rc = mtmd_tokenize(s.models.mtmd(), raw_chunks, &input_text, bptrs.data(), bptrs.size());
        if (tok_rc != 0) {
            mtmd_input_chunks_free(raw_chunks);
            res.status = 400;
            const char* why = tok_rc == 1
                ? "number of images does not match the number of media markers the chat template rendered"
                : "image preprocessing error";
            res.set_content(json{{"error", {{"message", why}, {"type", "invalid_request_error"}}}}.dump(),
                            "application/json");
            return;
        }
        mm_chunks = std::shared_ptr<mtmd_input_chunks>(raw_chunks, mtmd_input_chunks_free);
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
        std::lock_guard<std::mutex> lock(s.mu);
        GenerationResult r = is_multimodal ? s.engine.complete_multimodal(mm_chunks.get(), n_predict, sampling)
                                            : s.engine.complete(prompt, n_predict, sampling);
        // Prove-it-fires observability (Phase 6): report how much of the
        // prompt was served from the KV prefix cache vs. re-decoded. On a
        // pure-attention model a repeated persona/system prefix shows a high
        // `cached` here; on a hybrid-recurrent model (qwen35moe) it stays 0
        // by design (partial reuse is architecturally impossible there) and
        // the engine safely cold-decodes -- see the design doc's Phase 6.
        std::fprintf(stderr, "[pleiades-engine-server] chat: prompt_tokens=%d prefix_cached=%d decoded=%d\n",
                     r.n_prompt_tokens, r.n_prompt_cached, r.n_prompt_tokens - r.n_prompt_cached);

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
        GenerationResult r;
        {
            std::lock_guard<std::mutex> lock(s.mu);
            r = is_multimodal ? s.engine.complete_multimodal(mm_chunks.get(), n_predict, sampling)
                              : s.engine.complete(prompt, n_predict, sampling);
        }
        std::fprintf(stderr,
                     "[pleiades-engine-server] chat(stream,tools): prompt_tokens=%d prefix_cached=%d decoded=%d\n",
                     r.n_prompt_tokens, r.n_prompt_cached, r.n_prompt_tokens - r.n_prompt_cached);

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
    res.set_chunked_content_provider(
        "text/event-stream",
        // mm_chunks captured by (cheap, refcounted) copy -- this lambda runs
        // AFTER handle_chat() returns, per the class of bug this file's
        // header already warns about for exactly this content-provider
        // pattern (a captured reference to a local would dangle); a plain
        // capture-by-value of a shared_ptr is exactly what keeps the
        // pointee alive that long. is_multimodal is a trivial bool copy.
        [&s, prompt, n_predict, id, created, sampling, is_multimodal,
         mm_chunks](size_t /*offset*/, httplib::DataSink& sink) {
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
                auto on_piece = [&](const std::string& piece) { return write_chunk({{"content", piece}}, nullptr); };
                GenerationResult sr = is_multimodal
                    ? s.engine.generate_multimodal(mm_chunks.get(), n_predict, on_piece, sampling)
                    : s.engine.generate(prompt, n_predict, on_piece, sampling);
                std::fprintf(stderr,
                             "[pleiades-engine-server] chat(stream): prompt_tokens=%d prefix_cached=%d decoded=%d\n",
                             sr.n_prompt_tokens, sr.n_prompt_cached, sr.n_prompt_tokens - sr.n_prompt_cached);
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
    std::string slot_save_path;  // empty = /slots disabled, matching llama-server's own gate
    std::string mmproj;          // empty = vision disabled, matching llama-server's own --mmproj gate
};

// Phase 9.4 (docs/specs/2026-07-21-native-inference-engine-design.md): a
// deliberately simpler stand-in for llama.cpp's own fs_validate_filename()
// (common/common.cpp) -- that function does full UTF-8 codepoint validation
// because arbitrary user filenames are a real input there. Every caller on
// our side (pleiades/models.py's _slot_save/_slot_restore) only ever sends
// the single hardcoded constant "latest.bin", so a strict ASCII allowlist is
// both sufficient for the real threat model and avoids pulling in more of
// common/'s UTF-8 machinery than chat templating already needs. Still a real
// path-traversal guard, not a formality: rejects separators, "..", and
// leading/trailing dots regardless of what a caller actually sends.
bool is_safe_slot_filename(const std::string& name) {
    if (name.empty() || name.size() > 255) {
        return false;
    }
    if (name.find("..") != std::string::npos) {
        return false;
    }
    if (name.front() == '.' || name.back() == '.') {
        return false;
    }
    for (unsigned char c : name) {
        bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
        if (!ok) {
            return false;
        }
    }
    return true;
}

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --model PATH [--host HOST] [--port PORT] [--ctx N] [--ngl N]\n"
        "       [--n-cpu-moe N] [--ub N] [--batch N] [--fa on|off|auto]\n"
        "       [--ctk TYPE] [--ctv TYPE] [--alias NAME] [--threads N] [--mlock]\n"
        "       [--slot-save-path DIR] [--mmproj PATH] [--caps]\n"
        "\n"
        "legacy positional mode (still accepted): %s <model.gguf> <host> <port>"
        " [n_ctx] [n_gpu_layers] [alias]\n",
        prog, prog);
}

// Phase 9.2 (docs/specs/2026-07-21-native-inference-engine-design.md): a
// machine-readable, per-BUILD capability report, printed and exited before
// any model is touched -- this replaces pleiades/runtime.py::caps()'s old
// approach of grepping --help text (which can only ever say what flags this
// binary parses, not what the resolved backend can actually do). Consumers:
// autofit.place() should degrade MoE placement when moe_offload is false
// rather than silently mis-sizing VRAM for a split that can't happen (see
// Phase 9's Risk R1 -- this flag is what makes that risk *observable*, not
// what proves the offload is correct on non-CUDA backends).
void print_caps() {
    json caps = {
        {"backend", PLEIADES_ENGINE_BACKEND},
        // model_manager.cpp's tensor_buft_overrides mechanism (--n-cpu-moe)
        // is implemented generically against ggml_backend_cpu_buffer_type(),
        // not gated to any one GPU backend in this engine's own code -- so
        // the MECHANISM is present whenever there's a GPU backend to split
        // against (nothing to offload between CPU and itself on a CPU-only
        // build). Whether it's numerically CORRECT/fast on non-CUDA backends
        // is unverified -- see Phase 9's Risk R1 -- this flag reports
        // presence, not a correctness guarantee.
        {"moe_offload", std::string(PLEIADES_ENGINE_BACKEND) != "cpu"},
        // Build-time capability (this binary links mtmd and understands
        // --mmproj), NOT runtime state -- --caps prints before any --model/
        // --mmproj is ever parsed, so it cannot know whether THIS invocation
        // will actually load one. A caller planning whether to pass --mmproj
        // at all wants exactly this answer; whether a specific mmproj file
        // loaded successfully is only knowable after boot (see the startup
        // log, and mtmd_support_vision() once a model is live).
        {"vision", true},
        {"slots", true},     // POST /slots/(\d+)?action=save|restore -- live since Phase 9.4.1
        {"resize", true},    // POST /resize -- live since Phase 3
        {"engine_version", "0.1.0"},
    };
    std::printf("%s\n", caps.dump().c_str());
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
    if (first == "--caps") {
        print_caps();
        std::exit(0);
    }
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
        } else if (arg == "--slot-save-path") {
            a.slot_save_path = next(i);
        } else if (arg == "--mmproj") {
            a.mmproj = next(i);
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
        models.load(args.model, args.n_gpu_layers, args.n_cpu_moe, args.use_mlock,
                    /*statewise_map=*/"", args.mmproj);
        if (!args.mmproj.empty()) {
            std::fprintf(stderr, "[pleiades-engine] mmproj loaded: %s (vision: %s)\n", args.mmproj.c_str(),
                         mtmd_support_vision(models.mtmd()) ? "supported" : "NOT supported by this mmproj");
        }
        // Chat templating is built once here (ModelManager::load(), from the
        // model's own GGUF jinja template) -- logged loudly, including which
        // template llama.cpp resolved and whether it advertises tool-calling
        // support (a template without it means every /v1/chat/completions
        // request that offers `tools` will have them ignored -- see
        // handle_chat()'s WARNING log for that case).
        std::fprintf(stderr, "[pleiades-engine] chat template: %s (tool-calling: %s)\n",
                     models.chat_templates().source().c_str(),
                     models.chat_templates().supports_tools() ? "supported" : "not advertised");
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
        svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
            // pleiades/models.py's ModelManager.state()/_is_our_server() poll this
            // (not /health) as the "server is actually up" signal, and match a
            // running registry entry by `id` == the alias this process was
            // launched with -- never the GGUF path. Without this route those two
            // functions 404 forever: state() never leaves "loading", and
            // `pleiades model start`/`pleiades serve` time out on an otherwise
            // healthy server. See docs/specs/2026-07-21-native-inference-engine-
            // design.md's cutover-phase note on this gap.
            res.set_content(json{{"object", "list"},
                                 {"data", json::array({{{"id", state.alias},
                                                        {"object", "model"},
                                                        {"created", 0},
                                                        {"owned_by", "pleiades"}}})}}
                                 .dump(),
                             "application/json");
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
            try {
                int actual = state.ctx.resize(requested);
                double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                res.set_content(json{{"n_ctx", actual}, {"took_ms", static_cast<int>(ms)}}.dump(), "application/json");
            } catch (const std::exception& e) {
                // Same shape as server.py's /resize handler ("surface load
                // failures honestly"): a failed grow -- routine on low-VRAM
                // hardware -- must be a clean JSON 500, not cpp-httplib's
                // bare no-body 500 from its routing try/catch. The governor
                // rolled back to the previous n_ctx (see
                // ContextGovernor::resize), so the server stays up; n_ctx
                // reports where the engine actually is (0 only if even the
                // rollback failed).
                res.status = 500;
                res.set_content(json{{"error", std::string("resize failed: ") + e.what()},
                                     {"n_ctx", state.ctx.n_ctx()}}.dump(),
                                "application/json");
            }
        });
        // Phase 9.4 (docs/specs/2026-07-21-native-inference-engine-design.md):
        // wire contract matches upstream llama-server's /slots/:id_slot
        // exactly (pleiades/models.py's _slot_save/_slot_restore already
        // speak it -- POST .../slots/0?action=save|restore, JSON
        // {"filename": "..."}, 200 on success) so ModelManager needed zero
        // changes. Every character restart currently gets a warm-state load
        // this way; without it every restart is a full cold persona/memory
        // reprocess -- this session's design doc calls that the single most
        // visible regression a native-engine cutover could introduce.
        // Single sequence (id 0) only -- this engine has no multi-slot
        // concept (--parallel is effectively always 1, see http_server.cpp's
        // one global request mutex).
        svr.Post(R"(/slots/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
            if (args.slot_save_path.empty()) {
                res.status = 501;
                res.set_content(json{{"error", "This server does not support slots action. "
                                                "Start it with --slot-save-path"}}.dump(),
                                "application/json");
                return;
            }
            if (req.matches[1] != "0") {
                res.status = 400;
                res.set_content(json{{"error", "invalid slot id -- only 0 exists"}}.dump(), "application/json");
                return;
            }
            std::string action = req.has_param("action") ? req.get_param_value("action") : "";
            if (action != "save" && action != "restore") {
                res.status = 400;
                res.set_content(json{{"error", "action must be save or restore"}}.dump(), "application/json");
                return;
            }
            json body;
            try {
                body = json::parse(req.body);
            } catch (const std::exception&) {
                res.status = 400;
                res.set_content(json{{"error", "filename (string) required"}}.dump(), "application/json");
                return;
            }
            if (!body.contains("filename") || !body["filename"].is_string()) {
                res.status = 400;
                res.set_content(json{{"error", "filename (string) required"}}.dump(), "application/json");
                return;
            }
            std::string filename = body["filename"].get<std::string>();
            if (!is_safe_slot_filename(filename)) {
                res.status = 400;
                res.set_content(json{{"error", "invalid filename"}}.dump(), "application/json");
                return;
            }
            std::string filepath = args.slot_save_path;
            if (!filepath.empty() && filepath.back() != '/' && filepath.back() != '\\') {
                filepath += '/';  // matches llama-server's own params.slot_save_path convention (trailing sep expected)
            }
            filepath += filename;

            std::lock_guard<std::mutex> lock(state.mu);
            llama_context* ctx = state.ctx.ctx();
            if (!ctx) {
                res.status = 500;
                res.set_content(json{{"error", "no live context"}}.dump(), "application/json");
                return;
            }
            auto t0 = std::chrono::steady_clock::now();
            if (action == "save") {
                // Whatever ResidentMap currently tracks as resident for
                // sequence 0 IS the true token sequence in the KV -- Engine's
                // generate() keeps the two in lockstep on every call (Phase 6),
                // so there's no separate "ask the KV what it holds" step
                // needed the way upstream's per-slot `prompt.tokens` does.
                const std::vector<llama_token>& tokens = state.engine.resident_map().tokens();
                size_t nwrite = llama_state_seq_save_file(ctx, filepath.c_str(), /*seq_id=*/0,
                                                          tokens.data(), tokens.size());
                double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                res.set_content(json{{"filename", filename}, {"n_tokens", tokens.size()},
                                     {"n_bytes", nwrite}, {"took_ms", static_cast<int>(ms)}}.dump(),
                                "application/json");
            } else {
                std::vector<llama_token> buf(static_cast<size_t>(std::max(state.ctx.n_ctx(), 1)));
                size_t token_count = 0;
                size_t nread = llama_state_seq_load_file(ctx, filepath.c_str(), /*seq_id=*/0,
                                                         buf.data(), buf.size(), &token_count);
                if (nread == 0) {
                    // Missing file, corrupt blob, or a save that doesn't fit
                    // this context's n_ctx -- all "nothing usable," never a
                    // partial/inconsistent restore. Match models.py's own
                    // contract (best-effort, never worse than a cold start):
                    // clear whatever the failed attempt may have touched so
                    // the next request cold-decodes cleanly rather than
                    // inheriting a half-restored KV.
                    llama_memory_clear(llama_get_memory(ctx), /*data=*/true);
                    state.engine.reset_resident_map();
                    res.status = 400;
                    res.set_content(json{{"error", "unable to restore slot -- missing/invalid save file "
                                                    "or incompatible with this context"}}.dump(),
                                    "application/json");
                    return;
                }
                buf.resize(token_count);
                // The KV now physically holds these tokens at their saved
                // positions (llama_state_seq_load_file restores that
                // verbatim) -- seed_resident_map brings ResidentMap's
                // bookkeeping back into agreement with it. See that method's
                // own comment for what silently breaks if this is skipped.
                state.engine.seed_resident_map(buf);
                double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                res.set_content(json{{"filename", filename}, {"n_tokens", token_count},
                                     {"n_bytes", nread}, {"took_ms", static_cast<int>(ms)}}.dump(),
                                "application/json");
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
