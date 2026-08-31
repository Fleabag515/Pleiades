#include "pleiades_engine/engine.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace pleiades_engine {

namespace {

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text, bool add_special) {
    std::vector<llama_token> tokens(text.size() + 8);
    int n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(),
                            static_cast<int32_t>(tokens.size()), add_special, /*parse_special=*/true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(),
                            static_cast<int32_t>(tokens.size()), add_special, /*parse_special=*/true);
        if (n < 0) {
            throw std::runtime_error("pleiades_engine: tokenization failed");
        }
    }
    tokens.resize(n);
    return tokens;
}

std::string token_to_piece(const llama_vocab* vocab, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), /*lstrip=*/0, /*special=*/false);
    if (n < 0) {
        // Piece didn't fit -- extremely unlikely for a single token, but
        // handle it rather than silently truncating.
        std::vector<char> big(-n);
        n = llama_token_to_piece(vocab, token, big.data(), static_cast<int32_t>(big.size()), 0, false);
        return std::string(big.data(), n > 0 ? n : 0);
    }
    return std::string(buf, n);
}

// Builds the sampler chain for one generation from a SamplingParams. Ordering
// mirrors llama.cpp's own canonical default chain (common/sampling.cpp):
// penalties first, then the truncation samplers (top_k -> typical -> top_p ->
// min_p), then temperature, then the seeded final distribution sampler. Each
// stage is only added when it would actually do something, so a
// default-constructed SamplingParams (temperature 0) collapses to exactly the
// single greedy sampler engine.cpp built before this function existed --
// keeping the deterministic bench/test paths byte-identical.
//
// temperature <= 0 is llama.cpp's own "greedy" convention (llama_sampler_init_temp's
// own doc: t <= 0 keeps the max logit and -inf's the rest); we short-circuit
// to a pure argmax greedy sampler so temp==0 requests are deterministic
// regardless of the other knobs, matching llama-cpp-python.
llama_sampler* build_sampler_chain(const SamplingParams& sp) {
    llama_sampler_chain_params cparams = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(cparams);

    if (sp.temperature <= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        return chain;
    }

    if (sp.repeat_penalty != 1.0f || sp.frequency_penalty != 0.0f || sp.presence_penalty != 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(sp.penalty_last_n, sp.repeat_penalty,
                                                                     sp.frequency_penalty, sp.presence_penalty));
    }
    if (sp.top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(sp.top_k));
    }
    if (sp.typical_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_typical(sp.typical_p, /*min_keep=*/1));
    }
    if (sp.top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(sp.top_p, /*min_keep=*/1));
    }
    if (sp.min_p > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(sp.min_p, /*min_keep=*/1));
    }
    llama_sampler_chain_add(chain, llama_sampler_init_temp(sp.temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(sp.seed));
    return chain;
}

// Earliest byte offset at which any complete stop string occurs in `text`, or
// std::string::npos if none does.
size_t earliest_stop_pos(const std::string& text, const std::vector<std::string>& stops) {
    size_t best = std::string::npos;
    for (const std::string& s : stops) {
        if (s.empty()) {
            continue;
        }
        size_t p = text.find(s);
        if (p != std::string::npos && p < best) {
            best = p;
        }
    }
    return best;
}

// Length of the longest suffix of `text` that is a PROPER prefix of some stop
// string -- i.e. the tail that must be held back from a stream because the
// next token could still complete a stop match. 0 when no such overlap exists
// (the whole text is safe to emit). Never returns a length that would match a
// COMPLETE stop (that case is handled by earliest_stop_pos before this is
// consulted), so at most len(stop)-1 bytes are ever withheld.
size_t stop_overlap_suffix(const std::string& text, const std::vector<std::string>& stops) {
    size_t held = 0;
    for (const std::string& s : stops) {
        size_t max_l = std::min(s.size() > 0 ? s.size() - 1 : 0, text.size());
        for (size_t l = max_l; l > held; --l) {
            if (text.compare(text.size() - l, l, s, 0, l) == 0) {
                held = l;
                break;
            }
        }
    }
    return held;
}

}  // namespace

Engine::Engine(ModelManager& models, ContextGovernor& ctx, int cache_reuse)
    : models_(models), ctx_(ctx), cache_reuse_(cache_reuse) {}

void Engine::ensure_capacity(size_t prompt_len, int n_predict) {
    const size_t headroom = 1 + static_cast<size_t>(n_predict > 0 ? std::min(n_predict, 512) : 0);
    const size_t need = prompt_len + headroom;
    if (static_cast<size_t>(ctx_.n_ctx()) >= need) {
        return;
    }
    const int target = ctx_.next_gear_for(need);  // throws past a hard --ctx-max pin
    (void)grow_context_to(target);
}

ContextGovernor::GrowResult Engine::grow_context_to(int target) {
    const int old_ctx = ctx_.n_ctx();
    ContextGovernor::GrowResult gr = ctx_.grow_preserving(target);
    if (gr.preserved) {
        // Same KV, new context: re-bind the token mirror to the new epoch so
        // the reuse logic keeps working without a re-prefill.
        std::vector<llama_token> toks = prefix_.tokens();
        prefix_.set(std::move(toks), ctx_.epoch());
    } else {
        // RoPE regime changed (YaRN bucket) or the snapshot failed: the token
        // list stands but the KV is cold — the next decode re-prefills.
        prefix_.invalidate(ctx_.epoch());
    }
    std::fprintf(stderr,
                 "[pleiades-engine] context grew %d -> %d tokens (preserved=%d spilled=%d yarn=%.0fx kv_on_cpu=%d)\n",
                 old_ctx, gr.n_ctx, gr.preserved ? 1 : 0, gr.spilled ? 1 : 0,
                 ctx_.yarn_factor(), ctx_.kv_on_cpu() ? 1 : 0);
    return gr;
}

void Engine::reset_prefix_cache() {
    llama_context* ctx = ctx_.ctx();
    if (ctx) {
        // Hard-clear the KV so the tracked cache and the real KV agree (both
        // empty). data=true zeroes the backend buffers, not just the cell
        // metadata -- required so a subsequent flash-attention decode can't
        // read leftover K/V from these now-freed cells (see the stale-KV guard
        // in generate() for the full rationale).
        llama_memory_clear(llama_get_memory(ctx), /*data=*/true);
    }
    prefix_.invalidate(ctx_.epoch());
}

GenerationResult Engine::complete(const std::string& prompt, int n_predict, const SamplingParams& sampling) {
    return generate(prompt, n_predict, nullptr, sampling);
}

GenerationResult Engine::generate(const std::string& prompt, int n_predict,
                                   const std::function<bool(const std::string&)>& on_token,
                                   const SamplingParams& sampling) {
    if (!models_.is_loaded()) {
        throw std::runtime_error("pleiades_engine: no model loaded");
    }
    const llama_vocab* vocab = llama_model_get_vocab(models_.model());
    llama_context* ctx = ctx_.ctx();
    llama_memory_t mem = llama_get_memory(ctx);

    GenerationResult result;

    // -- prompt processing (Phase 6 prefix cache) --------------------------
    //
    // The pre-Phase-6 path decoded the ENTIRE prompt every call with no KV
    // bookkeeping, which -- because llama_batch_get_one() leaves batch.pos
    // == nullptr and llama.cpp then appends at memory->seq_pos_max(seq)+1 --
    // duplicated any repeated prefix into the KV at growing positions. Here
    // we instead: (1) drop the cache if the context was recreated under us
    // (resize bumps the governor epoch and empties the KV), (2) find the
    // longest prompt prefix already resident, (3) trim the KV back to just
    // that prefix, and (4) decode only the new suffix.
    auto t0 = std::chrono::steady_clock::now();

    if (prefix_.epoch() != ctx_.epoch()) {
        // Context was (re)created since we last decoded -- KV is empty.
        prefix_.invalidate(ctx_.epoch());
    }

    // High-water mark of the sequence currently resident in the KV -- the
    // number of contiguous positions [0, resident) that hold live K/V (the
    // previous request's prompt + everything it generated). prefix_ mirrors
    // the KV exactly (see PrefixCache), so this is just its length. Captured
    // before any trim below because the FA-safety check needs the *pre-trim*
    // resident length.
    const size_t resident = prefix_.size();

    std::vector<llama_token> prompt_tokens = tokenize(vocab, prompt, /*add_special=*/true);
    result.n_prompt_tokens = static_cast<int>(prompt_tokens.size());

    // Phase 2 — elastic context: grow (KV-preservingly) BEFORE the reuse
    // logic so the prompt + generation headroom always fits. This may
    // recreate the llama_context under us — refresh the handles below.
    ensure_capacity(prompt_tokens.size(), n_predict);
    ctx = ctx_.ctx();
    mem = llama_get_memory(ctx);

    size_t n_reuse = prefix_.reusable_prefix(prompt_tokens);
    result.n_chunks_reused = 0;

    // Fill level: how much of the prompt is already resident (the LCP prefix,
    // possibly extended below by chunk reuse and interleaved decoding).
    size_t n_past = n_reuse;
    // Position-indexed mirror of the KV contents (kv[i] = token at position
    // i). Starts as the pre-request resident sequence and is mutated in
    // lockstep with the real cache by every move/decode/trim below, so match
    // searches always see exactly what the cache holds.
    std::vector<llama_token> kv(prefix_.tokens());

    const bool fa_maybe_on = ctx_.params().flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    // Flash-attention stale-KV correctness guard (rationale preserved below):
    // with FA on, a prompt SHORTER than the resident sequence would leave
    // stale K/V in freed cells beyond the new prompt's end, and this pinned
    // llama.cpp's FA kernel leaks those masked cells into attention
    // (empirically proven on CPU *and* CUDA). The reuse pass is skipped and
    // the KV data scrubbed in exactly that case; without FA the explicit mask
    // makes stale cells exact, so reuse is always safe. (When the prompt is
    // >= the resident sequence, every freed cell is overwritten by the suffix
    // decode, so reuse under FA is safe too.)
    const bool fa_shrink_guard = fa_maybe_on && prompt_tokens.size() < resident;
    if (cache_reuse_ > 0 && !fa_shrink_guard && llama_memory_can_shift(mem) &&
        ctx_.yarn_factor() == 1.0f && kv.size() > n_reuse) {
        // -- Middle-of-prompt chunk reuse (llama-server's --cache-reuse,
        //    ported) ---------------------------------------------------------
        //
        // Anamnesis rewrites the <memory> block every turn, so the new prompt
        // diverges from the resident sequence right after the system prefix
        // -- but long stretches of it often exist VERBATIM further into the
        // resident KV, just at different positions (most commonly when the
        // rewritten block SHRANK or kept its length). Ported from
        // tools/server/server-context.cpp's n_cache_reuse loop: when a run of
        // >= cache_reuse_ prompt tokens appears in the resident sequence, the
        // KV cells holding it are MOVED into place (llama_memory_seq_rm frees
        // the gap first, llama_memory_seq_add re-labels the chunk -- exact at
        // this pin via the graph's k_shift input), and the walk continues.
        // Chunks are claimed strictly left-to-right in BOTH spaces (head_p in
        // the prompt, head_c in the KV), so sources never overlap and every
        // move targets the current fill boundary -- which keeps the alive
        // sequence a contiguous [0, n_past) prefix, the only shape
        // llama_decode accepts (its batch must start at seq pos_max + 1).
        //
        // Known limits: (1, upstream shares it) a chunk sitting after an
        // INSERTION (the memory block GREW past its old length) cannot be
        // salvaged, because decoding the insertion would force non-contiguous
        // alive positions -- the real fix is fixed-size memory slots on
        // Anamnesis's side; (2) the pass is gated to yarn_factor == 1 (plain
        // RoPE): the k_shift rotation correction under YaRN scaling is not
        // verified exact at this pin, and inside n_ctx_train (the common
        // case) yarn is inactive anyway -- see the design doc.
        const size_t prompt_cap = prompt_tokens.size() - 1;  // last token always decodes fresh
        size_t head_c = n_reuse;  // walk head over the resident sequence
        size_t head_p = n_reuse;  // fill head over the prompt
        bool cold = false;
        while (head_c < kv.size() && head_p < prompt_cap) {
            size_t n_match = 0;
            while (head_c + n_match < kv.size() && head_p + n_match < prompt_tokens.size() &&
                   kv[head_c + n_match] == prompt_tokens[head_p + n_match]) {
                ++n_match;
            }
            if (static_cast<int>(n_match) >= cache_reuse_) {
                // Move resident cells [head_c, head_c+n_match) to
                // [head_p, head_p+n_match). Order matters: free the gap
                // BEFORE the re-label (the chunk's target positions must be
                // vacant), leave later chunks' cells alive, and drop the
                // obsolete tail only after this move (its range would
                // otherwise swallow not-yet-moved chunks).
                if (!llama_memory_seq_rm(mem, /*seq_id=*/0, static_cast<llama_pos>(head_p),
                                         static_cast<llama_pos>(head_c))) {
                    cold = true;
                    break;
                }
                const int64_t kv_shift = static_cast<int64_t>(head_p) - static_cast<int64_t>(head_c);
                if (kv_shift != 0) {
                    llama_memory_seq_add(mem, /*seq_id=*/0, static_cast<llama_pos>(head_c),
                                         static_cast<llama_pos>(head_c + n_match),
                                         static_cast<llama_pos>(kv_shift));
                }
                if (!llama_memory_seq_rm(mem, /*seq_id=*/0,
                                         static_cast<llama_pos>(head_p + n_match), /*p1=*/-1)) {
                    cold = true;
                    break;
                }
                n_past = head_p + n_match;
                head_p += n_match;
                head_c += n_match;
                ++result.n_chunks_reused;
            } else {
                // Upstream's exact scan semantics: advance the KV head one
                // token per miss (head_p only moves when a chunk lands).
                head_c += 1;
            }
        }
        if (cold) {
            // Half-moved layout: abandon reuse entirely and cold-decode.
            llama_memory_seq_rm(mem, /*seq_id=*/0, /*p0=*/-1, /*p1=*/-1);
            n_past = 0;
            result.n_chunks_reused = 0;
        } else {
            // Drop the obsolete tail so the alive sequence is the contiguous
            // [0, n_past) prefix llama_decode requires.
            if (!llama_memory_seq_rm(mem, /*seq_id=*/0, static_cast<llama_pos>(n_past), /*p1=*/-1)) {
                llama_memory_seq_rm(mem, /*seq_id=*/0, /*p0=*/-1, /*p1=*/-1);
                n_past = 0;
                result.n_chunks_reused = 0;
            }
        }
    } else if (fa_shrink_guard) {
        llama_memory_clear(mem, /*data=*/true);
        prefix_.invalidate(ctx_.epoch());
        n_past = 0;
    } else {
        // Plain path: the common prefix only. Trim [n_past, inf) -- the
        // previous sequence's tail beyond n_past no longer matches anything.
        // When n_past == 0 this is the KV clear the old code was missing (the
        // active-duplication bug: without it, a repeated persona prefix
        // re-decoded at ever-growing positions).
        if (!llama_memory_seq_rm(mem, /*seq_id=*/0, static_cast<llama_pos>(n_past), /*p1=*/-1)) {
            // Partial removal is refused by some cache types (e.g. SWA can't
            // truncate mid-sequence). Full clear + cold decode -- correct,
            // just forfeits reuse for this call.
            llama_memory_seq_rm(mem, /*seq_id=*/0, /*p0=*/-1, /*p1=*/-1);
            n_past = 0;
        }
    }

    if (n_past > prompt_tokens.size() - 1) {
        // The final prompt token always decodes fresh (sampler logits): give
        // it back its cell before reporting/decoding.
        n_past = prompt_tokens.size() - 1;
        llama_memory_seq_rm(mem, /*seq_id=*/0, static_cast<llama_pos>(n_past), /*p1=*/-1);
    }

    result.n_prompt_cached = static_cast<int>(n_past);

    // Decode only the suffix beyond the fill level. Both the prefix path
    // (reusable_prefix caps at prompt.size()-1) and the chunk path (loop cap
    // prompt_cap + post-loop guarantee below) keep at least one token to
    // decode -- the sampler needs fresh logits for the final prompt token.
    // Positions auto-assign to [n_past, ...] since we trimmed the KV to
    // seq_pos_max == n_past-1.
    int n_suffix = static_cast<int>(prompt_tokens.size() - n_past);
    llama_batch prompt_batch = llama_batch_get_one(prompt_tokens.data() + n_past, n_suffix);
    if (n_suffix > 0 && llama_decode(ctx, prompt_batch) != 0) {
        // Scrub the KV data (not just metadata) so a partially-written,
        // now-abandoned decode can't leak into the next request's flash-
        // attention pass -- then drop the tracked prefix to match.
        llama_memory_clear(mem, /*data=*/true);
        prefix_.invalidate(ctx_.epoch());
        result.n_chunks_reused = 0;
        throw std::runtime_error("pleiades_engine: llama_decode failed on prompt");
    }

    // The resident sequence is now exactly this prompt's tokens.
    prefix_.set(prompt_tokens, ctx_.epoch());

    auto t1 = std::chrono::steady_clock::now();
    result.prompt_seconds = std::chrono::duration<double>(t1 - t0).count();

    // -- generation ---------------------------------------------------------
    // Sampler chain per the request (greedy by default -- see
    // build_sampler_chain / SamplingParams). llama_sampler_sample() accepts
    // each sampled token into the chain, so the penalties sampler tracks the
    // generation window on its own; the prompt tokens are not fed to it (the
    // pre-sampling engine didn't either, and greedy needs no history).
    llama_sampler* chain = build_sampler_chain(sampling);

    const bool have_stops = !sampling.stop.empty();
    std::string text;      // full generated text (already stop-truncated once a stop hits)
    size_t emitted = 0;    // bytes of `text` handed to on_token so far (stop path only)
    int generated = 0;
    bool stopped_on_stop = false;
    bool client_cancelled = false;

    for (; generated < n_predict; ++generated) {
        llama_token next = llama_sampler_sample(chain, ctx, -1);
        if (llama_vocab_is_eog(vocab, next)) {
            break;
        }
        std::string piece = token_to_piece(vocab, next);
        text += piece;

        // Stop-string detection: if a complete stop string is now present,
        // trim the output at its first occurrence and halt (the stop text
        // itself is never surfaced -- matches llama-cpp-python).
        if (have_stops) {
            size_t pos = earliest_stop_pos(text, sampling.stop);
            if (pos != std::string::npos) {
                text.erase(pos);
                stopped_on_stop = true;
            }
        }

        // Emission. With no stop strings this is byte-for-byte the old
        // behavior: one on_token(piece) per token. With stop strings we hold
        // back any tail that could still grow into a stop match, so the stop
        // text is never leaked mid-stream.
        bool keep_going = true;
        if (on_token) {
            if (!have_stops) {
                keep_going = on_token(piece);
            } else {
                size_t safe = stopped_on_stop ? text.size()
                                              : text.size() - stop_overlap_suffix(text, sampling.stop);
                if (safe > emitted) {
                    keep_going = on_token(text.substr(emitted, safe - emitted));
                    emitted = safe;
                }
            }
        }

        if (stopped_on_stop) {
            ++generated;  // count the token that completed the stop match
            break;
        }
        if (!keep_going) {
            ++generated;  // count the token we just emitted before stopping
            client_cancelled = true;
            break;
        }

        llama_batch next_batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, next_batch) != 0) {
            llama_sampler_free(chain);
            llama_memory_clear(mem, /*data=*/true);  // scrub partial KV (see prompt-decode failure above)
            prefix_.invalidate(ctx_.epoch());
            throw std::runtime_error("pleiades_engine: llama_decode failed during generation");
        }
        // This token is now resident in the KV -- keep the cache mirroring
        // the KV exactly so the next request can reuse it too.
        prefix_.append(next);
    }

    // Flush any held-back (but safe) tail on a clean end -- EOG or n_predict
    // reached without a stop match. Skipped when a stop halted us (the tail
    // was the stop text, already trimmed) or the client cancelled.
    if (on_token && have_stops && !stopped_on_stop && !client_cancelled && emitted < text.size()) {
        on_token(text.substr(emitted));
    }

    auto t2 = std::chrono::steady_clock::now();
    result.generate_seconds = std::chrono::duration<double>(t2 - t1).count();
    result.n_generated_tokens = generated;
    result.stopped_on_stop_string = stopped_on_stop;
    result.text = std::move(text);

    llama_sampler_free(chain);
    return result;
}

}  // namespace pleiades_engine
