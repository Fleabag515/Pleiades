#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "llama.h"
#include "pleiades_engine/engine.h"

namespace pleiades_engine {

// Slot-0 state persistence: the engine's equivalent of llama-server's
// --slot-save-path + POST /slots/0?action=save|restore (Phase 1 parity -- see
// docs/specs/2026-08-30-engine-default-elastic-context.md). models.py already
// drives exactly that HTTP contract best-effort around stop()/start(), so the
// engine serves the same semantics: persist sequence-0's full state (logits +
// embedding + KV memory) to one file, and restore it into a fresh process so a
// model restart skips the cold re-prefill of the persona/memory prefix.
//
// File format (our own; llama-server's slot files are NOT interchangeable --
// different binaries embed different state layouts):
//   magic  "PLST" | u32 version(1) | u32 n_tokens | i32 tokens[n]
//          | u64 state_bytes | raw llama_state_seq_get_data blob
// The token list rides along because the raw state blob does not expose the
// resident token ids back to us, and Engine's PrefixCache needs them to keep
// mirroring the KV after a restore (without them a restored state would still
// WORK, but every request would cold-decode -- losing most of the point).

// Serialize sequence 0 of `ctx` (plus `prefix_tokens`, the Engine's current
// PrefixCache view of that sequence) to `path`. Returns false (never throws)
// when there is no state to save, the directory can't be written, or the
// state copy fails -- callers treat all of these as "nothing saved", which is
// the same best-effort stance models.py takes.
bool save_slot_state(llama_context* ctx, const std::vector<llama_token>& prefix_tokens,
                     const std::string& path);

// Load a file written by save_slot_state and restore it into sequence 0 of
// `ctx`, then rebind `engine`'s PrefixCache to the restored token list (so
// subsequent requests reuse the restored KV instead of cold-decoding over it).
// Returns false (never throws) on any parse failure, model/state mismatch,
// or state that doesn't fit the current context -- the caller just does an
// ordinary cold start, exactly like a missing save file.
bool restore_slot_state(llama_context* ctx, Engine& engine, const std::string& path);

}  // namespace pleiades_engine
