# Pleiades native inference engine — design & phased plan

**Status:** proposed, not started. **Origin:** Fleagle, 2026-07-21 — "build our own
proprietary inference engine ... consult the council." Council = qwen3.8-max
(thinking mode) + a fresh Opus subagent, independently asked to weigh (A) deep fork
of llama.cpp/llama-server vs (B) thin server against libllama's public C API vs
(C) other. Both converged on B. This doc is that synthesis plus the concrete
phase plan.

## Why B, not A

- llama.cpp's C API (`include/llama.h`, `common/`) is comparatively stable; its
  server internals (`tools/server/*.cpp`) are not — a deep fork means merging
  upstream server churn forever, on top of the two runtime forks Pleiades
  already tracks (`~/.pleiades/runtime/moe-fork`, benchmarked in
  `pleiades/runtime.py::find_native`).
- All backend breadth (CUDA/ROCm/Vulkan/Metal/SYCL/CPU) lives in `ggml`, below
  the C API. Linking against libllama inherits it for free — nothing to
  reinvent for the AMD/Intel/Mac targets in the ask.
- Opus's addition: build the core as a **library first** (model manager,
  context governor, request scheduler), with HTTP as one thin transport shim
  on top. That's what makes "Android later" reachable — same core, a JNI shim
  instead of an HTTP shim — instead of a rewrite.
- Qwen's addition: real resize still costs a pause (new KV alloc + reprocess).
  The honest goal is "no reload, no restart, session preserved," not zero-cost
  growth. Prefix-hash caching is what makes that pause small later, but see
  the parity finding below — it's not required for v1 parity.

## Phase 0 — Documentation discovery (done, findings below)

**Base source:** `~/llamacpp-cuda/llama.cpp` — clean checkout of
`github.com/ggerganov/llama.cpp`, `master` @ `b46812de78f8fbcb6cf0154947e8633ebc78d9ac`
(2026-05-08, "Feature hexagon l2 norm #22816"). No submodule/vendor copy inside
`~/Pleiades` itself — Pleiades only *consumes* prebuilt `llama-server` binaries
via `pleiades/runtime.py` today. This checkout is the base to build the new
engine against (or a pinned fork of it — see Phase 5).

**CMake structure confirmed** (`CMakeLists.txt`): `ggml` → `add_subdirectory(src)`
(builds `libllama`) → `add_subdirectory(common)` (sampling/chat/json-grammar/
download helpers, its own static lib) → `add_subdirectory(tools)` (reference
`llama-server`, now itself split into `server-context.cpp` / `server-models.cpp`
/ `server-queue.cpp` / `server-task.cpp` / `server-http.cpp` etc. — useful as a
*reference* for request-lifecycle shape, not something we patch). Confirms the
library is already a first-class, separately-buildable CMake target — no need
to extract it ourselves.

**Allowed APIs — context lifecycle** (`include/llama.h`):
- `llama_model_load_from_file()` / `llama_model_free()` — model load is
  separate from context creation, weights stay resident independent of `n_ctx`.
- `llama_context_default_params()` → set `.n_ctx` → `llama_init_from_model(model, params)`.
  **`n_ctx` is fixed at context creation and has no live-resize call.**
  (`llama_new_context_with_model` is the deprecated old name for the same thing.)
- `llama_free(ctx)` — resize = `llama_free` the old context, then
  `llama_init_from_model` a new one at the new `n_ctx`, same `llama_model*`.
  This is the *exact* native analogue of what
  `pleiades/inference/server.py::EngineState.load()` already does today via
  `llama_cpp.Llama` (see Phase 0 finding on current behavior below) — same
  shape, just via the C API directly instead of through the Python binding.

**Allowed APIs — state / KV** (`include/llama.h`):
- `llama_state_get_size/get_data/set_data`, `llama_state_seq_get_size/get_data/set_data`,
  `llama_state_(seq_)save_file/load_file` — whole-context or per-sequence KV
  snapshot/restore. **Available for a v2 prefix-cache optimization, not required
  for v1 parity** (see finding below).
- `llama_memory_seq_rm/cp/keep/add/div`, `llama_memory_seq_pos_min/max`,
  `llama_memory_can_shift` — KV manipulation primitives (trim, copy, shift) via
  `llama_get_memory(ctx)`.

**Allowed APIs — inference:**
- `llama_batch_init/free`, `llama_decode`, `llama_encode`.
- `llama_sampler_chain_init/add`, `llama_sampler_init_{greedy,dist,top_k,top_p,
  min_p,typical,temp,temp_ext}` — full sampler chain, matches what
  `llama-cpp-python` already exposes to `EngineState` today.
- `common/chat.h` — jinja-capable chat template application (the
  `llama_chat_apply_template` C function is explicitly pre-defined-templates-only
  per its own doc comment; `common/chat.h` is the one to use for real template
  fidelity).
- `common/json-schema-to-grammar.h` — structured-output/tool-calling grammar
  generation, if/when Pleiades' harness wants it natively.

**Critical finding — current elastic engine's actual resize semantics**
(`pleiades/inference/server.py:66-120`): `EngineState.resize()` does **not**
preserve KV/session state across a resize today. `load()` closes the old
`Llama` object entirely and constructs a brand-new one at the new `n_ctx`;
speed comes from the model file staying hot in the OS page cache (mmap), not
from any KV continuity. This is fine because Anamnesis is turn-based, not
KV-cache-based (per `[[pleiades-project]]` memory) — it resends whatever
prefix a turn needs; the engine has no standing conversation state to lose.

**This simplifies v1 scope a lot: parity with today's behavior does *not*
require `llama_state_*` session preservation.** Resize = free + recreate +
let the caller's next request reprocess its own prompt, exactly like today.
State-save/restore and prefix-hash caching (qwen's addition) become a Phase 6
latency optimization on top of a working v1, not a blocking requirement.

**Gear table to port verbatim** (`pleiades/hardware.py:578`):
`CONTEXT_GEARS = [4096, 8192, 16384, 32768, 65536, 131072]`, plus
`EngineState.clamp_gear()` / `next_gear()` (lines 100-109) and
`resolve_layers()` (referenced from `pleiades/hardware.py`) — these are the
policy pieces to reimplement in the new engine's context governor, not
reinvent.

**Runtime discovery integration point:** `pleiades/runtime.py::find_native()`
(prefers `cuda-main` > `moe-fork` > legacy prebuilt via a `rank()` function) and
`pleiades/launch.py::build_command()` (`place_ctx = n_ctx_max if native else
n_ctx_v`) are the two places a third runtime option plugs in later (Phase 5).

**Not found locally / out of scope for this doc:** no local checkout of Ion's
`llama.cpp-Statewise` fork or its `FLEAGLE_HANDOFF.md` — only a compiled
`~/.pleiades/runtime/moe-fork` binary exists. Statewise integration (Phase 7)
needs Ion's source cloned first; not blocking anything below.

## Phase 1 — Engine core skeleton (library-first)

**Goal:** a new CMake target, `pleiades-engine`, that links `libllama` +
`common` from `~/llamacpp-cuda/llama.cpp` (or a pinned submodule of it — decide
at kickoff whether Pleiades vendors a pinned commit vs. building against the
existing checkout) and exposes exactly three C++ classes, no HTTP yet:

- `ModelManager` — wraps `llama_model_load_from_file`/`llama_model_free`.
  One model resident at a time (matches today's single-model-per-process
  design in `EngineState`).
- `ContextGovernor` — owns the live `llama_context*`. `create(n_ctx)`,
  `resize(new_n_ctx)` (free + recreate, per the Phase 0 finding), `clamp_gear()`
  /`next_gear()` ported from `pleiades/hardware.py:100-109` verbatim (copy the
  gear table and clamping logic, don't reinvent it).
  `resolve_layers()`-equivalent GPU/CPU split logic can shell out to the
  existing Python `resolve_layers()` at first (subprocess call) to avoid
  duplicating that logic in two languages before it's proven worth porting.
- `Engine` — the request-level façade: `chat(messages) -> tokens`, `decode`
  loop using `llama_batch_init`/`llama_decode` + a sampler chain built from
  `llama_sampler_chain_init`/`llama_sampler_init_*`.

**Deliverable for this phase:** a CLI smoke-test binary (`pleiades-engine-cli`)
that loads a real GGUF, runs one prompt to completion, and prints tokens/sec —
no server, no Anamnesis integration yet.

**Verification:** builds clean via CMake against the existing llama.cpp
checkout; smoke-test binary produces coherent output on a small test GGUF
(reuse whatever synthetic/small GGUF fixtures `tests/` already has, per
`make_gguf` mentioned in Pleiades' own test conventions — don't burn the real
35B model on this phase).

**Anti-patterns:** do not touch `tools/server/*` in the llama.cpp checkout.
Do not invent new libllama functions — every call above is cited to a
specific line/section already confirmed present in `include/llama.h`.

## Phase 2 — Context governor parity + resize benchmark

**Goal:** exercise `ContextGovernor::resize()` against the real Ornith-35B MoE
GGUF on this box (RTX 2080 Ti) and compare wall-clock resize latency against
today's Python engine's `EngineState.resize()` (same model, same gear jump,
e.g. 4096→32768). Also compare against native `llama-server`'s only option
today (full process restart) to quantify the win the whole project is for.

**Verification checklist:**
- Resize latency: native governor ≤ Python engine's resize latency (parity
  floor) — ideally much lower since we skip the Python interpreter/binding
  overhead and can keep more state hot.
- No process restart, no full model reload from disk on any resize call.
- Output correctness: same prompt run at the pre- and post-resize `n_ctx`
  produces sane completions (basic regression, not benchmark-grade eval).

**Anti-patterns:** don't attempt `llama_state_*` save/restore in this phase —
per the Phase 0 finding, it isn't required for parity and adds a whole class
of bugs (state blob versioning, sequence-id bookkeeping) before the core
resize path is even proven.

## Phase 3 — HTTP transport shim (OpenAI-compatible)

**Goal:** thin HTTP layer over the Phase 1/2 core, matching the existing
contract in `pleiades/inference/server.py` closely enough that
`pleiades/launch.py` and Anamnesis's proxy don't need client-side changes:
`POST /v1/chat/completions` (streaming + non-streaming), `POST /resize
{"n_ctx": N}` returning `{n_ctx, took_ms}` (same shape as today,
`server.py:111-119`), `GET /props` returning `{n_ctx, n_ctx_train, n_ctx_max,
resizable:true, gears:[...]}` (same shape as `server.py:164-174`).

Use `common/http.h`/the `cpp-httplib` vendor already present in the llama.cpp
checkout (`vendor/cpp-httplib`, wired in `CMakeLists.txt:201`) rather than
pulling in a new HTTP dependency — it's already a proven, header-only choice
in this exact codebase.

**Verification:** existing Pleiades integration tests that hit the inference
server's HTTP contract pass unmodified against this new binary (point
`PLEIADES_RUNTIME_BIN`-equivalent env var at it for a manual A/B — don't wire
into `find_native()` yet, that's Phase 5).

## Phase 4 — Bench harness + test suite

**Goal:** a repeatable side-by-side benchmark (tokens/sec prefill+decode,
resize latency across the full gear ladder, memory footprint) run against
all three runtimes on the same hardware: today's Python elastic engine,
native `llama-server` (restart-based), and the new engine. Plus a proper
`tests/` suite for the new engine mirroring `tests/test_context_plan.py`'s
existing conventions (synthetic small-GGUF fixtures, no real 35B model in CI).

**Verification:** numbers written up in this doc's changelog section (append,
don't rewrite the design above); test suite green.

## Phase 5 — Cutover

**Goal:** wire the new engine into `pleiades/runtime.py::find_native()` as a
third ranked option and `pleiades/launch.py::build_command()`, **feature-flagged
off by default** until Phase 4's numbers clear the parity bar. Flip the
default only after that; demote native `llama-server` to an explicit fallback
(kept, not deleted — same posture as the existing `moe-fork` opt-in pattern in
`runtime.py`'s `rank()` function).

**Anti-patterns:** no silent default flip. This follows the same "prove it on
the real 35B MoE model first" discipline already used for the `moe-fork`
benchmark note in `runtime.py`.

## Phase 6 — Prefix-cache optimization (post-parity)

**Goal:** now that resize works and is in production, use
`llama_state_seq_get_data`/`llama_state_seq_set_data` to avoid reprocessing a
stable shared prefix (persona/owner-facts/task-block that Anamnesis resends
most turns) across a resize, per qwen's original suggestion. This is a
latency optimization layered on a working system, not a prerequisite.

## Phase 7 — Statewise integration (adaptive MoE expert residency)

**Goal:** bolt Ion's Statewise adaptive expert-caching (`llama_statewise_swap`
C API, per the earlier council discussion) onto the new engine's
`ModelManager`, once it's the default. Requires cloning
`github.com/ionizedd/llama.cpp-Statewise` locally first (not present on this
box currently) and reading its actual `FLEAGLE_HANDOFF.md` before touching
anything — not yet done.

## Phase 8 — Cross-platform matrix (later milestones, per the ask)

CUDA/Linux is the proving ground (Phases 1-6). Once cut over:
- **AMD (ROCm/Vulkan) + Windows + CPU-only:** inherited from `ggml`'s existing
  backend support — verify the CMake build + Phase 3 HTTP shim on each, no
  new engine-core work expected.
- **Mac (Metal):** same — `ggml` already has a Metal backend upstream.
- **Android:** this is where Opus's "library first" design pays for itself —
  same `ModelManager`/`ContextGovernor`/`Engine` core, a JNI shim instead of
  the Phase 3 HTTP shim. Explicitly deferred; not attempted until desktop
  platforms are solid.

---
*Written 2026-07-21 following a qwen3.8-max + Opus council consult, per
Fleagle's direction. See `~/Documents/Claude/Projects/.../memory/pleiades-project.md`
for session history; this file is the living design doc going forward —
update it in place as phases complete, don't create parallel docs.*
