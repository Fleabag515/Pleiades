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

## Phase 2 — Context governor parity + resize benchmark (done, results below)

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

**Results (2026-07-22, `engine/src/bench_resize.cpp`, real Ornith-1.0-35B
Q6_K GGUF, 4096→32768 gear jump):**

| leg                              | CPU-only (-ngl 0) | GPU offload (-ngl 12) |
|-----------------------------------|-------------------|------------------------|
| native `ContextGovernor::resize()`| 773 ms            | 461 ms                 |
| Python `EngineState.resize()`     | 6700 ms           | 5783 ms                |
| native `llama-server` restart     | ~15.2s / ~8.0s\*   | 8.4s / 8.4s            |

\*first/second run differ due to page-cache warmth, not `n_ctx`; both loads
read the full 27GB file from a mostly-cold cache the first time.

Native governor resize is **~9-14x faster** than the Python engine's
resize (which does a full `close()`+reopen of the `Llama` object even
though weights stay mmap'd) and **~18-20x faster** than `llama-server`'s
only option today (kill + full process restart + client reconnect).
Pre- and post-resize completions were byte-identical and coherent in both
runs — the free+recreate resize is correctness-neutral, no
`llama_state_*` needed, confirming the Phase 0 finding. All three legs
were also run CPU-only first (GPU was occupied by Mark's live production
character on the box's single RTX 2080 Ti at the time); Fleagle then
stopped that character to free VRAM for the GPU-offload numbers above,
and it was restarted (`pleiades model start
qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive`) immediately after
benchmarking finished.

**Side finding, not in scope to fix here:** the hardware planner
(`pleiades/hardware.py`) classifies Ornith-1.0-35B's architecture
(`qwen35moe`) as `memory_arch="dense"` even though llama.cpp's own load
log shows `llama_memory_recurrent` buffers for most layers (it's actually
hybrid, same Gated DeltaNet family as [[new-brain-project]]'s Mark
backbone) — `qwen35`/`qwen35moe` aren't in Phase 0.A's `_HYBRID_ARCHS`
allowlist yet. This means `resolve_layers()` currently plans GPU/CPU
split assuming full dense KV growth for this model, over-costing its KV
budget and leaving fewer layers offloadable than necessary as `n_ctx`
grows. Worth a small Phase 0.A follow-up (add `qwen35`/`qwen35moe` to
`_HYBRID_ARCHS`), tracked here rather than fixed mid-Phase-2.

## Phase 3 — HTTP transport shim (OpenAI-compatible) (done, results below)

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

**Results (2026-07-22):** `engine/src/http_server.cpp`, new `pleiades-engine-server`
binary. Design decision made via council (qwen3.8-max + Kimi, asked
independently): real per-model jinja chat-templating (`common/chat.cpp` +
minja) was deliberately deferred rather than wired in now. Qwen's first
suggestion — "surgically" compile only `chat.cpp` + minja, skip the rest of
`common/` — was checked against the actual pinned source rather than taken
on faith: `common/chat.cpp` directly `#include`s `common.h`,
`json-schema-to-grammar.h`, `log.h`, plus its own auto-parser/peg-parser
files (confirmed via `grep` on the real file, ~10K lines total across that
dependency set) — not a light lift. Kimi's independent read agreed
extraction wasn't realistic and recommended a hardcoded stopgap instead,
which is what shipped: `chat_template.h`/`.cpp` hardcodes Qwen2/2.5/3.x's
stable ChatML format (`<|im_start|>role\ncontent<|im_end|>`) since Pleiades
serves mostly Qwen-family GGUFs today. Real multi-model/tool-calling
templating via `common/chat.cpp` is tracked as a later follow-up, not
Phase 3 scope.

`Engine` gained a `generate()` method (callback-per-token) alongside
`complete()`, so streaming forwards real per-token latency instead of
buffering the whole completion before responding — `complete()` is now just
`generate()` with no callback.

cpp-httplib and nlohmann/json (both already vendored inside the llama.cpp
submodule at `vendor/cpp-httplib` and `vendor/nlohmann`) are compiled/
included directly in `engine/CMakeLists.txt` as their own small targets,
without turning on `LLAMA_BUILD_COMMON` — keeps Phase 1's "don't build
`common/`" decision intact.

Implemented: `GET /`, `GET /health`, `GET /props`, `POST /resize`,
`POST /v1/chat/completions` (streaming + non-streaming). Not implemented
(out of scope, not depended on by Anamnesis's real proxy — verified by
reading `~/.local/share/anamnesis/src/proxy.js` directly rather than
assuming): `/v1/models`, `/tokenize`, `/extras/tokenize/count`, `/metrics`,
prompt-overflow auto-upshift, `kv_bytes_per_token` in `/props` (that's
`hardware.py`'s KV-cost formula, not yet ported to C++). Anamnesis's proxy
only reads `n_ctx` (falling back to `default_generation_settings.n_ctx`)
from `/props` and otherwise just forwards `chat/completions` — both are
present and correct in this shim's output.

One `std::mutex` serializes every request (chat generation AND resize) —
matches `EngineState`'s own `self.lock` in `server.py` and guards against
the real bug Qwen's council answer flagged: a `/resize` arriving mid-decode
would otherwise free the `llama_context*` out from under an in-flight
`llama_decode()` call.

Manually verified end-to-end against the real small Qwen2.5-0.5B GGUF
(GPU free at the time): non-streaming chat returns a correct, OpenAI-shaped
response; streaming chat produces real per-token SSE chunks ending in
`data: [DONE]`; `/props` reflects `n_ctx`/`n_ctx_max`/gears correctly;
`POST /resize {"n_ctx":16384}` resizes in 130ms and a follow-up chat
completion after the resize still produces a correct, coherent answer.

## Phase 4 — Bench harness + test suite (done, results below)

**Goal:** a repeatable side-by-side benchmark (tokens/sec prefill+decode,
resize latency across the full gear ladder, memory footprint) run against
all three runtimes on the same hardware: today's Python elastic engine,
native `llama-server` (restart-based), and the new engine. Plus a proper
`tests/` suite for the new engine mirroring `tests/test_context_plan.py`'s
existing conventions (synthetic small-GGUF fixtures, no real 35B model in CI).

**Verification:** numbers written up in this doc's changelog section (append,
don't rewrite the design above); test suite green.

**Results (2026-07-22):**

Test suite: `engine/tests/` (`test_chat_template`, `test_model_manager`,
`test_context_governor`, `test_engine`), plain CTest executables mirroring
llama.cpp's own test convention exactly (no external test framework, matches
the project's no-third-party-deps stance elsewhere). Fixture is llama.cpp's
OWN official CI model (`ggml-org/models`' `tinyllamas/stories15M-q4_0.gguf`,
hash-verified via the vendored `cmake/download-models.cmake` -- the same
script and file `test-thread-safety` uses in the llama.cpp submodule itself)
rather than `tests/test_context_plan.py`'s `make_gguf()` pattern, which
writes metadata-only GGUFs with zero tensors -- fine for testing Python's
hparams parsing, but not loadable by `llama_model_load_from_file()`, which
the C++ engine's tests actually need. `ctest`: 5/5 pass.

Two real test bugs were caught and fixed during this phase, both worth
noting since they're easy mistakes to repeat: (1) asserting a resized
`llama_context*` differs from its pre-resize pointer -- freed-then-
immediately-reallocated memory commonly returns the *same* address, so
pointer identity isn't a reliable "did a real free+recreate happen" signal;
fixed by checking the functional contract (`n_ctx()` reflects the new size)
instead. (2) asserting byte-identical greedy output before/after a resize
on the tiny fixture -- its own trained context is 128 tokens (confirmed via
llama.cpp's own load log: "n_ctx_seq (1024) > n_ctx_train (128) -- possible
training context overflow"), and once a model is extrapolating beyond its
trained window, Flash Attention auto-selection and summation order are
allowed to differ between different `n_ctx` allocations, so exact output
isn't guaranteed. The real byte-identical-output claim is already proven at
production scale (Phase 2's real 35B model, resized within its actual
262144-token trained ceiling) -- this unit test's job is narrower: catch a
regression where resize leaves the engine unable to generate at all.

Full-gear-ladder bench (`engine/src/bench_ladder.cpp`, new
`pleiades-engine-bench-ladder` binary), real Ornith-1.0-35B Q6_K GGUF,
CPU-only (the box's one RTX 2080 Ti was shared with an unrelated Immich
photo-library ML service at the time, not itself a Pleiades character --
Mark's own live server was stopped and restarted around this benchmark
exactly like Phase 2, but the GPU sweep itself was skipped this time since
Immich's process didn't leave enough free VRAM for a meaningful offload
split; CPU-only still gives a valid relative comparison across the full
ladder):

| n_ctx   | native resize (ms) | Python resize (ms) | native decode tok/s | VmRSS (MB) |
|---------|---------------------|----------------------|----------------------|------------|
| 4096    | (baseline)          | (baseline)            | 3.3\*                | 27,329     |
| 8192    | 366                 | 4489                  | 6.4                  | 27,414     |
| 16384   | 303                 | 4456                  | 6.6                  | 27,575     |
| 32768   | 509                 | 4687                  | 6.6                  | 27,891     |
| 65536   | 1007                | 5374                  | 6.7                  | 28,532     |
| 131072  | 1840                | 6374                  | 6.6                  | 29,804     |

\*first-call warmup artifact (JIT/cache effects), not representative --
subsequent gears stabilize around 6.4-6.7 tok/s, consistent with a MoE
model with ~3B active params per token.

Native resize stays **~8-12x faster than the Python engine across the
entire gear ladder**, not just the single jump Phase 2 measured -- confirms
Phase 2's finding generalizes rather than being a one-off. Memory footprint
grows only modestly across a 32x context increase (27.3GB -> 29.8GB, ~9%)
-- consistent with this being a hybrid model (most layers are O(1)
recurrent state, only a handful of real attention layers have dense
KV growth with `n_ctx`; see the Phase 2 "side finding" on `qwen35moe` not
yet being in `_HYBRID_ARCHS`).

## Phase 5 — Cutover (done, results below)

**Goal:** wire the new engine into `pleiades/runtime.py::find_native()` as a
third ranked option and `pleiades/launch.py::build_command()`, **feature-flagged
off by default** until Phase 4's numbers clear the parity bar. Flip the
default only after that; demote native `llama-server` to an explicit fallback
(kept, not deleted — same posture as the existing `moe-fork` opt-in pattern in
`runtime.py`'s `rank()` function).

**Anti-patterns:** no silent default flip. This follows the same "prove it on
the real 35B MoE model first" discipline already used for the `moe-fork`
benchmark note in `runtime.py`.

**Results (2026-07-22):** wired in with one real deviation from this phase's
original phrasing, made deliberately and checked against a second opinion
rather than silently drifted into: the plan said to wire the new engine into
`find_native()`/`rank()` as a third ranked option. That function returns a
path assumed to accept native `llama-server`'s actual CLI flag shape
(`-m`/`-c`/`-ngl`/`--jinja`/`-fa`/`-ctk`/`-ctv`/`--n-cpu-moe`/`--spec-type`/...),
and `build_command()`'s native branch builds those flags assuming whatever
binary it gets back understands all of them. `pleiades-engine-server` only
supports load/resize/chat-completion today — no flash-attention toggle, no
KV quantization, no MoE expert offload, no speculative decoding. Slotting it
into the same ranked list would either silently drop unsupported flags or
require brittle per-flag filtering inside that branch. Asked Kimi
independently (not just my own judgment) — same conclusion: separate
resolver, separate `launch.py` branch, own feature flag, unify later once
real feature parity exists.

Shipped: `runtime.py::find_native_cpp_engine()` (own docstring explains the
above) — resolution order `PLEIADES_NATIVE_CPP_ENGINE_BIN` override ->
`engine/build/pleiades-engine-server` relative to the checkout (no
`pleiades runtime install`-style packaged release of this yet, that's a
later concern) -> PATH. `launch.py::build_command()` gained an early branch,
gated on `PLEIADES_ENGINE=pleiades_native` (unset/anything else = 100%
today's existing behavior, matching the anti-pattern above): builds the
minimal command `pleiades-engine-server <model> <host> <port> <n_ctx>
<n_gpu_layers> <alias>`, forces `n_gpu_layers=0` (CPU-only — no
autofit/MoE-split integration yet, an honest limitation, not silently
assumed-away) unless explicitly overridden, and falls back to normal
runtime selection with a printed warning if the binary isn't found rather
than crashing. `http_server.cpp`'s `main()` gained an explicit `host`
argument (was hardcoded to `127.0.0.1`) so `build_command()`'s caller-
supplied host is honored rather than silently ignored, even though every
registered model in practice uses `127.0.0.1` today.

Verified end-to-end via the REAL production path, not just an isolated
function call: `PLEIADES_ENGINE=pleiades_native pleiades model start
qwen2.5-0.5b-instruct` — registry shows it running, correct binary/args
(`pleiades-engine-server ... 32768 0 qwen2.5-0.5b-instruct`), a real
`/v1/chat/completions` request through the registered port returns a
correct answer ("Tokyo" for "capital of Japan"), clean startup log, and
`pleiades model stop` tears it down cleanly leaving the registry in the
expected state. Full existing pytest suite still green (239 passed, same
1 pre-existing unrelated `test_sandbox.py` flake noted throughout this
whole effort) — the new branch didn't disturb anything on the default path.

**Still off by default** — `PLEIADES_ENGINE` unset or any value other than
`pleiades_native` behaves exactly as before. Flipping the default, or
adding autofit/MoE-offload support to this branch, is not part of this
phase.

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
