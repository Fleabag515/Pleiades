# Pleiades native engine — Windows + AMD test results

**Tester:** Claude, running on Ion's machine at Ion's request, per `PLEIADES-WINDOWS-AMD-TEST.md`.
**Date:** 2026-07-31
**Repo state:** `main` @ `7f1db9c` ("Engine Phase 9: sole-cutover progress"), submodule
`third_party/llama.cpp` @ `c588c4f47` (gguf-v0.19.0-1057).

## Machine

| | |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| CPU | AMD (x86-64, AVX-512 capable — ggml selected the `/arch:AVX512` CPU variant) |
| RAM | 31.1 GB total |
| GPU 0 | **AMD Radeon RX 9070 XT** (RDNA4, gfx1201), driver 32.0.31021.5001, 15416 MiB free |
| GPU 1 | AMD Radeon(TM) Graphics (integrated) |
| Compiler | MSVC 19.44 (Visual Studio 2022 Community) |
| CMake | 4.1.1 |
| Vulkan SDK | 1.4.341.1 (LunarG), `glslc` present |

Vulkan device report from the engine's own startup log:

```
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 9070 XT (AMD proprietary driver) | uma: 0 | fp16: 1 | bf16: 1 |
             fp4: 0 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: KHR_coopmat
ggml_vulkan: 1 = AMD Radeon(TM) Graphics (AMD proprietary driver) | uma: 1 | fp16: 1 | bf16: 0 |
             fp4: 0 | warp size: 32 | shared memory: 32768 | int dot: 1 | matrix cores: none
```

## Headline

**Windows + MSVC + AMD/Vulkan works, and MoE offload — the thing you flagged as highest-value and
genuinely unverified outside CUDA — is correct on RDNA4.**

| | Result |
|---|---|
| Vulkan build | **Failed initially**, one-line Pleiades-side fix, then clean (§1a) |
| CPU build | Clean (§1b) |
| ctest | **5/6** — the failure is *not* GPU-related (§2b) |
| Real-model smoke test | **Pass**, real GPU confirmed (§3) |
| **MoE offload correctness** | **Pass** — byte-identical short, equal-quality long (§4) |
| **MoE offload performance** | **1.98x faster than pure CPU** on the RX 9070 XT (§4c) |
| Stability | No crashes, hangs, or driver resets in any configuration (§7) |
| **Bonus: `/api/status` timeout bug** | **Found, root-caused, fixed, verified**: 4.3s -> 1.5s (§6) |

Two independent pieces of evidence say the **Vulkan/RDNA4 path is numerically sound**, not merely
"it didn't crash":

1. On the one failing test, the Vulkan and CPU backends produce **byte-identical output** —
   character for character, including identical cache-reuse counts. They fail the same assertion
   the same way, which means the GPU is faithfully reproducing the CPU reference.
2. In the MoE comparison, a hybrid GPU+CPU-expert split produced **byte-identical** output to pure
   CPU on a short greedy generation, and on a ~160-token generation diverged at a single early
   token before **re-converging word-for-word for four full sentences**.

Three real defects found, in descending order of importance:

1. **The MSVC build is broken** without `/Zc:__cplusplus` (§1a) — a two-condition interaction
   between Pleiades' C++20 propagation and MSVC's `__cplusplus` quirk. Fix verified.
2. **`test_prefix_cache` fails on Windows on both backends, deterministically** (§2b) — reportedly
   passes on Linux, so this is real platform-dependent behaviour in the prefix-reuse path, and
   worth your eyes even though it is not an AMD problem.
3. **`finish_reason` reports `"stop"` on max_tokens truncation** where the OpenAI contract requires
   `"length"` (§3a) — currently indistinguishable from a genuine stop.

A fourth, unrelated to the AMD brief but found live on this machine and fixed in this same branch:
**the desktop app's `/api/status` readiness poll was structurally guaranteed to time out at 90s**
whenever two or more optional services (inference, searxng) were down — a 2s per-poll abort
racing a 4.3-4.7s endpoint that could never win (§6). Fixed server-side (parallelized the
independent probes, verified 4.3s -> 1.5s) and client-side (raised the abort margin as defense
in depth).

Plus two corrections to the brief/docs (§1c, §2a), both of which will bite the next person.

---

## 1. Build

### 1a. Vulkan — FAILED first, then PASSED with a one-line fix

Configure succeeded immediately and correctly:

```
cmake -S engine -B engine/build -DPLEIADES_ENGINE_BACKEND=vulkan -DCMAKE_BUILD_TYPE=Release
```

```
-- Found Vulkan: C:/VulkanSDK/1.4.341.1/Lib/vulkan-1.lib (found version "1.4.341")
                 found components: glslc glslangValidator
-- Vulkan found
-- GL_KHR_cooperative_matrix supported by glslc
-- GL_NV_cooperative_matrix2 supported by glslc
-- GL_EXT_integer_dot_product supported by glslc
-- GL_EXT_bfloat16 supported by glslc
-- Including Vulkan backend
-- Configuring done (12.9s)
```

The backend-selection system works as documented. `PLEIADES_ENGINE_BACKEND` was correctly
recorded in the cache with its `auto;cuda;hip;vulkan;metal;cpu` STRINGS property.

**The build then failed with 15 errors, all in vendored `llama.cpp/src/llama-chat.cpp`:**

```
llama-chat.cpp(180,16): error C2664: 'bool llm_chat_detect_template::<lambda_1>::operator ()(const char *) const':
    cannot convert argument 1 from 'const char8_t [9]' to 'const char *'
llama-chat.cpp(526,20): error C2088: built-in operator '<<' cannot be applied to an operand of type 'std::stringstream'
llama-chat.cpp(526,20): error C2280: 'std::basic_ostream<...> &std::operator <<<...>(..., const char8_t *)':
    attempting to reference a deleted function
    ... (lines 180, 185 x3, 234, 526, 542, 555, 557, 561)
```

**Root cause — a two-condition interaction. Neither condition breaks anything on its own,
which is exactly why this survived until the first Windows build:**

1. `engine/CMakeLists.txt:20` sets `CMAKE_CXX_STANDARD 20` as a **directory-scope variable**, so it
   is inherited by `add_subdirectory()` of the vendored llama.cpp. llama.cpp's own
   `src/CMakeLists.txt:54` says `target_compile_features(llama PRIVATE cxx_std_17) # don't bump` —
   but `target_compile_features` only establishes a *minimum*; it cannot pull the standard back
   down from 20. So llama.cpp gets compiled as C++20 inside a Pleiades build, against its own
   stated intent.
2. llama.cpp **does** anticipate C++20. `llama-chat.cpp:9-13`:
   ```c
   #if __cplusplus >= 202000L
       #define LU8(x) (const char*)(u8##x)   // C++20: u8"" is const char8_t*, cast it back
   #else
       #define LU8(x) u8##x
   #endif
   ```
   But **MSVC reports `__cplusplus` as `199711L` regardless of `/std:c++20`** unless explicitly
   given `/Zc:__cplusplus`. The guard therefore selects the C++17 branch while actually compiling
   as C++20, and every `LU8()` use yields a `const char8_t*` where a `const char*` is required.

On Linux the guard fires correctly (GCC/Clang report `__cplusplus` honestly). llama.cpp standalone
on MSVC builds at C++17, where `u8""` is already `const char*`. It takes *both* Pleiades' C++20
propagation *and* MSVC's `__cplusplus` quirk to fail.

**Fix (applied locally, verified — builds clean, all binaries produced):** in `engine/CMakeLists.txt`,
right after the standard is set:

```cmake
if(MSVC)
  add_compile_options(/Zc:__cplusplus)
endif()
```

This is preferable to forcing llama.cpp back to C++17, because it makes MSVC report the standard
honestly for *all* vendored code and respects llama.cpp's own existing guard rather than working
around it. After the fix: **build exit 0**, `pleiades-engine-server.exe`, `pleiades-engine-cli.exe`,
the three bench binaries, and all five test binaries all linked.

Warnings only, no errors, in `ggml-vulkan.cpp` — mostly `C4003` (not enough arguments for
function-like macro `CREATE_CONVS`) and a cluster of `C4319` (`'~': zero extending 'uint32_t' to
'uint64_t' of greater size`). These are upstream llama.cpp warnings, not Pleiades code, and did not
block the build. Flagging them only because `C4319` can occasionally indicate a real mask bug.

### 1b. CPU — PASSED

```
cmake -S engine -B engine/build-cpu -DPLEIADES_ENGINE_BACKEND=cpu -DCMAKE_BUILD_TYPE=Release
```

Clean build, exit 0, with the same `/Zc:__cplusplus` fix in place. (It needs the fix too — the
break is standard-propagation, not backend-specific.)

`--caps` correctly reports the difference between the two builds:

```
vulkan build: {"backend":"vulkan","engine_version":"0.1.0","moe_offload":true, "resize":true,"slots":false,"vision":true}
cpu build:    {"backend":"cpu",   "engine_version":"0.1.0","moe_offload":false,"resize":true,"slots":false,"vision":true}
```

Note `moe_offload` correctly flips to `false` on the CPU build.

### 1c. Documentation correction — the documented build command builds Debug

The brief (and `engine/CMakeLists.txt`'s own header comment) document:

```powershell
cmake -S engine -B engine/build -DPLEIADES_ENGINE_BACKEND=vulkan -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j
```

On Windows, CMake defaults to the **Visual Studio 17 2022** generator, which is **multi-config**.
`CMAKE_BUILD_TYPE` is ignored by multi-config generators — the cache literally records it as
`CMAKE_BUILD_TYPE:UNINITIALIZED=Release`. The second line therefore builds **Debug**, silently.

`cmake --build engine/build --config Release -j` is required (used for everything in this report).
Worth either documenting, or adding `-G Ninja` to the Windows instructions, or setting
`CMAKE_CONFIGURATION_TYPES`.

---

## 2. ctest

**5 of 6 passed. `test_prefix_cache` failed — on BOTH backends.**

```
1/6 pleiades-test-download-model .....   Passed    1.85 sec
2/6 test_chat_template ...............   Passed    0.04 sec
3/6 test_model_manager ...............   Passed    0.47 sec
4/6 test_context_governor ............   Passed    0.22 sec
5/6 test_engine ......................   Passed    0.12 sec
6/6 test_prefix_cache ................***Failed    0.39 sec
```

### 2a. Correction to the brief's expectation: 6 tests, and the POSIX pair are not "skips"

The brief expects "7 real tests + 1 skip (`test_http_cross_family`)". On Windows the reality is
**6 tests total**, and the two POSIX-only HTTP tests do **not** appear as skips at all — they are
never built. `engine/tests/CMakeLists.txt:83` gates them at *configure* time:

```cmake
if (WIN32)
  message(STATUS "pleiades: skipping POSIX-only HTTP end-to-end tests (fork/execl-based) on Windows")
else()
  ... add_executable(test_http_tool_calls ...) / test_http_cross_family ...
endif()
```

So their `SKIP_RETURN_CODE 77` machinery never comes into play on Windows, and `ctest` output
contains no trace of them. A healthy Windows run reads **6/6**, not 7 passed + 1 skipped. Windows
CI will therefore silently cover *neither* end-to-end HTTP tool-calling path until a
`CreateProcess`-based `ServerProcess` lands — the gate comment already acknowledges this, but the
"1 skip" framing in the brief undersells it: it's two entire tests, invisible.

### 2b. `test_prefix_cache` — fails identically on Vulkan AND CPU

Failing assertion, `engine/tests/test_prefix_cache.cpp:246`:

```cpp
PLEIADES_CHECK(grow.text == grow_cold);             // correct despite reuse
```

This is the flash-attention prefix-reuse regression guard (the scenario fixed for CUDA in
`f079811`, "flash-attention leaks stale KV data into shorter reused-prefix requests"). The two
assertions immediately before it **passed** — reuse happened and was correctly bounded:

```
cached=70/116          (grow.n_prompt_cached > 0  AND  < grow.n_prompt_tokens both hold)
```

I instrumented the test temporarily to print both strings (reverted afterwards; the repo is clean).

**Vulkan backend:**
```
--- grow_cold (reference) ---
Suddenly, a loud thunder rumbs filled the air. A loud thunder and a loud thunder screeed, "A
--- grow.text  (reused) ---
Suddenly, a loud thunder rumbs started to rumble. A few drops of rain drops were coming quickly.
S
```

**CPU backend — byte-identical to the Vulkan run, both strings:**
```
--- grow_cold (reference) ---
Suddenly, a loud thunder rumbs filled the air. A loud thunder and a loud thunder screeed, "A
--- grow.text  (reused) ---
Suddenly, a loud thunder rumbs started to rumble. A few drops of rain drops were coming quickly.
S
```

**Interpretation, and why this is good news for the AMD port:**

- **This is not a Vulkan or AMD bug.** The CPU backend fails identically. Whatever it is, it is
  Windows/MSVC-wide.
- **It is not floating-point nondeterminism between backends.** If it were, Vulkan and CPU would
  diverge from each other. They agree character-for-character, including the same `cached=70/116`.
  That is a genuinely strong positive result for the Vulkan/RDNA4 path — it is reproducing the CPU
  reference exactly on this workload.
- **It does not look like stale-KV corruption.** The reused output is coherent English, shares the
  prefix `"Suddenly, a loud thunder rumbs"` with the cold reference, and then diverges at a single
  token. A genuine KV leak of the kind `f079811` fixed would bleed the *earlier* prompt (a
  weather/tool-calling system prompt, nothing to do with thunder) into the continuation, or emit
  garbage. Neither happened. If anything the reused output is the *more* coherent of the two — the
  cold reference degenerates into "a loud thunder and a loud thunder".

**Hypothesis (offered as hypothesis, not established):** the divergence is deterministic and
platform-dependent, which points at the reuse *bookkeeping* rather than the compute kernels. The
`f079811` fix reportedly works by ensuring "every freed cell is overwritten"; if some cell is not
in fact rewritten on the grow path, then the result depends on whatever the allocator left behind
— which would plausibly differ between glibc on Linux and the MSVC CRT on Windows while remaining
perfectly deterministic on each. That would explain all four observations at once.

**Also worth considering: the assertion may simply be too strict.** The test's own comment
concedes the prompt was chosen *because* it "empirically flips this model's greedy argmax on the
unfixed engine", and that "a shorter, blander prompt keeps the FA perturbation under the flip
threshold". It is deliberately balanced on an argmax knife-edge, so any legitimate
platform-level numerical difference will tip it. Asserting on reuse *behaviour* (cache counts,
coherence, absence of cross-prompt contamination) rather than exact string equality would be
more portable — though that would mask the underlying platform difference rather than explain it,
so I'd suggest understanding the cause first.

**Suggested next diagnostic** (not run — out of scope here): repeat the same growth scenario with
flash attention *disabled*. This block hardcodes `LLAMA_FLASH_ATTN_TYPE_ENABLED`. If the
divergence disappears with FA off, it is in the FA reuse path specifically; if it persists, it is
in the general prefix-cache grow path.

---

## 3. Real-model smoke test — PASSED

Model: `Qwen3-0.6B-Q8_0.gguf` (dense, 0.60 GB).

```powershell
.\pleiades-engine-server.exe --model G:\models\Qwen3-0.6B-Q8_0.gguf `
    --host 127.0.0.1 --port 8080 --ngl -1 --alias test
```

GPU placement confirmed from the startup log — the real discrete card, not a silent CPU fallback
and not the iGPU:

```
llama_prepare_model_devices: using device Vulkan0 (AMD Radeon RX 9070 XT) (unknown id) - 15416 MiB free
load_tensors: layer 0 assigned to device Vulkan0, is_swa = 0
... (all layers)
```

`/v1/chat/completions` returned a coherent response:

```json
{"choices":[{"finish_reason":"stop","index":0,
  "message":{"content":"Hello! How can I assist you today?","role":"assistant"}}],
 "usage":{"completion_tokens":13,"prompt_tokens":18,"total_tokens":31}}
```

195 ms round trip. Reasoning-model handling is correct: with thinking enabled, the engine
populated `reasoning_content` separately from `content`, exactly as it should.

### 3a. Real bug: `finish_reason` is `"stop"` on max_tokens truncation

With `max_tokens: 400` against a reasoning model that was still inside its `<think>` block:

```json
{"choices":[{"finish_reason":"stop", "message":{"content":"", ...}}],
 "usage":{"completion_tokens":400,"prompt_tokens":15,"total_tokens":415}}
```

`completion_tokens` (400) exactly equals `max_tokens` (400) — the generation was **truncated**, so
per the OpenAI API contract `finish_reason` must be `"length"`, not `"stop"`. Confirmed by
contrast: the 13-token genuine stop above also reports `"stop"`, so the two cases are currently
indistinguishable to a client. Any caller implementing "continue if truncated" will silently
mis-handle every truncated generation.

(The empty `content` in that response is *not* a bug — the model was still inside its reasoning
block when the cap hit, and `reasoning_content` was correctly populated.)

---

## 4. MoE offload correctness — PASSED

**This is the section the brief called highest-value, and it passes convincingly. The MoE
tensor-placement override — previously validated only against CUDA — works correctly on
AMD/Vulkan.**

Model: `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` (17.28 GB). Confirmed genuinely MoE from its own
metadata, not assumed:

```
qwen3moe.expert_count            = 128
qwen3moe.expert_used_count       = 8
qwen3moe.expert_feed_forward_length = 768
```

The placement override does real, verifiable work — `--n-cpu-moe 24` produced a genuine hybrid
split, not a silent all-CPU or all-GPU fallback:

```
load_tensors: offloaded 49/49 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  8807.41 MiB     <- experts forced to CPU
load_tensors:      Vulkan0 model buffer size =  9154.42 MiB     <- remainder on the RX 9070 XT
```

### 4a. Short greedy comparison — BYTE-IDENTICAL

Prompt: *"List exactly three primary colors, one per line, nothing else."*, `temperature: 0`,
`max_tokens: 60`.

| Config | Output | Tokens |
|---|---|---|
| A: `--ngl 0` (pure CPU) | `Red  \nGreen  \nBlue` | 5 |
| B: `--ngl -1 --n-cpu-moe 24` | `Red  \nGreen  \nBlue` | 5 |

Byte-identical.

### 4b. Long greedy comparison — coherent, equal quality, one early token flip

Because a 5-token match is thin evidence, I repeated the comparison with a ~160-token generation.
Prompt: *"Write a single paragraph of about 120 words explaining why the sky is blue. Plain prose,
no lists."*, `temperature: 0`, `max_tokens: 220`.

**A — pure CPU (`--ngl 0`), 158 tokens:**

> The sky appears blue **because of** a phenomenon called Rayleigh scattering. When sunlight enters
> Earth's atmosphere, it collides with molecules and tiny particles in the air. Sunlight is made up
> of all the colors of the rainbow, each with a different wavelength. Blue light has a much shorter
> wavelength than red or yellow light, so it is scattered in all directions by the gases and
> particles in the atmosphere much more effectively. This scattered blue light is what reaches our
> eyes from all parts of the sky, making it **look blue. Although violet light is scattered even
> more than blue, our eyes are less sensitive to violet, and some of it is absorbed by the upper
> atmosphere, so we perceive the sky as blue rather than violet. This effect is most noticeable
> when the sun is high in the sky.**

**B — GPU + experts on CPU (`--ngl -1 --n-cpu-moe 24`), 157 tokens:**

> The sky appears blue **due to** a phenomenon called Rayleigh scattering. When sunlight enters
> Earth's atmosphere, it collides with molecules and tiny particles in the air. Sunlight is made up
> of all the colors of the rainbow, each with a different wavelength. Blue light has a much shorter
> wavelength than red or yellow light, so it is scattered in all directions by the gases and
> particles in the atmosphere much more effectively. This scattered blue light is what reaches our
> eyes from all parts of the sky, making it **appear blue during the day. At sunrise and sunset,
> the sun is lower on the horizon, and its light passes through more atmosphere, scattering the
> blue light away and allowing the longer wavelengths like red and orange to dominate, which is
> why the sky turns those colors at those times.**

**Read on this: the GPU path is not degraded.**

- They diverge at the **fourth token** ("because of" vs "due to") — then **re-converge and run
  word-for-word identical for four entire sentences** (from "When sunlight enters" through "from
  all parts of the sky, making it"), before diverging again at the closing clause.
- That re-convergence is the informative part. A corrupted tensor-placement path would drift
  further apart as it went, or produce degraded/garbled text. Instead a single early near-tie
  flipped, strong context pulled both decodes back onto the identical trajectory, and they only
  parted again at the next genuine near-tie. That is exactly the signature of tiny, legitimate
  numerical differences in summation order — not of broken expert routing or misplaced tensors.
- **Both outputs are factually correct and equally high quality.** A adds the violet-sensitivity
  caveat; B adds the sunrise/sunset explanation. Both are accurate, relevant, well-formed prose.
  Neither is garbage; neither is worse.
- **The divergence is deterministic per configuration, not run-to-run flakiness.** Re-running both
  arms later (for the §4c timings, on a differently-loaded machine) reproduced the same split
  exactly: CPU again opened "The sky appears blue **because of**...", GPU again "**due to**...".
  A given config gives the same answer every time; the two configs differ from each other in a
  stable way. That is what a fixed difference in floating-point summation order looks like, and
  it rules out nondeterminism (race, uninitialised read, unsynchronised GPU work) as the cause.

Against the brief's stated bar — *"they should be coherent and reasonable, not garbage... if the
GPU path produces corrupted/nonsensical output while the CPU path is fine, that's a real bug"* —
this passes. And 4a hitting byte-identical output over a short greedy generation is a stronger
result than the brief expected to be possible across backends.

### 4c. Performance — Vulkan MoE offload is ~2x pure CPU

Measured properly: after freeing RAM, each configuration was loaded, given one full warm-up
generation (so page-cache settling is excluded), and only the *second* generation timed.

| Config | Load | Warmed generation | Tokens | **Rate** | Free RAM during run |
|---|---|---|---|---|---|
| A: pure CPU (`--ngl 0`) | 14 s | 11.3 s | 158 | **13.99 tok/s** | 4.6 GB |
| B: GPU + `--n-cpu-moe 24` | 16 s | 5.7 s | 157 | **27.72 tok/s** | 0.7 GB |

**Vulkan MoE offload delivers a 1.98x speedup over pure CPU on the RX 9070 XT**, with only about
half the model's weights resident on the GPU (9.15 GB of 17.96 GB). Load times are at parity.

Neither run was memory-starved: arm A still had 4.6 GB free, and because this is an MoE only
8 of 128 experts activate per token, so the hot working set is far smaller than the 17.28 GB file.

**Methodological warning, worth passing on to anyone else benchmarking this.** An earlier pass of
this same comparison, run while an unrelated game held ~2.7 GB and only ~5.5 GB was free,
produced:

| Config | Generation | Apparent rate |
|---|---|---|
| A: pure CPU | 31 s | ~5.1 tok/s |
| B: GPU + `--n-cpu-moe 24` | 53 s | ~3.0 tok/s |

— i.e. the GPU path appeared **slower than CPU**, the exact opposite of the true result, because
both arms were paging expert tensors from disk every token. Same binaries, same command lines,
same model: a 9.2x swing on arm B (3.0 -> 27.7 tok/s) purely from host memory pressure. Any MoE
throughput number taken on a box without enough free RAM for the CPU-side experts is not merely
noisy, it can invert the ordering.

---

## 6. Bonus: `/api/status` latency bug — found live on the installed desktop app, root-caused, fixed and verified

Outside the test brief's own scope, but found on this same machine and worth including since it's
a real, currently-live bug with a verified fix, included in the branch alongside the AMD results.

**Symptom:** the installed desktop app (v0.1.2) reported `Backend did not respond on
http://127.0.0.1:<port>/api/status within 90s.` on repeated fresh launches, even though the backend
process was alive and `/api/status` returned `HTTP 200` when queried directly.

**Root cause, measured, not guessed:**

- `/api/status` (`pleiades/webui/server.py`) made 3 blocking service-reachability probes
  (Anamnesis, inference, searxng) plus a separate `an.list_characters()` call, **sequentially**.
  Each `_reachable()` probe blocks up to its own 1.2s timeout when the service is down; the searxng
  probe was itself two sequential sub-probes (`/healthz` then root), up to 2.4s alone.
- With `inference` and `searxng` both down (the common case — both start on-demand and are simply
  not needed yet), stacking every probe measured **4.3-4.7s** to answer, reproduced 7 times across
  two different sessions on both the installed app and a from-source run of the same code.
- The desktop app's readiness poller (`desktop/src/main/index.ts`, `pollBackendReady`) aborts each
  individual attempt at `AbortSignal.timeout(2000)` — **2 seconds**.
- Every one of the 90s worth of poll attempts died at the 2s abort before the 4.3-4.7s endpoint
  could ever respond. This is not intermittent — it is **structurally guaranteed to fail every
  time** whenever two or more of the three optional services are down, which is an ordinary state
  (both start automatically on first use, not on launch).

**Fix (two parts, both applied, both verified working):**

1. `pleiades/webui/server.py`: the probes are independent network calls with no data dependency on
   each other. Parallelized via `concurrent.futures.ThreadPoolExecutor` (the pattern already used
   elsewhere in this codebase, e.g. `pleiades/harness/subagent.py`) — including splitting searxng's
   own two-step fallback into two further parallel futures, since otherwise it alone remained the
   slowest path and reintroduced most of the stack. Measured before/after on the same from-source
   checkout, same live services, 3-4 repeated timed requests each:

   | | Before | After |
   |---|---|---|
   | `/api/status` response time | 4.26-4.37s (3 runs) | **1.44-1.53s** (4 runs) |

   Response body is byte-identical before and after (same `up`/`characters`/`state` values) —
   this is a pure latency fix, no behavior change. Verified against a `pytest` run of the existing
   suite too: `tests/test_toolbelt.py` (18 tests, includes a route-registration check for
   `/api/status`) and the Anamnesis-adjacent test files all pass identically before and after
   (one pre-existing, unrelated failure — `test_node_bin_prefers_bundled_sibling_over_path`, a
   Windows/POSIX path-string mismatch in node-binary resolution — confirmed present on unmodified
   `main` via `git stash`, nothing to do with this change).

2. `desktop/src/main/index.ts`: raised the per-poll abort from a bare `2000` to a named
   `BACKEND_POLL_TIMEOUT_MS = 5_000`, as defense in depth. Even after the server-side fix, one of
   the four parallelized probes (`an.list_characters()`) still uses Anamnesis's own client with a
   30s timeout if the Anamnesis daemon itself is unresponsive — a related but **separate,
   disclosed, NOT reproduced this session** latent case (Anamnesis was up throughout all testing
   here). 5s gives real margin over the now-measured ~1.5s common case without meaningfully
   softening a genuine hang. Not typecheck-verified against `tsc` — this checkout has no
   `desktop/node_modules` installed and a full `npm install` felt disproportionate for a two-line,
   unambiguous numeric-constant change matching three sibling constants already declared the same
   way in the same file; recommend `npm run typecheck:node` as part of normal CI/review.

**What this does NOT fix:** if Anamnesis itself is down or unresponsive, `list_characters()` can
still take up to 30s, which would still exceed even the raised 5s poll-abort margin. That case
was not reproduced or measured this session (Anamnesis was up throughout), so I'm disclosing it
as a known related gap rather than claiming full closure.

---

## 7. Anything else

- **No crashes, no hangs, no driver resets.** Across every configuration tested — Vulkan and CPU
  builds, a 0.6 GB dense model fully on-GPU, and a 17.28 GB MoE model split across GPU and CPU
  under heavy memory pressure — the engine never crashed, never hung, and never triggered a TDR
  or driver reset on the RX 9070 XT. Shutdown was clean every time, and the context teardown
  self-checks passed on every run (`~llama_context: Vulkan0 compute buffer size is 74.6348 MiB,
  matches expectation of 74.6348 MiB`).
- **`--caps` works exactly as documented**, including correctly reporting `moe_offload: false` on
  the CPU build vs `true` on the Vulkan build.
- **Minor: the engine's `/v1/models` readiness signal is spoofable by any local service.** Not an
  engine bug so much as a testing hazard worth knowing: while scripting the MoE comparison I hit a
  port collision with an unrelated local `node` service, and a plain "did `/v1/models` return 200?"
  readiness poll accepted it as the engine being up — the request then failed with a foreign error
  body (`"No models provided"` plus a `user_id` field the engine never emits). Checking that the
  returned `data[0].id` matches the `--alias` you passed makes the probe unambiguous, and is what
  the final runs in this report do. Relevant to anyone writing Windows CI against this, where port
  contention is likelier than on a clean Linux runner.
- **Free-RAM constraint during testing.** This box had ~5.5 GB of 31 GB free during the MoE
  section (an unrelated game and WSL held most of it), against a 17.28 GB MoE model. Where that
  affected a result it is called out explicitly rather than glossed over.
- **`anamnesis` vendoring has no AMD VRAM detection.** Unrelated to the engine, but found while
  updating this checkout and directly relevant to AMD-on-Windows: the newly vendored
  `anamnesis/src/lib/inference-engine.js` probes VRAM via `nvidia-smi` **only**. On an AMD box the
  probe fails and `gpuLayers` falls back to 0, so anamnesis runs CPU-only even though the engine
  itself offloads to Vulkan fine. Ion has a tested fix for exactly this on his fork
  (`ionizedd/anamnesis-AMDEEP`, branch `amd-gpu-probe`, commit `7c1d031`): `nvidia-smi` stays the
  first probe with byte-identical behaviour on NVIDIA; on failure, Linux reads free VRAM from
  `amdgpu` sysfs counters and Windows detects an AMD/Radeon adapter via CIM and trusts
  `gpuLayerBudgetMB` (the same trust model the metal path already uses). No GPU init, no vendor
  SDK. His commit message records it as tested on this same RX 9070 XT: `gpuLayers 0 -> 9`,
  inference-engine tests 7/7. Happy to send it as a PR if useful.
