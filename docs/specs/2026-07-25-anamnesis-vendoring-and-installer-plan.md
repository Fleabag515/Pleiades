# Vendoring Anamnesis + bundling tools into Pleiades — architectural plan

*Written 2026-07-25. Design doc, not yet executed — see task list for phase status.*

## TL;DR recommendation (for the phase-1 go/no-go)

**Go.** This is the right call and the codebase is already 80% shaped for it — the `runtime.py` "detect platform → fetch matching prebuilt → verify" pattern and the PyInstaller+`extraResources` bundling story are exactly reusable for a Node payload. Concretely:

1. **Vendor the fork as a first-class monorepo component at repo root `anamnesis/`** via `git subtree` (preserves the 70-commit history and blame; future changes become normal Pleiades commits). **Archive** `Fleabag515/anamnesis` on GitHub read-only. There is nothing to unpublish on npm — the npm global is *empty* and the installer's `npm i -g anamnesis` step is dead/wrong code pointing at the unrelated upstream package (see finding below).
2. **Replace systemd supervision with a Pleiades-managed daemon lifecycle** (a new `pleiades/anamnesis_runtime.py` + supervisor mirroring `models.py`'s `ModelManager`), spawned by the shared webui/`server.py` process so both the CLI and the desktop app's bundled backend bring it up automatically.
3. **Ship the Node payload the same way the Python backend already ships** — a per-platform `node_modules` (pruned to the matching native prebuilts) + a fetched Node runtime binary, delivered via a *second* `extraResources` entry. Do **not** use `pkg`/`nexe`/SEA — they can't embed the `.node` native addons this project depends on.
4. **De-Docker SearXNG** by freezing it like the backend and supervising it as a subprocess; drop the Docker gate entirely.
5. **One pushback:** don't literally bundle the ~986 MB Qwen helper GGUF into every platform installer (that's ~3-4 GB of installers). Keep it a first-run SHA-verified fetch (the code already does this) with an optional installer pre-warm.

### Phase map (front-loaded)

| Phase | What | Depends on | Risk | Ships behavior change? |
|---|---|---|---|---|
| 0. Vendor-in + stop the bleeding | `git subtree add --prefix=anamnesis`; gitignore its `node_modules`; delete the broken `npm i -g anamnesis` installer step | — | Low | No (daemon still runs via systemd) |
| 1. Pleiades-owned supervision | `anamnesis_runtime.py` + `AnamnesisDaemon` supervisor; `server.py` ensures-running; `pleiades anamnesis {start,stop,status,adopt}`; migration off systemd | 0 | Med | Yes (process owner changes) |
| 2. Cross-platform Node packaging | Build-time per-platform `node_modules` + Node runtime fetch; 2nd `extraResources`; bundle the 23 MB embedding model | 1 | Highest (native ABI x OS x arch x GPU) | Yes (fresh installs need nothing) |
| 3. De-Docker SearXNG | Freeze/subprocess SearXNG; replace `pleiades search up` + `docker-compose.yml`; per-install secret | independent of 1/2 | High (uWSGI->Flask subprocess) | Yes |
| 4. Remaining tools + installer unification | Camoufox/Playwright browser fetches folded into one installer pass; drop Docker checks; collapse `install.sh` flags | 2, 3 | Low-Med | Yes |
| 5. Retire the separate project | Archive GitHub repo; migration GA; docs | 1-4 | Low | No |

Phases 3 and 4-partial can run in parallel with 1/2. **Phase 2 and Phase 3 are the two genuinely hard, multi-session pieces** — everything else is plumbing.

---

## The load-bearing finding that justifies all of this

The current setup is **internally inconsistent**, which is the real problem to fix:

- The **actually-running** memory proxy is a hand-maintained **git checkout** at `~/.local/share/anamnesis` (fork, remote `Fleabag515/anamnesis.git`, v0.6.0, 70 commits), launched by a **hand-created systemd unit** (`/etc/systemd/system/anamnesis.service`, `Nice=10 CPUWeight=25`, active) that the Pleiades installer **does not create**.
- The Pleiades installer's `install_anamnesis()` runs **`npm install -g anamnesis`** — which pulls the *unrelated upstream npm package*, not the fork. Confirmed the npm global prefix is **empty** and `anamnesis` on `PATH` is a `~/.local/bin/anamnesis` shell wrapper written by the *fork's own* `install.sh`, not npm.
- `pleiades/anamnesis.py` is a **pure HTTP client** to `127.0.0.1:9000` — it never installs, builds, or supervises anything.

So today, a "fresh" Pleiades install would install the *wrong* Anamnesis, and the working one exists only because it was set up out-of-band on this machine. Vendoring doesn't just satisfy the request — it fixes a latent "fresh install is broken" bug.

---

## Part 1 — Anamnesis vendoring

### 1a. Repo layout

**Proposed: a new top-level `anamnesis/` directory** (sibling to `pleiades/`, `desktop/`, `engine/`), containing the fork's source (`src/`, `package.json`, `package-lock.json`, `config.json`, importers, tests), with `anamnesis/node_modules/` gitignored.

Rejected alternatives:
- **`third_party/`** — that dir is for genuinely-upstream code Pleiades *tracks* (holds the `llama.cpp` submodule). Anamnesis is now *ours*; filing it there sends the wrong signal.
- **`pleiades/anamnesis-daemon/`** — pollutes the hatchling wheel (`packages = ["pleiades"]` would sweep a Node tree into the Python package) and mixes a Node project inside a Python import package.
- **`services/anamnesis/`** — defensible (SearXNG config already lives at `services/searxng/settings.yml`), but `services/` holds *config* today, not *code*. Fallback if preferred.

**Git strategy: `git subtree add --prefix=anamnesis https://github.com/Fleabag515/anamnesis main` (no `--squash`).**
- vs. submodule: a submodule *is* the separate-repo-with-its-own-update-cycle being killed. Reject.
- vs. clean copy-in: loses all 70 commits of history/blame on an actively-developed component. Subtree preserves history inside the Pleiades commit graph — future edits are ordinary Pleiades commits.

### 1b. Native deps, cross-platform (the hard part)

The fork's `node_modules` is **1.6 GB** because of **four** native chains, all shipping per-platform prebuilts:

| Native chain | Role | Prebuilt mechanism | Size note |
|---|---|---|---|
| `node-llama-cpp` ^3.18.1 | helper LLM (extraction/foresight/consolidation) | `@node-llama-cpp/<platform>` optional-dep packages | 703 MB for the 6 Linux variants alone |
| `@huggingface/transformers` ^3.8.1 -> `onnxruntime-node` | sentence embeddings (all-MiniLM-L6-v2) | per-OS prebuilts | linux tree is 404 MB (ships CUDA EP not needed for CPU-only MiniLM) |
| `better-sqlite3` ^9.6.0 | memory store | `prebuild-install \|\| node-gyp rebuild` | ~26 MB |
| `@img/sharp` + `@reflink` | image/io | per-platform prebuilts | small |

**Recommendation — build-time, per-platform, pruned `node_modules`:**

1. Per build target: `npm ci --omit=dev` with npm's `--os/--cpu/--libc` filters, then prune foreign prebuilts (drop non-target `@node-llama-cpp/*`, drop onnxruntime's non-target OS dirs, drop onnxruntime's CUDA execution provider since the embedder is CPU-only — reclaims ~350 MB).
2. `better-sqlite3`: ship the `prebuild-install` prebuilt for the target.
3. Ship a fetched official Node runtime binary per platform (same mechanism as `runtime.py.install()` fetching llama-server). Pin one LTS (e.g. 22.x). ~40-60 MB compressed.

Ballpark per-platform payload after pruning: **~250-400 MB** — same weight class as the PyInstaller backend already shipping.

**Do NOT use `pkg`/`nexe`/Node SEA.** All three native chains dlopen real `.node` files at runtime. These tools can't embed native addons cleanly — externalize anyway, gain nothing.

**Electron ABI caveat:** Electron 39 uses a different `NODE_MODULE_VERSION` than stock Node; native addons built for stock Node won't load inside Electron's process. **Keep the daemon a separate stock-Node subprocess** — never `require` these addons inside Electron's main process. Sidesteps `electron-builder install-app-deps` rebuild pain (bundle already sets `npmRebuild: false`).

### 1c. Launch / supervision going forward

**Replace systemd with a Pleiades-owned supervisor**, mirroring `pleiades/models.py`'s `ModelManager`:

- New `pleiades/anamnesis_runtime.py` (sibling to `runtime.py`): resolves the Node binary + daemon path (bundled `<resources>/anamnesis/` in the desktop build, repo `anamnesis/` in a dev/CLI install — same dev-vs-packaged fork `desktop/src/main/index.ts:bundledBackendPath()` already does for the backend).
- New `AnamnesisDaemon` supervisor mirroring `ModelManager.start()/stop()/_wait_ready()`: `subprocess.Popen([node, "src/daemon.js"], cwd=<anamnesis>, start_new_session=True)`, track pid in `PLEIADES_HOME/anamnesis-running.json` (like `models-running.json`), health-poll `GET http://127.0.0.1:9000/status`, stop via `killpg`. **Preserve the systemd `Nice=10` intent** via `preexec_fn=lambda: os.nice(10)` on POSIX — that comment documents a real 15->1 tok/s regression when memory chores ran at full priority.
- **Spawned by `pleiades/webui/server.py`** at startup. Because the desktop app's PyInstaller backend *is* that server, the desktop app gets daemon lifecycle for free; the CLI path gets it through the same code. Electron's `index.ts` stays unchanged.

Data stays put: `~/.anamnesis/` is hardcoded off `os.homedir()` in `daemon.js`/`char-config.js`/`registry.js`/`model-manager.js`. Never bundle/overwrite it. Control API (`:9000`), per-character proxy ports, `registry.json` layout unchanged — **`pleiades/anamnesis.py` needs zero changes**, zero data migration.

### 1d. Migration path for existing installs (like Minty right now)

New command `pleiades anamnesis adopt`:
1. Stop + disable + remove `/etc/systemd/system/anamnesis.service`.
2. Remove the `~/.local/bin/anamnesis` wrapper (written by the fork's own `install.sh`) — or repoint it at the vendored CLI.
3. Leave `~/.anamnesis/` untouched.
4. Hand ownership to the new supervisor; verify `GET /status` on `:9000` now comes from the vendored copy.
- `--keep-systemd` flag for anyone who deliberately wants the unit.

### 1e. Fate of `Fleabag515/anamnesis`

Archive read-only on GitHub, tag final `v0.6.0`, point README at Pleiades. No npm action needed (fork was never the published `anamnesis`). Updates now ride Pleiades releases.

---

## Part 2 — SearXNG + "all other tools" bundling

Inventory of "separately fetched" things (from `install.sh` flags + `pyproject.toml` extras):

| Thing | Today | Native/binary? | Recommendation |
|---|---|---|---|
| Anamnesis | systemd + git checkout + broken `npm i -g` | Node native x4 | Vendor + supervise (Part 1) |
| SearXNG (`--no-searxng`) | `docker compose up -d searxng`, Docker-gated | Python/uWSGI app | De-Docker: freeze + subprocess |
| Native llama-server (`--no-native-runtime`) | `pleiades runtime install` fetches from `ggml-org` releases | prebuilt binary | Keep as-is, or fold fetch into unified installer pass. Already gold-standard, low priority |
| Camoufox browser (`--no-browser`) | `python -m camoufox fetch` | browser binary | Python pkg already bundled via `collect_all`; fold the binary fetch into installer's one pass, or pre-seed the cache |
| Playwright Chromium | Playwright downloads Chromium | browser binary | Same treatment as Camoufox |
| Discord (`--no-discord`) | `discord.py` extra | pure Python | Already bundled via `collect_all`. Nothing to do |
| workspace/docs extras | anthropic, pypdf, python-docx, openpyxl, Pillow | pure Python | Already handled by `collect_all`. Nothing to do |

### SearXNG, de-Dockered

SearXNG is a Python Flask app (uWSGI is a deployment choice upstream, not a requirement).

**Recommended:** freeze it like the backend — dedicated PyInstaller onedir bundle (or second frozen venv), ship via `extraResources` next to `backend/`, supervise as a subprocess exactly like the Anamnesis daemon. Run under a small embedded server (werkzeug/waitress) on `127.0.0.1:8888`, load the existing `services/searxng/settings.yml`, generate a **per-install `SEARXNG_SECRET`** at first run instead of the `change-me-please` placeholder in `docker-compose.yml`. `pleiades/tools/search.py`/`harness/builtins/web.py` already resolve the endpoint from `Settings.searxng_url` — no client changes.

**Alternative (only if freezing SearXNG's deps fights back):** run under the same frozen backend interpreter as an in-process thread — simpler deployment, but couples SearXNG's heavy deps and failure modes into the backend process.

Then: delete `docker-compose.yml` dependency, rewrite `pleiades search up/down` (`cli.py:545`) to drive the supervised subprocess, remove `check_docker()`/`start_searxng()`'s Docker gate from `install.sh`.

### Model weights (the "downloads nothing" nuance)

Anamnesis pulls two models at first daemon run, not install time:
- **all-MiniLM-L6-v2** ONNX embedder (~23 MB) — bundle it, pre-seed the transformers.js cache / set `allowRemoteModels=false`.
- **Qwen2.5-1.5B-Instruct-Q4_K_M.gguf** (~986 MB, SHA256-verified, resumable) — do not bundle (see pushback). Keep first-run fetch, optional pre-warm during install.

The user's main chat GGUF is inherently user-chosen and multi-GB — out of scope by definition.

---

## Part 3 — Honest pushback / things not worth doing as literally asked

1. **Don't bundle the ~986 MB helper GGUF into every installer.** 3-4 platform installers = 3-4 GB of redundant download for a model already fetched once, resumably, with SHA verification, into persistent storage. Bundling quadruples installer size for zero reliability gain. Keep it a first-run fetch (offer `--prefetch-models` + a desktop first-run progress UI). Honors the *spirit* of "no separate manual step" without the bloat. Bundle only the tiny fixed 23 MB embedder.

2. **Don't ship all 6 `@node-llama-cpp` GPU variants (703 MB).** A 1.5B background-extraction model doesn't need CUDA+Vulkan+every-arch prebuilts on a given machine. Ship CPU + at most one GPU backend matching the detected vendor (reuse `hardware.detect()`/`_backend_priority()`). Prune onnxruntime's CUDA EP too (~350 MB).

3. **Don't reach for `pkg`/`nexe`/SEA** — incompatible with the `.node` native addons.

4. **Don't rebuild the native modules against Electron's ABI.** Spawn a stock-Node subprocess; keep `npmRebuild: false`.

5. **Out of scope but worth flagging:** Anamnesis embeds its *own* `node-llama-cpp` llama.cpp stack purely for the 1.5B helper — a *third* llama.cpp copy alongside Pleiades' `llama-cpp-python` and the native `llama-server`/`third_party/llama.cpp` submodule. Consolidating (pointing Anamnesis's helper at Pleiades' existing runtime over HTTP) would cut hundreds of MB, but it's a real behavioral change to the fork and should be a *later* project, not smuggled into this one.

6. **Don't touch `~/.anamnesis/` in any bundling step.** Holds `history.db`, `registry.json`, `characters/`, `daemon.json`, downloaded models — must survive reinstalls.

---

## Appendix — key facts verified on Minty

- **Fork:** `~/.local/share/anamnesis`, remote `Fleabag515/anamnesis.git`, v0.6.0, `main=src/proxy.js`, `bin.anamnesis=src/cli.js`; 37 files/328 KB under `src/`; 70 commits on `main` (+ `fix/topic-drift`), tag `v0.6.0`; `engines.node >=20`; running under Node v22.22.2.
- **Native deps:** `better-sqlite3 ^9.6.0`, `node-llama-cpp ^3.18.1` (Linux prebuilts: arm64/armv7l/x64/x64-cuda/x64-cuda-ext/x64-vulkan = 703 MB), `@huggingface/transformers ^3.8.1` -> `onnxruntime-node` (linux tree 404 MB) + `@img/sharp` + `@reflink`; `prompts ^2.4.2` pure; optional `node-windows`. Total `node_modules` = 1.6 GB.
- **Data dir** `~/.anamnesis` hardcoded off `os.homedir()` in `daemon.js`/`char-config.js`/`registry.js`/`model-manager.js`.
- **Runtime model fetches:** Qwen2.5-1.5B-Q4_K_M.gguf (986 MB, SHA256-verified, resumable) -> `~/.anamnesis/models/`; all-MiniLM-L6-v2 ONNX (~23 MB) via transformers.js.
- **Supervision today:** `/etc/systemd/system/anamnesis.service` (`Nice=10`, `CPUWeight=25`, `Restart=on-failure`) — active. `~/.local/bin/anamnesis` is a wrapper written by the fork's own `install.sh`; **npm global is empty**, so the Pleiades installer's `npm i -g anamnesis` is dead/wrong code.
- **Pleiades side:** `pleiades/anamnesis.py` = HTTP client only (`:9000`); `config.py` `Settings.anamnesis_control_url`, `searxng_url` default `127.0.0.1:8888`; `PLEIADES_HOME=~/.pleiades`.
- **Reusable precedents:** `runtime.py` (`pick_asset()`/`_backend_priority()`/`install()`); `models.py` `ModelManager` (`Popen(start_new_session=True)`, `models-running.json`, `_wait_ready()`, `killpg`); `desktop/backend-build/backend.spec`/`build.sh`/`entry.py`; `electron-builder.yml` (`extraResources`, `npmRebuild: false`); `desktop/src/main/index.ts:bundledBackendPath()`/`spawnBackend()`.
- **SearXNG today:** `docker-compose.yml` (`searxng/searxng:latest`, `127.0.0.1:8888->8080`, mounts `services/searxng/settings.yml`, secret `change-me-please`); `pleiades search up/down` (`cli.py:545`) shells `docker compose`; `install.sh` `check_docker()`/`start_searxng()` gate it on Docker.
