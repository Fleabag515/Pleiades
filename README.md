# Pleiades

A self-contained **inference workstation** for local LLM agents — as a native
**desktop app** or a **CLI + local web control panel**. Pleiades runs the model
itself, gives it persistent memory (Anamnesis, vendored directly into this repo),
and equips it as a full operator: local web search, a real headed browser it can
drive alongside you, its own email inbox, its own password vault, and optional
Discord hosting — all keyed to a single **character** identity. Set things up once
per character; never reconfigure a tool per service.

## The core idea

**A Pleiades profile *is* an Anamnesis character.** One name → one isolated memory
store, one email account, one password vault, one browser profile, one Discord
token. Spin up a character and every tool binds to it automatically. Run several at
once; nothing is shared or hand-wired between them.

## Two ways in

- **Desktop app** (recommended, installed automatically on Linux) — a native window:
  a bar of character bubbles, one live chat, a right-side panel that streams a live
  view of whatever the agent's browser tool is doing so you can watch and step in,
  and a Settings modal for characters, hardware, models, and updates. Details and
  build instructions below under [Desktop app](#desktop-app).
- **CLI + web control panel** — `pleiades <command>` for scripting, and
  `pleiades ui` for the same control panel the desktop app itself runs, in a browser
  tab instead of a native window.

Both drive the same local backend, so nothing gets out of sync between them — use
whichever fits the moment.

## Architecture at a glance

```
you ─▶ Pleiades engine ─▶ Anamnesis character proxy ─▶ Pleiades inference server
        (agent loop,         (memory injection &          (in-process llama.cpp —
         tool calls)          retrieval, auto-stores        native runtime, or the
                              every turn)                    bundled fallback server)
        │
        └─ tools: search · email · browser · vault   (all namespaced per character)

desktop app  ─┐
              ├─▶ same local backend (pleiades/webui/server.py)
 pleiades ui ─┘
```

- **Inference is ours.** Pleiades loads a GGUF model and serves an OpenAI-compatible
  endpoint in-process — no separate Ollama or server to install. `pleiades runtime
  install` fetches a native `llama-server` build for this machine (MoE expert
  offload, elastic context resize); without it, Pleiades automatically falls back to
  the bundled `llama-cpp-python` server. Any GGUF that llama.cpp supports works,
  including large quantized models with GPU offload.
- **Memory is Anamnesis's job**, and Anamnesis now lives *inside* this repo
  (`anamnesis/`) rather than being installed as a separate project — see
  [Memory (Anamnesis)](#memory-anamnesis). It sits between the engine and our
  inference server, injecting relevant memory and storing every turn automatically.
  Pleiades never re-implements memory or re-sends long histories.

## Install

### One-line install (recommended)

Installs everything — repo, virtualenv, the in-process inference engine (with
**auto-detected GPU** build), the native `llama-server` runtime, Anamnesis
(vendored, auto-supervised), the headed browser, a generated `.env`, SearXNG, and —
on Linux — the **desktop app**, registered in your system's application menu.

**Linux / macOS**

```bash
curl -fsSL https://raw.githubusercontent.com/Fleabag515/Pleiades/main/install.sh | bash
```

**Windows (PowerShell)**

```powershell
irm https://raw.githubusercontent.com/Fleabag515/Pleiades/main/install.ps1 | iex
```

The installer auto-installs Python and Node where it can (apt/dnf/pacman/zypper,
Homebrew, or winget), detects an NVIDIA/Apple GPU and builds `llama-cpp-python`
accordingly (falling back to CPU), fetches the native `llama-server` runtime, and
detects Docker for SearXNG (guiding you if it's missing). Options (pass after the
pipe on Linux, e.g. `... | bash -s -- --gpu --dir ~/apps/Pleiades`; as parameters on
Windows):

| Option (sh / ps1)              | Effect                                                      |
|---------------------------------|-------------------------------------------------------------|
| `--dir DIR` / `-Dir`            | Install location (default `~/Pleiades`)                     |
| `--branch N` / `-Branch`        | Git branch (default `main`)                                  |
| `--gpu` `--cpu` / `-Gpu`        | Force GPU or CPU build (default: auto-detect)                |
| `--core` / `-Core`              | Core only — skip browser, SearXNG, Discord, native runtime   |
| `--no-browser` / `-NoBrowser`   | Skip Camoufox                                                |
| `--no-searxng` / `-NoSearxng`   | Skip SearXNG                                                 |
| `--no-discord` / `-NoDiscord`   | Skip the Discord extra                                       |
| `--no-native-runtime`           | Skip the native `llama-server` runtime install               |
| `--no-desktop-app`              | Skip building/installing the desktop app (Linux)             |

### Manual install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[all]"          # core + browser + discord + web control panel
# core only:           pip install -e .
# with dev/test tools: pip install -e ".[all,dev]"
```

A manual/pip install skips what the one-line installer automates — run these once
yourself afterward:

```bash
pleiades runtime install                 # native llama-server (MoE offload, faster)
python -m camoufox fetch                 # headed browser (only if using [browser])
cd anamnesis && npm install --omit=dev   # vendored Anamnesis's own deps (needs Node)
```

Anamnesis itself starts on its own the first time `pleiades ui` or the desktop app
boots — see [Memory (Anamnesis)](#memory-anamnesis).

### GPU inference

The one-line installer detects your GPU and builds the right backend automatically
(NVIDIA CUDA, AMD ROCm — or Vulkan when ROCm isn't installed — and Apple Metal).
At runtime, GPU offload is **automatic too**: `n_gpu_layers` defaults to `auto`,
which reads each GGUF's real layer count, measures free VRAM/RAM, and plans the
best GPU/CPU split at launch (`pleiades hw` shows the plan and the reasoning).
Set an explicit number (`-1` all layers, `0` CPU) to override.

For manual installs, set the backend flag before pip:

```bash
CMAKE_ARGS="-DGGML_CUDA=on"   pip install -e ".[all]"  # NVIDIA
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install -e ".[all]" # AMD ROCm
CMAKE_ARGS="-DGGML_METAL=on"  pip install -e ".[all]"  # Apple Silicon
```

### Native inference runtime

`pleiades runtime install` fetches an official prebuilt `llama-server` binary
(ggml-org/llama.cpp releases) into `~/.pleiades/runtime/` — faster than the bundled
Python server, and the only path that supports MoE expert offload (splitting a
mixture-of-experts model between GPU and CPU residency) and elastic context
resizing without a restart. The one-line installer runs this automatically; check
what's active any time with `pleiades runtime status`. Without it, models still
run — just through the bundled `llama-cpp-python` server (dense GPU/CPU layer split
only, no MoE offload).

### Memory (Anamnesis)

Anamnesis — the memory proxy every character's context runs through — used to be a
separate project (`npm install -g anamnesis`, its own hand-set-up systemd service,
its own release cycle, and a real bug where that global npm package wasn't even
this fork). It's now vendored directly into this repo at `anamnesis/`: one
checkout, one `git pull` / `pleiades update` for source changes, no separate repo to
track. Pleiades supervises the daemon itself instead of a hand-written systemd unit:

```bash
pleiades anamnesis status      # is it running, on which port
pleiades anamnesis start       # start it (also happens automatically on first
pleiades anamnesis stop        #   `pleiades ui` / desktop app launch)
```

If a future update changes Anamnesis's own Node dependencies, re-run
`npm install --omit=dev` inside `anamnesis/` — `pleiades update` refreshes the
Python side today, not that step yet. If you're picking up a machine that still has
the old hand-set-up systemd service from before vendoring, `pleiades anamnesis
adopt` stops and disables that unit and hands the already-running daemon over to
Pleiades' own supervision without losing any character memory.

## Desktop app

A native window over the same local backend `pleiades ui` serves in a browser: a
bar of character bubbles across the top, one live chat below, a right-side panel
that streams a live view of whatever the agent's browser tool is doing (headed
Camoufox, or the Tor-routed variant) so you can watch and step in, and a Settings
modal with **Characters** (per-character email/Discord/model/vault config),
**Hardware** (this machine's detected GPU/RAM/CPU and the per-model offload plan),
**Models** (browse/download/register GGUFs), and **Updates**.

Installed automatically by the one-line installer on Linux (`.deb` + `.AppImage`,
shows up in your application menu). To build it yourself:

```bash
cd desktop
npm install
npm run dist:linux    # .deb + AppImage
npm run dist:win      # NSIS installer — unsigned, so SmartScreen will warn on first run
```

macOS isn't wired up yet (no `dist:mac` script). Updates ship through GitHub
Releases: the app checks automatically shortly after launch and any time from
Settings → Updates, downloads in the background, and a single button installs and
restarts.

## Quickstart

```bash
cp .env.example .env            # set PLEIADES_MODEL_PATH to your .gguf, fill secrets
pleiades new alice              # creates the Anamnesis character + Pleiades profile (prompts for email creds, etc.)
pleiades search up               # bring SearXNG online via docker compose
pleiades chat alice               # talk to the character (memory + tools wired in) — or open the desktop app / `pleiades ui`
pleiades work alice ...          # have the character DO machine work (see below)
pleiades discord alice            # host it as a Discord bot
pleiades vault alice list         # manage the character's secrets
```

## What each character gets

| Capability | Tool | Notes |
|---|---|---|
| Persistent memory | (Anamnesis) | Auto-injected & auto-stored; not a Pleiades tool. |
| Web search | `search` | Local SearXNG JSON API. |
| Email | `email` | IMAP/SMTP from its own inbox; reads verification codes, sends mail. |
| Browser | `browser` | Headed Camoufox, persistent per-character profile; live in the desktop app's side panel. |
| Password vault | `vault` | Encrypted SQLite; secrets reach the model only on explicit `vault.get`. |
| Discord | (connector) | Host the character as a bot. |

## Models, characters & updates

New to Pleiades? See **[QUICKSTART.md](QUICKSTART.md)**. The essentials:

```bash
pleiades model add qwen ~/models/qwen.gguf --gpu-layers -1   # register a GGUF
pleiades model start qwen        # run it (or it auto-starts on chat)
pleiades new alice               # create a character
pleiades adopt Mark              # OR adopt a pre-existing Anamnesis character (keeps its memory)
pleiades model use alice qwen    # pick which model a character uses
pleiades chat alice              # talk to it
pleiades update                  # pull the latest source from GitHub + reinstall Python deps
```

GPU: the build backend (NVIDIA/CUDA, AMD/ROCm, or CPU) is auto-detected by the
installer; at runtime use `--gpu-layers` (`-1` = all layers on GPU). Models run as
independent background servers, so different characters can use different models at
once. Pre-existing Anamnesis characters are never overwritten — adopt them to reuse
their memory.

`pleiades update` covers the engine, CLI, harness, and vendored Anamnesis source —
whether or not you're running the desktop app. The desktop app additionally
self-updates its own app shell (see [Desktop app](#desktop-app)); the two are
independent, so a fresh `pleiades update` won't by itself change which desktop app
build you're running, and vice versa.

## Workspace harness (Claude-Code-style)

Beyond chatting, a character can *operate the machine*. The agent harness
(`pleiades.harness`) gives every character a 60+ tool belt — files, git, shell,
processes, code formatting/linting/tests, document readers, webhooks, subagents,
tool-search, and an MCP client — driven by an autonomous loop with a permission
gate (`ask` / `allow` / `deny`).

```bash
pleiades work "find the largest .py file and summarize it"          # plain workspace
pleiades work --as alice "read my inbox, save any login codes to my vault"
pleiades work --tier coder --policy allow "add a test for vault.delete and run it"
```

Local llama.cpp inference is the default brain; `--tier cloud` (needs the
`[workspace]` extra + an API key) or `--tier ollama` switch backends. With
`--as <character>` the agent runs inside that character's workspace, memory,
vault, and inbox, routing inference through its Anamnesis proxy.

The harness was contributed by **ionizedd**; see `INTEGRATION.md` for the merge
and `GOLDEN_BASELINE.md` for the native-layer roadmap (LSP bridge, sandboxed
executor, streaming tools, multi-agent fabric).

## Security & responsible use

Vault secrets are encrypted at rest, never logged, and never auto-injected into
prompts — a secret reaches the model only when it explicitly calls `vault.get`. The
email + anti-fingerprint browser + credential-store stack is built for managing
**your own** agent's accounts and services. Keep usage within the terms of service
of any site the agent touches; this is not a tool for mass/automated account
creation.

## License

MIT © Fleabag515. Memory layer: Anamnesis, vendored into this repo at `anamnesis/`
(originally developed at [Fleabag515/anamnesis](https://github.com/Fleabag515/anamnesis),
now maintained here). Project conventions and the full build brief live in
`CLAUDE.md`.
