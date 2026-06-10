# Pleiades

A self-contained **inference workstation** for local LLM agents. Pleiades runs the
model itself, gives it persistent memory (via [Anamnesis](https://github.com/Fleabag515/anamnesis)),
and equips it as a full operator: local web search, a real headed browser it can
drive alongside you, its own email inbox, its own password vault, and optional
Discord hosting — all keyed to a single **character** identity. Set things up once
per character; never reconfigure a tool per service.

## The core idea

**A Pleiades profile *is* an Anamnesis character.** One name → one isolated memory
store, one email account, one password vault, one browser profile, one Discord
token. Spin up a character and every tool binds to it automatically. Run several at
once; nothing is shared or hand-wired between them.

## Architecture at a glance

```
you ─▶ Pleiades engine ─▶ Anamnesis character proxy ─▶ Pleiades inference server
        (agent loop,         (memory injection &          (in-process llama.cpp,
         tool calls)          retrieval, auto-stores       serves OpenAI-compatible
                              every turn)                   endpoint over any GGUF)
        │
        └─ tools: search · email · browser · vault   (all namespaced per character)
```

- **Inference is ours.** Pleiades loads a GGUF model with `llama-cpp-python` and
  serves an OpenAI-compatible endpoint in-process — no separate Ollama or server to
  install. Any GGUF that llama.cpp supports works, including large quantized models
  with GPU offload.
- **Memory is Anamnesis's job.** Anamnesis sits between the engine and our inference
  server, injecting relevant memory and storing every turn automatically. Pleiades
  never re-implements memory or re-sends long histories.

## Install

### One-line install (recommended)

Installs everything — repo, virtualenv, the in-process inference engine (with
**auto-detected GPU** build), Anamnesis, the headed browser, a generated `.env`,
and SearXNG — in one command.

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
accordingly (falling back to CPU), and detects Docker for SearXNG (guiding you if
it's missing). Options (pass after the pipe on Linux, e.g.
`... | bash -s -- --gpu --dir ~/apps/Pleiades`; as parameters on Windows):

| Option (sh / ps1)            | Effect                                            |
|------------------------------|---------------------------------------------------|
| `--dir DIR` / `-Dir`         | Install location (default `~/Pleiades`)           |
| `--branch N` / `-Branch`     | Git branch (default `main`)                        |
| `--gpu` `--cpu` / `-Gpu`     | Force GPU or CPU build (default: auto-detect)      |
| `--core` / `-Core`           | Core only — skip browser, SearXNG, Discord        |
| `--no-browser` / `-NoBrowser`| Skip Camoufox                                      |
| `--no-searxng` / `-NoSearxng`| Skip SearXNG                                       |
| `--no-discord` / `-NoDiscord`| Skip the Discord extra                            |

### Manual install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[all]"          # core + browser + discord
# core only:           pip install -e .
# with dev/test tools: pip install -e ".[all,dev]"
```

### GPU inference (optional but recommended for large models)

`llama-cpp-python` ships CPU-only by default. To build with CUDA/Metal/etc., set the
appropriate flag before install, e.g.:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install -e ".[all]"   # NVIDIA
CMAKE_ARGS="-DGGML_METAL=on" pip install -e ".[all]"  # Apple Silicon
```

### One-time external setup

```bash
npm install -g anamnesis && anamnesis status   # memory proxy daemon on :9000
camoufox fetch                                  # downloads the headed browser (only if using [browser])
```

## Quickstart

```bash
cp .env.example .env            # set PLEIADES_MODEL_PATH to your .gguf, fill secrets
pleiades new alice              # creates the Anamnesis character + Pleiades profile (prompts for email creds, etc.)
pleiades search up              # bring SearXNG online via docker compose
pleiades chat alice             # talk to the character (memory + tools wired in)
pleiades work alice ...         # NEW: have the character DO machine work (see below)
pleiades discord alice          # host it as a Discord bot
pleiades vault alice list       # manage the character's secrets
```

## What each character gets

| Capability | Tool | Notes |
|---|---|---|
| Persistent memory | (Anamnesis) | Auto-injected & auto-stored; not a Pleiades tool. |
| Web search | `search` | Local SearXNG JSON API. |
| Email | `email` | IMAP/SMTP from its own inbox; reads verification codes, sends mail. |
| Browser | `browser` | Headed Camoufox, persistent per-character profile. |
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
pleiades update                  # pull the latest from GitHub + reinstall
```

GPU: the build backend (NVIDIA/CUDA, AMD/ROCm, or CPU) is auto-detected by the
installer; at runtime use `--gpu-layers` (`-1` = all layers on GPU). Models run as
independent background servers, so different characters can use different models at
once. Pre-existing Anamnesis characters are never overwritten — adopt them to reuse
their memory.

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

MIT © Fleabag515. Memory layer: [Anamnesis](https://github.com/Fleabag515/anamnesis).
Project conventions and the full build brief live in `CLAUDE.md`.
