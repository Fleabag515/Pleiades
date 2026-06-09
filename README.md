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
