# Pleiades — build brief (auto-loaded)

This file is the single source of truth: full vision, architecture, the interface of
the upstream dependency (Anamnesis), the module-by-module spec, access needed, secrets
the human must supply, and build order. Read it top to bottom before touching code.

> Decision log: the owner chose to **build Pleiades' own inference engine** rather than
> depend on a separately-installed backend (Ollama/llama.cpp server). We implement this
> as an **in-process `llama-cpp-python` server** that loads any GGUF and exposes an
> OpenAI-compatible endpoint — self-contained, supports large quantized models + GPU,
> and still presents the OpenAI-compatible upstream that Anamnesis requires. "From
> scratch" was scoped to *owning the engine layer*, not reimplementing llama.cpp's
> quant kernels/architectures (which would forfeit "any GGUF" and large-model support).
> Email defaults: provider-agnostic IMAP with app-password presets for Gmail / mail.com.

---

## 1. What Pleiades is

A self-contained inference workstation for local LLM agents. The human runs one or more
"characters" (models with persistent personality + memory) and each is a fully equipped
operator: it can search the web, drive a real browser alongside the user, read and send
email from its own inbox, store and retrieve its own passwords, and be hosted as a
Discord bot — all without the human reconfiguring a tool per service.

The unifying idea: **a Pleiades profile *is* an Anamnesis character.** One name → one
isolated memory store, one email account, one password vault, one browser profile, one
Discord token. Spin up a character and every tool binds to it automatically.

Feature list (owner's verbatim intent):

- Inference engine that acts as its own workstation, efficient, built from scratch
  (this repo) — **we run the model ourselves, in-process, no external backend service.**
- **SearXNG** built in for free local web search the model can use.
- **Headed Camoufox** so the model can control a real browser alongside the user.
- **Anamnesis** integration so the model's context *is* Anamnesis.
- **Discord connector** to natively host a character as a Discord bot.
- The model gets **its own email** (human creates account, hands over IMAP/SMTP; model
  has full read/send control — useful for receiving verification codes).
- A **built-in password manager** for the model itself.
- Anamnesis character profiles **are** the profiles for email, vault, browser, etc.
  Multiple characters run at once, each isolated; never touch an individual tool's config.

---

## 2. The hard architectural fact: Anamnesis

Anamnesis (https://github.com/Fleabag515/anamnesis) is a self-organizing memory proxy.
It sits between an OpenAI-compatible client and an OpenAI-compatible backend and gives
the model persistent, intelligently-retrieved memory. **Pleiades must not reimplement
memory, context management, or character identity — Anamnesis owns all of that.**
Pleiades is the agent runtime, the tool layer, and (now) the inference server underneath.

### How Anamnesis works (the parts Pleiades depends on)

- A **Node.js** app, `npm install -g anamnesis`. Integration is pure HTTP, so the
  Python/JS split is irrelevant.
- Runs as a **background daemon** with a **control REST API on `127.0.0.1:9000`**.
- Each **character** is an independent proxy on its own auto-assigned port (e.g. `:8084`).
  Its OpenAI-compatible endpoint is `http://127.0.0.1:<port>/v1`. Point an OpenAI client
  there and memory injection/retrieval is automatic.
- Per-character state at `~/.anamnesis/characters/<name>/` (`config.json` + `history.db`);
  registry at `~/.anamnesis/registry.json` maps names → ports.
- **Turns are stored automatically** (it's a proxy). Pleiades sends chat completions
  through the character's proxy and does **not** separately persist conversation memory.

### Control API (base `http://127.0.0.1:9000`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/status` | daemon uptime + active character count |
| GET | `/characters` | list all characters |
| GET | `/characters/:name` | one character (includes its port) |
| POST | `/characters` | create a character |
| POST | `/characters/:name/start` | start its proxy |
| POST | `/characters/:name/stop` | stop its proxy |
| DELETE | `/characters/:name` | delete character + data |

CLI mirrors this: `anamnesis new|list|start|stop|restart|status|show|edit|remove|import|logs|install|update`.

### Character `config.json` keys Pleiades cares about

- `proxy.port` — the port to talk to (auto-assigned).
- `upstream.baseUrl` — **Pleiades sets this to its own inference server** (e.g.
  `http://127.0.0.1:8080/v1`).
- `upstream.apiKey` — bearer sent upstream; blank = pass the client's `Authorization`.
- `upstream.disableThinking` — default `true`. Relevant when using thinking models for
  tool calling (if tool calls vanish, check this).
- `context.tokenBudget` (50000), `context.recencyTurns` (8), `context.rotatingSlots` (6).
- `persona.source` — `auto|file|inline|disabled`. May be surfaced at profile creation.
- `embedding.model`, `extraction.model`, `foresight.model` — small helper models. With
  our own inference server these can point at the same server (or a small GGUF); document
  what the human must provide.

**Optimize context with Anamnesis in mind:** keep system prompt + tool defs lean; send
only new user turn(s) + tool results + tool schemas. Anamnesis supplies history.

---

## 3. Stack decisions (made — don't relitigate without reason)

- **Language: Python ≥3.10.** Camoufox is Python-first, SearXNG ecosystem is Python,
  `llama-cpp-python` gives us in-process inference. Anamnesis is JS but HTTP-only.
- **Inference (ours):** `llama-cpp-python` with the `[server]` extra. Pleiades launches
  an in-process OpenAI-compatible server over a GGUF model and points Anamnesis
  `upstream.baseUrl` at it. Any GGUF works; large quantized models + GPU offload
  supported via the appropriate build flags. The engine layer is ours; llama.cpp is the
  swappable compute kernel. (`pleiades/inference/`.)
- **LLM client:** the `openai` package pointed at the character's Anamnesis proxy.
  Tool calling uses standard OpenAI function-calling format, passed through unchanged.
- **Web search:** **SearXNG** locally (Docker easiest), JSON API. The instance **must**
  have JSON enabled (`search.formats: [html, json]`) or it returns 403.
- **Browser:** **Camoufox**, **headed** (`headless=False`), one persistent context per
  character (`~/.pleiades/profiles/<name>/browser`). Requires `camoufox fetch` once.
- **Vault:** encrypted SQLite per profile, `cryptography` Fernet; key from
  `PLEIADES_MASTER_KEY` env, else generated `~/.pleiades/master.key` (0600). Secrets
  never logged, never sent to the model unless it calls the vault tool.
- **Email:** stdlib `imaplib` + `smtplib`. Per-profile creds in the vault. Default
  provider-agnostic IMAP; app-password presets for Gmail and mail.com.
- **Discord:** `discord.py`. **CLI:** `click` + `rich`.

Dependencies pinned in `pyproject.toml`. Browser/Discord are optional extras.

---

## 4. Repository layout + module spec

```
pleiades/
├── pyproject.toml                  # DONE
├── CLAUDE.md                       # this file
├── README.md                       # DONE
├── LICENSE                         # MIT (Fleabag515)
├── .gitignore                      # python + ~/.pleiades secrets never committed
├── .env.example                    # master key, model path, inference + searxng URLs
├── docker-compose.yml              # SearXNG service
├── install.sh                      # one-line installer (Linux/macOS): prereqs, GPU autodetect, full stack
├── install.ps1                     # one-line installer (Windows): winget prereqs, CUDA autodetect, full stack
├── services/searxng/settings.yml   # SearXNG config WITH json format enabled
└── pleiades/
    ├── __init__.py
    ├── config.py                   # paths + settings (platformdirs)
    ├── anamnesis.py                # control-API client + proxy URL resolver
    ├── vault.py                    # encrypted per-profile secret store
    ├── profiles.py                 # Profile model + manager (the unifying layer)
    ├── inference/__init__.py       # OUR engine: in-process llama.cpp OpenAI server
    ├── engine.py                   # the agent loop (inference call + tool calls)
    ├── cli.py                      # `pleiades` entrypoint
    ├── tools/
    │   ├── __init__.py             # Tool base + ToolBelt registry + dispatch
    │   ├── search.py               # SearXNG JSON search
    │   ├── email_box.py            # IMAP/SMTP: list_unread, read, send, search
    │   ├── browser.py              # Camoufox headed session
    │   └── vault_tool.py           # model-facing store/get/list credentials
    └── connectors/
        ├── __init__.py
        └── discord_bot.py          # host a character as a Discord bot
```

### Module contracts

**`config.py`** — `PLEIADES_HOME = ~/.pleiades`; `profiles_dir`, `master_key_path`.
`Settings` dataclass from env/.env: `anamnesis_control_url`, `searxng_url`,
`model_path`, `inference_host/port`, `n_ctx`, `n_gpu_layers`, `chat_format`,
`backend_base_url`/`backend_api_key` (override only). `profile_dir(name) -> Path` +
helpers for `vault.db` / `browser/` / `profile.json`.

**`anamnesis.py`** — thin `httpx` client. `Anamnesis: status(); list_characters();
get_character(name); create_character(name, **cfg); start(name); stop(name);
delete(name)`. `proxy_base_url(name) -> str`. `ensure_running(name)`.

**`vault.py`** — `Vault(path, key)`: `set/get/list/delete`. SQLite, Fernet at rest.
`load_key()`. Reserved keys (`email.password`, `discord.token`) vs model site creds
(`site:<domain>`).

**`profiles.py`** — `@dataclass Profile`: name, email_address, imap/smtp host+port,
browser_dir, discord_enabled, non-secret config in `profile.json`; secrets in the vault.
`ProfileManager`: `create/get/list/delete/open_vault`. Enforces "set up once per
character, never per tool."

**`inference/__init__.py`** — `InferenceServer(settings)`: launches `llama_cpp.server`
over `settings.model_path`, binds `inference_host:port`, OpenAI-compatible. `start()`,
`stop()`, `wait_ready()`, `base_url` (`http://host:port/v1`), `is_running()`.
`ensure_inference(settings) -> base_url` starts it if needed and returns the URL to wire
into Anamnesis `upstream.baseUrl`.

**`tools/__init__.py`** — `Tool: name; description; parameters(JSON schema); run(ctx,
**kwargs) -> str`. `@dataclass ToolContext`: profile, vault, settings + lazy shared
resources. `ToolBelt`: `openai_schema() -> list[dict]`; `dispatch(name, args, ctx)`.

**`tools/search.py`** — `GET {searxng_url}/search?q=…&format=json`; top-N
titles+urls+snippets as compact text.

**`tools/email_box.py`** — one tool, `action`: `list_unread|read|send|search`. Creds
from vault. `read`/`list_unread` return clean body text (verification codes).

**`tools/browser.py`** — `BrowserSession` over `camoufox` (`headless=False`, persistent
`user_data_dir`). Actions: `goto/click/type/read/screenshot`. Session alive across tool
calls in a turn. Needs `camoufox fetch`.

**`tools/vault_tool.py`** — model-facing over `Vault`: `store/get/list`. Only path by
which a secret reaches the LLM.

**`engine.py`** — loop: resolve profile → `ensure_inference(settings)` → set Anamnesis
`upstream.baseUrl` → `anamnesis.ensure_running(name)` → `openai` client at proxy URL →
build `ToolBelt` → send new message + tool schemas → dispatch any `tool_calls`, append
results, loop (hard cap) → return final message. Stream when possible.

**`connectors/discord_bot.py`** — `discord.py`; token from vault; on DM/mention call
`engine.run(profile, text)` and reply. One bot per character.

**`cli.py`** (click): `new`, `list`, `chat`, `discord`, `vault <set|get|list|rm>`,
`search up|down`, plus `serve` (bring the inference server up) for debugging.

---

## 5. Access Cowork/dev needs

Shell (git/pip/npm/docker/anamnesis/camoufox + long-lived processes); network (pip/npm
installs, `camoufox fetch`, SearXNG, Discord gateway, IMAP/SMTP — inference is local so
no LLM backend network needed); filesystem write to the repo, `~/.pleiades/`,
`~/.anamnesis/`; Docker (or pip SearXNG); ability to run daemons (Anamnesis, SearXNG,
the inference server, the Discord bot); GitHub (`gh repo create` — repo doesn't exist yet).

## 6. Secrets / setup the human must provide

- **`PLEIADES_MASTER_KEY`** — vault key; else generated `~/.pleiades/master.key` (0600).
- **`PLEIADES_MODEL_PATH`** — path to a GGUF model file Pleiades serves. (Replaces any
  external backend; no Ollama needed.) For Anamnesis helper models, either point them at
  this server or supply a small GGUF.
- **An email account per character** — address, IMAP/SMTP host+port, app password
  (Gmail/mail.com app-password presets included; generic IMAP for anything else).
- **A Discord bot token per character** (if hosting) — Developer Portal; enable Message
  Content intent.
- **Anamnesis installed** — `npm install -g anamnesis`; `anamnesis status`.

## 7. First-session checklist

> **Automated path:** end users install via the one-liners in README —
> `curl -fsSL .../install.sh | bash` (Linux/macOS) or `irm .../install.ps1 | iex`
> (Windows). They do hybrid prereq handling (auto-install Python/Node, guide for
> Docker), auto-detect GPU for the `llama-cpp-python` build, clone, venv, install
> `pleiades[all]`, install Anamnesis, `camoufox fetch`, generate `.env`, and start
> SearXNG. The manual steps below are the same sequence for development.

1. Confirm `pyproject.toml` + this file present.
2. `gh repo create Pleiades --private --source . --remote origin` (or add remote
   manually) → `git add -A && git commit -m "Initial scaffold" && git push -u origin main`.
3. `python -m venv .venv && . .venv/bin/activate && pip install -e ".[all]"` (add
   `CMAKE_ARGS=...` for GPU builds of llama-cpp-python).
4. `npm install -g anamnesis && anamnesis status`.
5. `camoufox fetch` (if using the browser).
6. Build order: `config.py` → `anamnesis.py` → `vault.py` → `profiles.py` →
   `inference/__init__.py` → `tools/__init__.py` → `tools/{search,email_box,vault_tool}` →
   `engine.py` → `cli.py` → `tools/browser.py` → `connectors/discord_bot.py` → SearXNG
   compose + settings → tests.
7. Smoke test the spine: start the inference server, create a character, send one message
   through the engine, confirm Anamnesis stored the turn and the reply comes back.

## 8. Gotchas & guardrails

- **Memory is Anamnesis's job.** Engine sends only new turns + tool results + schemas.
- **Inference is ours and local.** Set Anamnesis `upstream.baseUrl` to our server before
  starting the character. Don't reintroduce an external backend dependency.
- **GGUF coverage = llama.cpp coverage.** Any GGUF/quant llama.cpp supports works; GPU
  needs the right build flags for `llama-cpp-python`.
- **SearXNG JSON 403** → enable `json` in `settings.yml`.
- **Camoufox** is headed + persistent per character; needs `camoufox fetch`.
- **Tool calling** depends on the served model supporting function calling — verify with
  your chosen GGUF and set `chat_format` if needed.
- **Vault secrets** never logged, never auto-injected; reach the model only via `vault.get`.
- **Per-character isolation is the whole point** — every path/vault/browser/inbox/token
  is namespaced by character. No shared globals.
- **`disableThinking`** defaults true; if tool calls vanish with a thinking model, check it.
- **Responsible use:** email + anti-fingerprint browser + saved creds is powerful; it's
  for managing *your own* agent's accounts within services' ToS.

---

*Owner: Fleabag515. Upstream memory: github.com/Fleabag515/anamnesis. If you deviate from
this brief, note why in the commit and update this file.*
