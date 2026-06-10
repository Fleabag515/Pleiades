# Pleiades Control Panel (`pleiades.webui`)

A polished, local web UI for managing Pleiades characters, models, and settings.
It wraps the existing managers (`ProfileManager`, `ModelManager`, `Vault`,
`Anamnesis`, `Settings`) behind a small FastAPI backend and serves a single-page
frontend. Because it runs in the browser, it behaves **identically on Linux and
Windows** — no native GUI toolkit, no per-platform packaging.

Everything it shows is your own local state under `~/.pleiades`. The server binds
to `127.0.0.1` by default; nothing leaves the machine.

## What it covers

| Area | What you can do |
|---|---|
| **Overview** | Live status of Anamnesis, the inference engine, and SearXNG; counts of characters/models; running model servers. |
| **Characters** | Create characters, adopt existing Anamnesis characters, delete, and drill into a per-character detail view with tabs. |
| **· Email** | Provider presets (Gmail / mail.com / Outlook / generic), IMAP/SMTP host+port, and an app password written **encrypted** to the vault (`email.password`). |
| **· Credentials** | View every vault entry (metadata only by default), **reveal** a value on demand, add/edit/delete custom `site:<domain>` logins and reserved keys. |
| **· Discord** | Toggle hosting, store the bot token encrypted (`discord.token`), with setup guidance (Message Content intent). |
| **· Model** | Select which registered model a character uses (per-profile), or fall back to the default engine. |
| **Models** | Register a GGUF, edit context window / GPU layers / chat format, start & stop the per-model server, and remove models. |
| **Settings** | Default inference engine (path, host/port, ctx, GPU layers, chat format), Anamnesis + SearXNG URLs, agent-harness tier/policy/steps, master-key status, and resolved paths. |

Security model is preserved end-to-end: vault values never appear in list
responses or logs — they're only returned by the explicit single-key reveal
endpoint when you click **Reveal**.

## Install

Add a `ui` extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
ui = ["fastapi>=0.110", "uvicorn>=0.29"]
# and add fastapi + uvicorn to the `all` list if you want them by default
```

Then:

```bash
pip install -e ".[ui]"      # or: pip install fastapi uvicorn
```

Wire the CLI command: paste the block in `cli_command.py` into `pleiades/cli.py`
(the `pleiades ui` command is built in). The `webui/` package directory drops straight
into `pleiades/`.

> Hatchling note: since the build targets `packages = ["pleiades"]`, the
> `webui/static/*` assets are included automatically. If a slim build ever omits
> them, add:
> ```toml
> [tool.hatch.build.targets.wheel.force-include]
> "pleiades/webui/static" = "pleiades/webui/static"
> ```

## Run

```bash
pleiades ui                 # auto-pick a free port and open the browser
pleiades ui --port 8800     # fixed port
pleiades ui --no-browser    # headless / remote (pair with --host 0.0.0.0)

# or without the CLI wiring:
python -m pleiades.webui
```

## Layout

```
pleiades/webui/
├── __init__.py        # create_app / run exports
├── __main__.py        # python -m pleiades.webui
├── server.py          # FastAPI app: REST over the managers + static host
├── README.md          # this file
└── static/
    ├── index.html     # app shell
    ├── styles.css     # deep-space design system
    └── app.js         # SPA: views, modals, API client
```

## API surface (all under `/api`)

```
GET    /status                          dashboard: services + counts + running models
GET    /settings        PUT /settings   global settings (writes .env)
GET    /profiles        POST /profiles  list / create characters
GET    /profiles/{n}    PUT  /profiles/{n}   DELETE /profiles/{n}
POST   /profiles/{n}/email              email config + encrypted password
POST   /profiles/{n}/discord            token + enable flag
POST   /profiles/{n}/model              assign a registered model
GET    /profiles/{n}/vault              entry metadata (never values)
POST   /profiles/{n}/vault              add/update an entry
GET    /profiles/{n}/vault/{key}        reveal one value (explicit)
DELETE /profiles/{n}/vault/{key}        delete an entry
POST   /adopt                           adopt an orphan Anamnesis character
GET    /models          POST /models    list / register
PUT    /models/{n}      DELETE /models/{n}
POST   /models/{n}/start  POST /models/{n}/stop
GET    /email/presets                   provider presets
```
