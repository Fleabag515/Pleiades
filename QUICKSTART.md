# Pleiades — Quickstart

Pleiades is a **command-line tool**, not a GUI app. After install you get a
`pleiades` command. Everything below assumes you've activated the project's venv:

```bash
cd ~/Pleiades
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pleiades --help
```

(Prefer not to activate each time? Call `~/Pleiades/.venv/bin/pleiades` directly.)

## 1. Add a model

Pleiades runs models itself (local llama.cpp). Register any `.gguf` file by name:

```bash
pleiades model add qwen  ~/models/qwen2.5-7b-instruct-q4.gguf  --gpu-layers -1
pleiades model add big    ~/models/llama-3.3-70b-q4.gguf       --gpu-layers -1 --ctx 16384
pleiades model list
```

- `--gpu-layers -1` offloads the whole model to the GPU; `0` is CPU-only; a number
  offloads that many layers (useful to split a big model across limited VRAM).
- `--ctx` sets the context window; `--chat-format` overrides the chat template if a
  model needs it (e.g. `chatml`, `llama-3`).

### NVIDIA and AMD

Both work — the GPU backend is chosen **at install time** and the installer
auto-detects it:

- **NVIDIA** → CUDA build (needs the CUDA toolkit for the GPU build).
- **AMD** → ROCm build.
- No GPU → CPU build.

At runtime you just use `--gpu-layers`. To switch backends later, reinstall with the
flag, e.g. `CMAKE_ARGS="-DGGML_CUDA=on" pip install -e ".[all]"` (NVIDIA) or
`-DGGML_HIPBLAS=on` (AMD).

## 2. Start / stop a model

Each model runs as its own background server on its own port:

```bash
pleiades model start qwen      # launches it (waits until ready)
pleiades model running         # what's live right now
pleiades model stop qwen       # shut it down
```

You usually don't *need* to start manually — a character auto-starts its assigned
model the first time you chat. Logs are in `~/.pleiades/logs/model-<name>.log`.

## 3. Make a character (or adopt an existing one)

A **character** is one identity: its own memory (Anamnesis), email, password vault,
browser profile, and Discord token.

```bash
pleiades new alice             # creates a new character (prompts for email, optional)
```

Already had **Anamnesis** characters before installing Pleiades? Keep and reuse them
(their memory is preserved):

```bash
pleiades list                  # shows adoptable Anamnesis characters at the bottom
pleiades adopt Mark            # wrap an existing character as a Pleiades profile
```

## 4. Choose which model a character uses

```bash
pleiades model use alice qwen      # alice now thinks with 'qwen'
pleiades model use Mark  big       # Mark uses the 70B
```

Different characters can use different models at the same time (each model is its own
server). The assignment is remembered; the model auto-starts when that character chats.

## 5. Talk to it / put it to work

```bash
pleiades chat alice                                   # conversational REPL
pleiades work --as alice "summarize this repo and run the tests"   # agent does machine work
pleiades discord alice                                # run it as a Discord bot
```

## 6. Web search (optional)

```bash
pleiades search up        # start local SearXNG (docker)
pleiades search down
```

## 7. Update Pleiades

```bash
pleiades update           # git pull latest from GitHub + reinstall
```

(Re-running the one-line installer also updates an existing install.)

## How the pieces fit

```
you ─▶ pleiades (CLI) ─▶ <character>'s Anamnesis proxy ─▶ the model server (llama.cpp)
                          (injects memory, stores turns)   (the model you assigned)
```

You never wire Anamnesis by hand: `pleiades new`/`adopt` registers the character with
the Anamnesis daemon, and `model use` points that character's proxy at the model you
chose. Memory is automatic.
