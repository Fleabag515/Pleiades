# Anamnesis

_From Greek ἀνάμνησις — the deep recollection of what the mind already knows._

A self-organizing memory proxy for LLM agents. Anamnesis sits between any OpenAI-compatible client and any OpenAI-compatible backend, giving the model persistent, structured, intelligently-retrieved memory across unlimited context and sessions. Run multiple characters, each with their own isolated memories, all managed from a single CLI.

## Install

**Linux / macOS:**

```bash
curl -fsSL https://raw.githubusercontent.com/Fleabag515/anamnesis/main/install.sh | bash
```

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/Fleabag515/anamnesis/main/install.ps1 | iex
```

**npm (any platform):**

```bash
npm install -g anamnesis
```

> If `anamnesis` isn't found after install, open a new terminal or add `~/.local/bin` to your PATH.

## Quick Start

```bash
# Create your first character (interactive wizard)
anamnesis new

# Start it
anamnesis start mycharacter

# Point your OpenAI-compatible client at the port shown
# e.g. http://127.0.0.1:8084/v1
```

That's it. Anamnesis runs as a background daemon and starts automatically when you use any CLI command.

## CLI Reference

```
anamnesis new                   Create a new character (interactive wizard)
anamnesis list                  List all characters and their status
anamnesis start <name>          Start a character's proxy
anamnesis stop <name>           Stop a character's proxy
anamnesis restart <name>        Restart a character's proxy
anamnesis status                Show daemon status and active characters
anamnesis show <name>           Show character config
anamnesis edit <name>           Edit character settings (interactive)
anamnesis remove <name>         Delete a character and its data
anamnesis import <name> <file>  Import memories from a file
anamnesis logs [name]           Tail logs
anamnesis install               Register as a system service (auto-start on boot)
anamnesis uninstall             Remove the system service
anamnesis update                Check for updates and install if available
anamnesis --help                Show this list
```

## Architecture

```
anamnesis daemon (port 9000 — control API)
    │
    ├── character "alice"  → proxy on :8084 → upstream LLM
    ├── character "bob"    → proxy on :8085 → upstream LLM
    └── character "..."    → proxy on :...  → upstream LLM
```

Each character is a fully independent proxy with its own:

- `~/.anamnesis/characters/<name>/config.json` — settings
- `~/.anamnesis/characters/<name>/history.db` — SQLite memory store

A registry at `~/.anamnesis/registry.json` tracks all characters and their ports.

### Memory pipeline

```
Turn received
    │
    ├─→ Store raw turn + embedding (sync, crash-safe)
    │
    └─→ [background] Engram extraction
              │
              ├─→ Foresight scan in parallel
              │
              └─→ [periodic] Episode consolidation
                      │
                      └─→ Decay scoring + pruning
```

### Retrieval (per request)

```
Query embedding
    │
    ├─→ Score all Episodes by cosine similarity
    │       └─→ Expand top scenes → constituent turn IDs
    │               └─→ Rank by similarity + importance, fill token budget
    │
    └─→ Fallback: raw turn similarity (no scenes yet)

Context injected:
  [ system + <memory> + <foresight> ] + [ rotating relevant turns ] + [ last N turns verbatim ]
```

### Why not just a sliding window?

|                       | Sliding window      | Anamnesis                               |
| --------------------- | ------------------- | --------------------------------------- |
| Old turns             | Dropped permanently | Stored forever, retrieved when relevant |
| Retrieval             | Recency only        | Scene-guided cosine similarity          |
| Memory structure      | Flat                | Hierarchical (turn → cell → scene)      |
| Forgetting            | Hard cutoff         | Soft decay by age + recall frequency    |
| Background processing | None                | Engram + Foresight extraction           |
| Streaming             | Native              | Native SSE pass-through                 |

## Importing Memories

The `import` command accepts files in several formats — Anamnesis will auto-detect the format and use an LLM to extract structured memories:

```bash
anamnesis import mycharacter /path/to/chatlog.txt
anamnesis import mycharacter /path/to/export.json
```

Supported formats: plain text or markdown files (character sheets, memory logs, SOUL.md / persona files from OpenClaw, Hermes, or similar agents), SillyTavern exports, Character.AI exports, Odysseus exports, prior Anamnesis exports, and similar structured or unstructured sources. If the format isn't directly recognised, an LLM pass extracts whatever memories it can find.

## Character Config

Each character's `config.json` controls its behaviour. Key settings:

| Key                              | Default                  | Description                                                                   |
| -------------------------------- | ------------------------ | ----------------------------------------------------------------------------- |
| `proxy.port`                     | auto-assigned            | Port this character listens on                                                |
| `upstream.baseUrl`               | auto-detected            | Any OpenAI-compatible endpoint                                                |
| `upstream.apiKey`                | _(blank)_                | Bearer token sent upstream. Empty = pass client's own `Authorization` through |
| `upstream.disableThinking`       | `true`                   | Suppress thinking tokens for Qwen3-style models                               |
| `embedding.model`                | `nomic-embed-cpu:latest` | Ollama embedding model                                                        |
| `extraction.model`               | `qwen3:0.6b`             | Small LLM for engram extraction                                               |
| `foresight.model`                | `qwen3:0.6b`             | Small LLM for intention extraction                                            |
| `context.tokenBudget`            | `50000`                  | Total token budget                                                            |
| `context.recencyTurns`           | `8`                      | Recent turns always included verbatim                                         |
| `context.rotatingSlots`          | `6`                      | Additional turns added via scene retrieval                                    |
| `persona.source.type`            | `auto`                   | `auto` \| `file` \| `inline` \| `disabled`                                    |
| `memory.consolidationIntervalMs` | `120000`                 | How often scenes are rebuilt                                                  |
| `memory.sceneClusterThreshold`   | `0.72`                   | Cosine similarity threshold for clustering                                    |
| `memory.decayPruneThreshold`     | `0.05`                   | Score below which cells are pruned                                            |
| `history.maxAgeDays`             | `90`                     | Raw turn retention period                                                     |

### Persona

If the character has a persona file (e.g. a SOUL.md or character definition), set:

```json
"persona": {
  "source": {
    "type": "file",
    "file": { "path": "~/.config/mycharacter/persona.md" }
  }
}
```

Or paste inline:

```json
"persona": {
  "source": {
    "type": "inline",
    "inline": { "content": "You are ..." }
  }
}
```

### Environment

| Var             | Values                              | Notes          |
| --------------- | ----------------------------------- | -------------- |
| `ANAMNESIS_LOG` | `error` / `warn` / `info` / `debug` | Default `info` |

## Migrating from v0.4

If you had Anamnesis v0.4 installed (single proxy, `~/.anamnesis/history.db`), the daemon auto-migrates on first run: your existing history is imported as a character named `default` and your old systemd unit is disabled.

## Status API

The control server exposes a REST API on `127.0.0.1:9000`:

```
GET  /status                    Daemon uptime and active character count
GET  /characters                List all characters
GET  /characters/:name          Show a single character
POST /characters                Create a character
POST /characters/:name/start    Start a character
POST /characters/:name/stop     Stop a character
DELETE /characters/:name        Delete a character
```

## Development

```bash
npm test         # run test suite (node:test)
npm run lint     # ESLint
npm run format   # Prettier --write
```

CI runs on every push against Node 18, 20, and 22.

## Roadmap

- [x] Multi-character daemon with isolated memory per character
- [x] Interactive character creation wizard
- [x] Universal memory import (text, SillyTavern, Character.AI, Odysseus)
- [x] One-liner installers (Linux, Windows, npm)
- [x] v0.4 → v0.5 auto-migration
- [ ] `anamnesis export` — export character memories to a portable file
- [ ] Web UI for browsing the memory graph
- [ ] Cross-session scene merging

## License

[MIT](./LICENSE)
