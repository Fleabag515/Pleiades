# Pleiades — harness integration (ionizedd merge)

Pleiades now has two halves, unified into one project:

1. **Identity + inference engine** (the original repo): a profile *is* an Anamnesis
   character with its own encrypted vault, email inbox, headed browser, and Discord
   token, talking to our **in-process llama.cpp** OpenAI-compatible server.
2. **The agent harness** (`pleiades/harness/`, from ionizedd's parallel build): a
   Claude-Code-style agent loop with a 60+ tool belt (files, git, shell, process,
   code-quality, documents, events, system, subagents), tool-search for large
   catalogs, and an MCP client.

A character can now *operate* — read/write files, run and test code, drive git,
browse, react to webhooks — while keeping its identity and memory.

## Decisions (owner)

- **Core idea wins, plus the harness.** The repo's character/identity model is the
  base; his tool belt makes the models also work Claude-Code-style on machine tasks.
- **Anamnesis is the fundamental context manager.** The external Anamnesis daemon
  (`pleiades.anamnesis`) stays canonical. His in-process working-memory + semantic
  recall (`pleiades.harness.anamnesis`) is folded in as an *augmenting* tier, scoped
  per-character.
- **This is an inference engine.** Local llama.cpp is the primary backend (default
  tier). Cloud/API and standalone Ollama are optional tiers so the workspace *can*
  call hosted models, but local inference through the project is the point.
- **Identity stays.** Per-character vault, email, browser, Discord.

## What landed (phased, on `main`)

| Phase | What | Status |
|---|---|---|
| A | Vendor harness into `pleiades/harness/` (zero logic changes, no collisions) | done |
| B | Default tier routes to our llama.cpp engine; cloud/Ollama optional | done |
| C | `identity.bind_character()` — run the harness as a character (workspace, vault, email, proxy routing) | done |
| D | Anamnesis stays canonical; in-process memory tier scoped per-character | done |
| E | `pleiades work` CLI, docs, optional extras, tests | done |

Run it:

```bash
pleiades work "find the largest .py file and summarize it"     # plain workspace
pleiades work --as alice "check my inbox for a verification code and save it to the vault"
pleiades work --tier coder --policy allow "add a test for vault.delete and run it"
```

## What's next (open)

- **Unify config**: the harness `Config` and the repo `Settings` still co-exist;
  collapse into one source of truth (env + config.json + tiers).
- **Dedupe overlapping tools**: harness `web_search`/`browser_*` vs the repo's
  SearXNG/Camoufox tools — keep one, character-scoped.
- **Native layer** (his roadmap): LSP bridge, sandboxed executor, streaming tools,
  multi-agent fabric. See `GOLDEN_BASELINE.md` (from his build) for the vision.

*Co-owners: Fleabag515 + ionizedd.*
