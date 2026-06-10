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

## Follow-ups done

- **Config unified.** `pleiades.config.Settings` is now the single source of truth
  (defaults + config.json + env + tiers + `tier()`); `pleiades/harness/config.py`
  is a shim re-exporting it. `openai_host` derives from the local inference engine,
  so engine and harness agree on one endpoint.
- **Web/browser deduped.** Harness `web_search`/`deep_research` read the unified
  `searxng_url`; the repo's chat-path `SearchTool` now delegates to the harness
  `web_search` (one implementation). The harness browser and the chat-path browser
  share ONE persistent profile per character (`bind_browser_profile` ->
  `profile.browser_dir`).

## What's next (open)

- **One agent loop.** The chat path (`engine.py`, class-based `ToolBelt`) and the
  work path (`harness/agent.py`, the `@tool` registry) still have separate loops.
  Merging chat into the harness loop would leave a single loop + single tool surface
  (the browser would then have one driver instead of two sharing a profile).
- **Native layer** (his roadmap): LSP bridge, sandboxed executor, streaming tools,
  multi-agent fabric. See `GOLDEN_BASELINE.md` (from his build) for the vision.

*Co-owners: Fleabag515 + ionizedd.*
