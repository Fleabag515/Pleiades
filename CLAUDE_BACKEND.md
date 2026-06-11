# Pleiades — Claude (subscription) backend

> Status: design + phase-1 prototype landed. Co-owner work; see CLAUDE.md / INTEGRATION.md
> for the surrounding architecture.

This document explains how to run Pleiades' agent harness with **Claude itself as the
brain**, billed to your **Claude subscription** (Pro/Max) rather than per-token API
usage — and how that turns Pleiades into a place where Claude can stress-test and
improve its own tool belt from the inside.

---

## 1. The billing reality (read this first)

There are two completely different ways to "use Claude" from code, and they bill
differently:

| Path | How it bills | In this repo |
|---|---|---|
| **Anthropic Messages API** (`anthropic.Anthropic(api_key=...)`) | per-token, pay-as-you-go | the existing `cloud` / `cloud-fast` tiers in `harness/llm.py` (`backend="anthropic"`) |
| **Claude Agent SDK / Claude Code CLI**, authenticated via `claude login` | your **subscription** (Pro/Max), via the Agent SDK monthly **credit** | the new `claude` tier (`backend="claude-code"`) added here |

The thing you asked for — "use my membership, my usage, not per-token" — is **only**
available through the second path. The mechanism: the Agent SDK runs the Claude Code CLI
under the hood, and the CLI uses your local `claude login` credentials.

### Two hard constraints, stated plainly

1. **Timing.** Subscription plans get a monthly Agent SDK credit that explicitly covers
   "the Agent SDK, `claude -p`, and third-party apps built on the Agent SDK" — but that
   credit pool **starts June 15, 2026**. Before then, subscription-backed programmatic use
   is in flux; after then it's the supported, documented model. (Sources in the chat thread
   that produced this doc: support.claude.com Agent-SDK-with-your-plan + Claude-Code-with-Pro/Max.)
2. **Terms of Service.** As of Feb 2026, OAuth credentials are restricted to Claude Code
   and Claude.ai. The **supported** way to spend subscription usage is to let the Agent SDK
   use your `claude login` session. The **prohibited** way is to scrape the OAuth token out
   of `~/.claude/.credentials.json` and feed it to the raw Messages API yourself.
   → Practically: **never point the existing `backend="anthropic"` tier at an OAuth token.**
   That tier is the per-token API path and must stay that way. Subscription usage goes
   exclusively through `backend="claude-code"`.

Your machine today: Claude Code v2.1.91 installed, logged in (`subscriptionType: pro`),
**no `ANTHROPIC_API_KEY` set**. That's exactly the clean state this backend wants.

---

## 2. The architecture decision

You left the architecture choice to this doc. There were two shapes:

**(A) Pleiades drives, Claude is a one-turn brain.** Keep the harness `Agent` loop in
`agent.py`; add a `backend="claude-code"` branch to `LLM.chat()` that returns a single
assistant turn. **Rejected.** The Agent SDK is built to *own* the loop (planning, tool
dispatch, compaction, subagents). Bending it into "give me exactly one assistant turn with
tool_use blocks against my schemas" fights the grain of the SDK, throws away the part of
Claude Code that's actually good, and would be brittle across SDK versions.

**(B) Claude drives, Pleiades' tools are exposed to it. ✅ Chosen.** We publish the
harness tool belt to the SDK as an **in-process MCP server** (`create_sdk_mcp_server`),
hand control to the SDK's loop, and let Claude operate *through Pleiades' real tools*.

Why (B) wins for your stated goal: the whole point is "a different perspective from the
inside." If Claude uses its own built-in Read/Write/Bash, it never touches your tools and
learns nothing about them. By routing it through `read_file`, `edit_file`, `web_search`,
`dispatch_subagents_parallel`, the vault, the browser, etc., **every bug Claude hits is a
bug a real Pleiades character would hit** — which is exactly what makes it a useful
stress-tester.

```
            ┌─────────────────────────────────────────────┐
            │  Claude Agent SDK  (runs `claude` CLI)        │
   task ───▶│  • plans, calls tools, compacts, subagents    │
            │  • auth = your `claude login`  → SUBSCRIPTION  │
            └───────────────┬─────────────────────────────┘
                            │  mcp__pleiades__<tool>
                            ▼
            ┌─────────────────────────────────────────────┐
            │  in-process MCP server (claude_backend.py)    │
            │  wraps every Pleiades @tool in the Registry   │
            │  • honours `safe` + exec_policy via can_use_tool
            └───────────────┬─────────────────────────────┘
                            ▼
                   Pleiades Registry (the real tool belt)
```

The existing local-llama.cpp loop (`agent.py`) is untouched and remains the default. This
backend is an **additional tier**, selected explicitly.

---

## 3. What landed (phase 1)

- **`pleiades/harness/claude_backend.py`** — wraps `registry` tools as SDK MCP tools,
  builds `ClaudeAgentOptions`, runs `query()`, streams events, returns answer + usage +
  cost. Permission gate (`can_use_tool`) mirrors Pleiades policy: `safe` tools always run;
  side-effecting tools obey `exec_policy` (allow/deny/ask).
- **New tier** `claude` in `config.py` `DEFAULT_TIERS` (`backend="claude-code"`).
- **CLI**: `pleiades work --tier claude "<task>"` routes to the new runtime instead of the
  in-house loop. Auth is your subscription as long as `ANTHROPIC_API_KEY` is unset (the
  runtime warns if it's set).
- **`pleiades audit`** — the stress-test charter: points a subscription Claude session at
  the live tool belt + repo with instructions to exercise tools, record failures, and
  propose fixes/tests. `--fix` lets it write patches; default is read-only triage.

### Usage

```bash
# one-off task, Claude as the brain, billed to your subscription
pleiades work --tier claude "find the largest .py file and explain what it does"

# run it as a character (identity/vault/email/memory bound), Claude driving
pleiades work --tier claude --as alice "check the inbox for a verification code, store it in the vault"

# stress-test the tool belt (read-only triage)
pleiades audit

# let it actually write fixes + tests for what it finds
pleiades audit --fix --policy allow
```

---

## 4. How far this goes (the roadmap)

1. **Self-audit loop (done, phase 1).** Claude exercises each tool, logs what breaks,
   writes a report. With `--fix`, it branches, patches, and adds regression tests.
2. **Scheduled tool-health pass.** A weekly `pleiades audit` run that opens a PR with any
   fixes + a health summary. Cheap on a Pro plan if scoped to a tool subset per run.
3. **Schema hardening.** The local 7B models mis-call tools far more than Claude does;
   Claude's failures on the *same schemas* surface ambiguous descriptions and missing
   constraints. Feed those back into the `@tool` docstrings/types.
4. **Differential testing.** Run the same task through `--tier local` and `--tier claude`,
   diff the tool-call traces, and surface where the local model goes wrong — a training/
   prompt signal you can't get any other way.
5. **One tool surface.** Eventually the in-process MCP server is also mountable by real
   Claude Code / Cowork, so the *same* belt is usable from outside Pleiades too.

---

## 5. Caveats / gotchas

- **Pro tier = small credit pool.** Keep audits scoped (a subset of tools, a capped
  `max_turns` / `max_budget_usd`) rather than open-ended marathons.
- **Don't double-bill.** If you ever set `ANTHROPIC_API_KEY` for the raw-API `cloud`
  tiers, the `claude` runtime will detect it and warn — because with a key present, Claude
  Code may bill the API instead of your subscription.
- **The SDK needs the `claude` CLI on PATH** (it is: `~/.local/bin/claude`).
- **Loop ownership.** In `--tier claude`, Pleiades' own loop features (tool-search mode,
  context squeeze, the `agent.py` system prompt) do **not** apply — the SDK provides its
  own equivalents. The harness contributes the *tools*, not the loop.
