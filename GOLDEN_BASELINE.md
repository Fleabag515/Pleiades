# Golden Baseline — native-layer roadmap

This file tracks the planned "native layer" work for the Pleiades workspace
harness: the pieces that move the agent from a capable tool-belt toward a
first-class operating environment for local LLMs. It is referenced from the
README's *Workspace harness* section.

The harness shipped in `pleiades/harness/` is the current baseline: a
backend-agnostic agent loop, an auto-schematized tool registry, tool-search for
large fleets, a two-tier memory system, subagents, and an MCP client. The items
below are the next increments on top of that baseline.

## 1. LSP bridge

Give code-navigation tools (`get_symbols`, `grep_search`) real language
intelligence by talking to a Language Server (textDocument/documentSymbol,
definition, references, diagnostics) instead of AST + regex heuristics. Falls
back to the current heuristic path when no server is available, so it stays
offline-friendly.

## 2. Sandboxed executor — landed `0f6e678` (branch `sandboxed-executor`)

`run_shell` / `run_python` currently execute unsandboxed behind the permission
gate. The native layer adds an opt-in sandbox (resource limits, a scoped
filesystem view, and configurable network egress) so untrusted or autonomous
runs can be contained without losing the gated-but-direct default.

Implemented in `pleiades/harness/sandbox.py`, wired into `builtins/shell.py`
(`run_shell`, `run_python`) and `builtins/process.py` (`start_process`).
Strictly additive underneath the approval gate (`exec_policy` /
`Agent._default_approve`), which is unchanged and stays mandatory. Ships:
an unconditional destructive-command floor (not configurable, on even with
the sandbox disabled), a memory ceiling (POSIX `RLIMIT_AS` + a cross-platform
`psutil` watchdog — the watchdog is the real enforcement path on Windows),
opt-in network egress control, and a real in-process filesystem write-guard
for `run_python` specifically (best-effort, regex-based for `run_shell` —
Windows has no chroot/namespace primitive without containers, tracked as a
harder follow-up, not pretended away). New `Settings` fields:
`sandbox_enabled`, `sandbox_mem_mb`, `sandbox_network`. Covered by
`tests/test_sandbox.py` (30 tests); full suite at 174 passed / 2 pre-existing
unrelated failures (confirmed via `git stash` — present without this change).

## 3. Streaming tools

Stream long-running tool output (builds, test runs, dev servers) back into the
loop incrementally rather than returning one capped blob, so the agent can react
mid-run and the user sees progress live. Pairs with the existing
`start_process` / `read_process_output` process tools.

## 4. Multi-agent fabric

Extend subagents from a parent->child tree into a coordinated fabric: shared
working memory, message passing between peers, and role-aware scheduling -- while
keeping each agent's tool sandbox (see the tool-search allow-list) intact.

---

*Status: roadmap. As each item lands, note the commit here and update the
harness section of the README.*
