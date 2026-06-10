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

## 2. Sandboxed executor

`run_shell` / `run_python` currently execute unsandboxed behind the permission
gate. The native layer adds an opt-in sandbox (resource limits, a scoped
filesystem view, and configurable network egress) so untrusted or autonomous
runs can be contained without losing the gated-but-direct default.

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
