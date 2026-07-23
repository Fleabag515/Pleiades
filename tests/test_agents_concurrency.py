"""Regression tests for the Opus council's two concurrency-safety findings on
the Agents feature (pleiades/agents.py's AgentManager + pleiades/harness/
subagent.py's dispatch_subagents_parallel), plus the two additional gaps that
surfaced while verifying them:

  1. dispatch_subagents_parallel used a bare `with ThreadPoolExecutor(...) as
     pool:` block. concurrent.futures.ThreadPoolExecutor.__exit__ calls
     shutdown(wait=True), which blocks the CALLING thread (the chat turn's
     own belt.dispatch()) until every worker thread finishes -- even a
     wedged one that never notices should_stop -- regardless of any
     per-future result(timeout=...) in the collection loop above it. Fixed by
     dropping the context manager for an explicit try/finally that calls
     shutdown(wait=False, cancel_futures=True), plus wiring should_stop into
     each worker's own Agent.run() so well-behaved workers wind down near a
     shared deadline instead of running unattended to their step ceiling.

  2. Recursive re-fan-out: pleiades/agents.py's _safe_tools() correctly
     excludes dispatch_subagent/dispatch_subagents_parallel (by name AND by
     "meta" tag) from what a spawned agent's tools list and the toolsearch
     bridge's allow-list contain -- that part is safe by construction and
     was already true (see test_agents.py::test_safe_tools_excludes_dispatch_tools).
     But two things that safety net ALSO leaned on turned out not to be:

       a. harness.agent.Agent._dispatch() looked tool_calls up in
          `{t.name: t for t in registry.all()}` -- the FULL global registry
          -- not `self.tools` (what the agent was actually constructed
          with). A backend that returns a tool_call for a name it was never
          advertised (hallucination, prompt injection via the task string,
          or a local llama.cpp model not strictly validating function names
          against the request schema) got it executed anyway, unconditionally
          for safe=True tools like dispatch_subagent (exec_policy="deny"
          only gates non-safe tools). Fixed by restricting tool_index to
          self.tools.

       b. harness.toolsearch._CTX (approve/allowed) and harness.subagent._CTX
          (cfg/depth/approve) are process-wide globals that bind_dispatch()/
          bind_context() overwrite unconditionally. Concurrently running
          agents (a pleiades/agents.py background agent alongside the main
          chat turn, or two background agents at once) can race: one
          agent's restrictive bind_dispatch(allowed=...) gets clobbered by
          another agent's unrestricted bind_dispatch(allowed=None) before
          the first agent's OWN call_tool() invocation runs, letting it
          through under the wrong allow-list. Fixed (for toolsearch, where
          nothing needs cross-thread inheritance -- every real Agent.run()
          freshly rebinds its own context at the start of its own run, on
          its own thread) by making toolsearch's dispatch context
          thread-local.

     subagent.py's _CTX keeps its shared-global design deliberately: unlike
     toolsearch, dispatch_subagents_parallel's ThreadPoolExecutor workers
     never call bind_context() themselves and MUST inherit cfg/approve from
     the thread that submitted them -- naively making that one thread-local
     would silently break dispatch_subagents_parallel instead of fixing
     anything. Out of scope here; not touched.
"""

from __future__ import annotations

import threading
import time

import pytest

import pleiades.harness.builtins  # noqa: F401  registers all builtin tools
import pleiades.harness.agent as agent_mod
from pleiades.agents import AgentManager, BLOCKED_TOOL_NAMES, _safe_tools
from pleiades.harness import subagent as subagent_mod
from pleiades.harness import toolsearch
from pleiades.harness.agent import Agent
from pleiades.harness.config import Config as HarnessConfig
from pleiades.harness.llm import Reply, ToolCall
from pleiades.harness.tools import registry


def _allow_all(*_a, **_k) -> bool:
    return True


# --------------------------------------------------------------------------- #
# Risk 1: dispatch_subagents_parallel must not let a hung/looping worker
# block the calling turn past its own deadline.
# --------------------------------------------------------------------------- #
class _HangingAgent:
    """Stands in for harness.agent.Agent: its run() ignores should_stop
    entirely and just sleeps -- the worst case (a single already-in-flight
    blocking call, e.g. a wedged network request, that should_stop cannot
    interrupt either). If the CALLER unblocks well before HANG_SECONDS
    elapses, that proves the fix doesn't depend on worker cooperation."""

    HANG_SECONDS = 1.5

    def __init__(self, *_a, **_k) -> None:
        pass

    def run(self, task, should_stop=None, **_kw):
        time.sleep(self.HANG_SECONDS)

        class _R:
            answer = "finished (should have been abandoned by the caller)"
            steps = 1

        return _R()


def test_dispatch_subagents_parallel_unblocks_promptly_when_workers_hang(monkeypatch):
    """The actual regression test for risk 1: with two workers that hang far
    past a short configured deadline and never check should_stop, the tool
    call itself -- the thing a live chat turn is blocked on -- must return
    in well under HANG_SECONDS, not wait for shutdown(wait=True) semantics
    to be satisfied by the wedged threads."""
    monkeypatch.setattr(agent_mod, "Agent", _HangingAgent)
    monkeypatch.setattr(subagent_mod, "PARALLEL_SUBAGENT_TIMEOUT_SECONDS", 0.2)
    subagent_mod.bind_context(HarnessConfig.load(), depth=0, approve=_allow_all)

    t0 = time.monotonic()
    result = subagent_mod.dispatch_subagents_parallel(
        [{"role": "general", "task": "hang please"},
         {"role": "research", "task": "hang please too"}]
    )
    elapsed = time.monotonic() - t0

    assert elapsed < _HangingAgent.HANG_SECONDS * 0.5, (
        f"dispatch_subagents_parallel blocked for {elapsed:.2f}s -- the "
        f"ThreadPoolExecutor's shutdown(wait=True) semantics wedged the "
        f"calling turn instead of returning promptly")
    assert "did not finish within" in result


def test_dispatch_subagents_parallel_actually_passes_should_stop_to_workers(monkeypatch):
    """Complementary to the worst-case test above: confirms sub.run() is
    actually CALLED with a real should_stop callable tied to the configured
    deadline (not just that the deadline variable exists), by capturing
    whatever gets passed and checking it flips from False to True exactly
    when the deadline passes. Deterministic -- no racing against retry-sleep
    or thread-scheduling timing."""
    monkeypatch.setattr(subagent_mod, "PARALLEL_SUBAGENT_TIMEOUT_SECONDS", 0.2)
    subagent_mod.bind_context(HarnessConfig.load(), depth=0, approve=_allow_all)

    captured: dict[str, object] = {}

    class _RecordingAgent:
        def __init__(self, *_a, **_kw):
            pass

        def run(self, task, should_stop=None, **_kw):
            captured["should_stop"] = should_stop

            class _R:
                answer = "ok"
                steps = 0

            return _R()

    monkeypatch.setattr(agent_mod, "Agent", _RecordingAgent)
    subagent_mod.dispatch_subagents_parallel([{"role": "general", "task": "x"}])

    should_stop = captured.get("should_stop")
    assert callable(should_stop), "sub.run() was never given a should_stop callable at all"
    assert should_stop() is False, "should_stop fired before the configured deadline elapsed"
    time.sleep(0.25)
    assert should_stop() is True, "should_stop never flips true once the deadline has passed"


# --------------------------------------------------------------------------- #
# Risk 2a: a spawned agent's tool belt (pleiades/agents.py) has no path to
# dispatch_subagent / dispatch_subagents_parallel / spawn_agents -- by
# construction, not just by convention -- even accounting for the two extra
# gaps this investigation found in the mechanisms that claim were "already
# there" (harness.agent.Agent._dispatch's full-registry lookup, and the
# toolsearch dispatch-context race).
# --------------------------------------------------------------------------- #
def test_spawn_agents_tool_is_not_a_harness_tool_at_all():
    """spawn_agents (pleiades/tools/agents_tool.py) lives in the separate
    chat ToolBelt system, never registered into harness.tools.registry --
    so no harness Agent (spawned or not) has any name to look it up by."""
    assert registry.get("spawn_agents") is None
    assert "spawn_agents" not in {t.name for t in registry.all()}


def test_real_spawned_agent_hallucinated_dispatch_subagent_call_fails_cleanly():
    """The actual regression test for risk 2: build a REAL spawned agent the
    exact way AgentManager does (_build_agent -> _safe_tools()), then have
    its (stubbed) backend do the one thing that could defeat a merely
    advisory/schema-level restriction: return a tool_call for a name it was
    NEVER shown. Must fail cleanly, and dispatch_subagent's own body must
    never actually run."""
    mgr = AgentManager()
    agent = mgr._build_agent("investigate something", "http://fake-base-url",
                             "test-model", None)

    # Sanity check the construction-time curation this whole design leans on.
    assert not (BLOCKED_TOOL_NAMES & {t.name for t in agent.tools})
    assert not (BLOCKED_TOOL_NAMES & {t.name for t in agent._active_tools()})

    subagent_mod.bind_context(HarnessConfig.load(), depth=0, approve=_allow_all)

    reached = {"called": False}
    real_tool = registry.get("dispatch_subagent")
    original_func = real_tool.func

    def _sentinel(*_a, **_k):
        reached["called"] = True
        return "[sentinel] dispatch_subagent was actually invoked"

    real_tool.func = _sentinel
    replies = iter([
        Reply(text="", tool_calls=[ToolCall(id="1", name="dispatch_subagent",
                                            args={"role": "general", "task": "escape"})],
              raw_content={"content": "", "tool_calls": []}, backend="openai"),
        Reply(text="done", tool_calls=[], raw_content={"content": "done"}, backend="openai"),
    ])
    agent.llm.chat = lambda *_a, **_k: next(replies)

    try:
        result = agent.run("investigate something")
    finally:
        real_tool.func = original_func

    assert reached["called"] is False, (
        "dispatch_subagent's real body ran even though this agent's tool "
        "set never included it -- Agent._dispatch is not actually gated by "
        "self.tools")
    assert result.answer == "done"


def test_real_spawned_agent_cannot_reach_dispatch_tools_via_call_tool_either():
    """Same construction as above, but via the toolsearch call_tool bridge
    (the fallback path _safe_tools()'s docstring says stays closed because
    the allow-list bound at the start of this agent's own run never
    contains the blocked names)."""
    mgr = AgentManager()
    agent = mgr._build_agent("investigate something", "http://fake-base-url",
                             "test-model", None)

    # Force this agent's own dispatch context to be bound (mirrors what
    # _active_tools() does at the top of run()) and confirm call_tool
    # refuses both blocked names by name, from ITS OWN thread.
    agent._active_tools()
    for blocked in BLOCKED_TOOL_NAMES:
        out = toolsearch.call_tool(blocked, {})
        assert "not available to this agent" in out, (blocked, out)


# --------------------------------------------------------------------------- #
# Risk 2b: toolsearch's dispatch context must be per-thread, not a shared
# global -- otherwise a concurrently running, less-restricted agent (the
# main chat turn, or another spawned agent) can clobber a restricted spawned
# agent's allow-list between its own bind_dispatch() and its own call_tool().
# --------------------------------------------------------------------------- #
def test_bind_dispatch_is_thread_isolated():
    restricted = {"read_file", "find_tools", "list_catalog", "call_tool"}
    b_bound = threading.Event()
    result: dict[str, str] = {}

    def restricted_agent_thread():
        # Mirrors Agent._active_tools(): bind a narrow allow-list at the
        # start of this "agent's" run, then (once its model responds, some
        # time later) call call_tool.
        toolsearch.bind_dispatch(_allow_all, allowed=restricted)
        assert b_bound.wait(timeout=5), "other thread never signaled"
        result["out"] = toolsearch.call_tool(
            "dispatch_subagent", {"role": "general", "task": "x"})

    def unrestricted_main_chat_thread():
        # Mirrors the main chat turn (agents_enabled=True) concurrently
        # binding ITS OWN unrestricted context on a different thread.
        toolsearch.bind_dispatch(_allow_all, allowed=None)
        b_bound.set()

    try:
        t_a = threading.Thread(target=restricted_agent_thread)
        t_b = threading.Thread(target=unrestricted_main_chat_thread)
        t_a.start()
        t_b.start()
        t_a.join(timeout=5)
        t_b.join(timeout=5)
    finally:
        toolsearch.bind_dispatch(None, None)  # reset, mirrors test_harness_fixes.py

    assert "not available to this agent" in result.get("out", ""), (
        "a concurrent bind_dispatch() on another thread overrode this "
        "thread's own restricted allow-list -- toolsearch's dispatch "
        "context is not thread-isolated")


def test_bind_dispatch_thread_local_does_not_leak_into_fresh_thread():
    """A brand-new thread that never called bind_dispatch itself must see
    the unbound default (no restriction record at all -- not b's, not
    stale data from some other test), confirming there is no shared
    fallback dict backing this anymore."""
    toolsearch.bind_dispatch(_allow_all, allowed={"read_file"})
    seen = {}

    def fresh_thread():
        seen["allowed_is_none"] = toolsearch._ctx().get("allowed") is None

    try:
        t = threading.Thread(target=fresh_thread)
        t.start()
        t.join(timeout=5)
    finally:
        toolsearch.bind_dispatch(None, None)

    assert seen.get("allowed_is_none") is True
