"""pleiades.agents: ephemeral background sub-agents spawned from chat.

Covers the safety properties the owner called out as most important:
concurrency cap, depth-cap-1 (no recursive spawning), should_stop actually
stopping a running thread promptly, and -- the one explicitly flagged as
not-to-be-skipped -- that the agents_enabled toggle really does remove
dispatch_subagent/dispatch_subagents_parallel from the toolsearch bridge's
allowed set when off, closing the pre-existing backdoor described in
Engine._bridge_harness_tools.
"""

import threading
import time

import pytest

from pleiades.agents import AgentManager, BLOCKED_TOOL_NAMES, _EphemeralMemory, _safe_tools


class _Result:
    def __init__(self, answer="", steps=0):
        self.answer = answer
        self.steps = steps


# --------------------------------------------------------------------------- #
# Tool set given to spawned agents never includes the dispatch tools
# --------------------------------------------------------------------------- #
def test_safe_tools_excludes_dispatch_tools():
    names = {t.name for t in _safe_tools()}
    assert not (BLOCKED_TOOL_NAMES & names), names
    # discovery trio should still be there (all safe=True) so a big safe
    # catalog stays reachable via tool-search
    assert {"find_tools", "list_catalog", "call_tool"} <= names
    # every tool handed to a spawned agent must be safe (never gated) --
    # the second, independent enforcement layer alongside exec_policy=deny
    assert all(t.safe for t in _safe_tools())


# --------------------------------------------------------------------------- #
# Concurrency cap (~3, configurable) — never more than N agents "running" at once
# --------------------------------------------------------------------------- #
def test_concurrency_cap_enforced():
    cap = 2
    mgr = AgentManager(max_concurrent=cap)
    release = threading.Event()
    lock = threading.Lock()
    state = {"current": 0, "peak": 0}

    class SlowAgent:
        def run(self, task, on_event=None, should_stop=None):
            with lock:
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
            release.wait(timeout=5)
            with lock:
                state["current"] -= 1
            return _Result("done", 1)

    mgr.spawn_agents(["a", "b", "c", "d"], agent_factory=lambda *a: SlowAgent())
    deadline = time.time() + 3
    while time.time() < deadline and sum(
        1 for a in mgr.list() if a["status"] == "running"
    ) < cap:
        time.sleep(0.02)

    running = [a for a in mgr.list() if a["status"] == "running"]
    queued = [a for a in mgr.list() if a["status"] == "queued"]
    assert len(running) == cap, mgr.list()
    assert len(queued) == 4 - cap, mgr.list()
    assert state["peak"] <= cap

    release.set()
    deadline = time.time() + 5
    while time.time() < deadline and any(
        a["status"] in ("queued", "running") for a in mgr.list()
    ):
        time.sleep(0.02)
    assert all(a["status"] == "done" for a in mgr.list()), mgr.list()
    assert state["peak"] <= cap


# --------------------------------------------------------------------------- #
# should_stop cancels a RUNNING agent promptly (not just at the next spawn)
# --------------------------------------------------------------------------- #
def test_should_stop_cancels_running_agent_promptly():
    mgr = AgentManager(max_concurrent=3)
    started = threading.Event()

    class LoopingAgent:
        def run(self, task, on_event=None, should_stop=None):
            started.set()
            steps = 0
            while steps < 10_000:
                if should_stop and should_stop():
                    return _Result("cancelled", steps)
                time.sleep(0.01)
                steps += 1
            return _Result("finished (never stopped)", steps)

    ids = mgr.spawn_agents(["long task"], agent_factory=lambda *a: LoopingAgent())
    assert started.wait(timeout=2), "agent never started running"

    t0 = time.time()
    assert mgr.stop(ids[0]) is True
    deadline = time.time() + 2
    while time.time() < deadline and mgr.get(ids[0])["status"] not in (
        "stopped", "done", "error",
    ):
        time.sleep(0.01)
    elapsed = time.time() - t0

    rec = mgr.get(ids[0])
    assert rec["status"] == "stopped", rec
    assert elapsed < 1.0, f"stop took {elapsed:.2f}s -- should be near-instant"


def test_stop_unknown_agent_returns_false():
    mgr = AgentManager()
    assert mgr.stop("no-such-id") is False


def test_spawn_agents_rejects_empty_task_list():
    mgr = AgentManager()
    with pytest.raises(ValueError):
        mgr.spawn_agents([], agent_factory=lambda *a: None)
    with pytest.raises(ValueError):
        mgr.spawn_agents(["   ", ""], agent_factory=lambda *a: None)


def test_spawn_agents_records_error_status_on_exception():
    mgr = AgentManager()

    class BrokenAgent:
        def run(self, task, on_event=None, should_stop=None):
            raise RuntimeError("boom")

    ids = mgr.spawn_agents(["x"], agent_factory=lambda *a: BrokenAgent())
    deadline = time.time() + 2
    while time.time() < deadline and mgr.get(ids[0])["status"] == "running":
        time.sleep(0.02)
    rec = mgr.get(ids[0])
    assert rec["status"] == "error"
    assert "boom" in rec["error"]


# --------------------------------------------------------------------------- #
# Ephemeral memory really is isolated from the character's real store
# --------------------------------------------------------------------------- #
def test_agent_memory_override_bypasses_global_store():
    """Agent(memory=...) must win over whatever builtins.memory._store()
    would otherwise return -- the global is process-wide, so a concurrently
    running chat turn for a real character could have it bound at the exact
    moment a background agent is constructed. This is the actual mechanism
    that keeps a spawned agent's memory ephemeral; not calling bind_memory()
    would NOT be enough on its own."""
    from pleiades.harness.agent import Agent
    from pleiades.harness.config import Config as HarnessConfig
    from pleiades.harness.builtins.memory import bind_memory

    class _RealCharacterStoreStandIn:
        def working(self):
            return "REAL CHARACTER MEMORY -- must never leak into a spawned agent"

        def recall(self, query):
            return ""

    bind_memory(_RealCharacterStoreStandIn())
    try:
        eph = _EphemeralMemory()
        cfg = HarnessConfig.load()
        agent = Agent(cfg, tools=[], memory=eph)
        assert agent.memory is eph
        assert agent.memory.working() == ""
        eph.scratch_append("hello")
        assert "hello" in agent.memory.working()
        assert "REAL CHARACTER" not in agent.memory.working()
    finally:
        bind_memory(None)


def test_ephemeral_memory_implements_expected_surface():
    m = _EphemeralMemory()
    assert m.working() == ""
    m.scratch_append("note one")
    assert "note one" in m.working()
    m.scratch_set_section("Goal", "find the bug")
    assert "find the bug" in m.working()
    slug = m.write("fact", "the sky is blue")
    assert "blue" in m.recall("sky")
    assert m.forget(slug) is True
    assert m.forget(slug) is False
    cp = m.checkpoint_save("mid-task")
    m.scratch_clear()
    assert m.working() == ""
    assert m.checkpoint_load(cp) is not None
    assert "find the bug" in m.working()


# --------------------------------------------------------------------------- #
# THE most important test: the agents_enabled toggle really does remove
# dispatch_subagent/dispatch_subagents_parallel from the toolsearch bridge's
# allowed set when off -- closing the backdoor described in
# Engine._bridge_harness_tools (chat's find_tools/call_tool bridge used to
# make the raw dispatch_subagent/dispatch_subagents_parallel tools reachable
# from ANY chat turn regardless of any UI setting, since bind_dispatch was
# always called with an unrestricted allow-list).
# --------------------------------------------------------------------------- #
def test_agents_disabled_blocks_dispatch_subagent_via_bridge():
    from pleiades import config
    from pleiades.engine import Engine
    from pleiades.harness import toolsearch
    from pleiades.profiles import Profile
    from pleiades.tools import ToolBelt

    e = Engine.__new__(Engine)
    e.settings = config.Settings.load()

    try:
        # OFF (the default for a new/unconfigured character): dispatch_subagent
        # must be unreachable both by name (call_tool) and by discovery
        # (find_tools/list_catalog), even though it's still sitting in the
        # global harness registry the whole time.
        belt_off = ToolBelt([])
        e._bridge_harness_tools(belt_off, Profile(name="agents-off-test", agents_enabled=False))
        out = toolsearch.call_tool("dispatch_subagent", {"role": "general", "task": "x"})
        assert "not available to this agent" in out, out
        out2 = toolsearch.call_tool("dispatch_subagents_parallel", {"tasks": []})
        assert "not available to this agent" in out2, out2
        catalog_off = toolsearch.list_catalog()
        assert "dispatch_subagent" not in catalog_off
        assert "dispatch_subagents_parallel" not in catalog_off
        # other, harmless tools remain reachable -- this isn't a global lockdown
        assert "read_file" in catalog_off

        # ON: the toggle restores the pre-existing (unrestricted) behavior.
        belt_on = ToolBelt([])
        e._bridge_harness_tools(belt_on, Profile(name="agents-on-test", agents_enabled=True))
        catalog_on = toolsearch.list_catalog()
        assert "dispatch_subagent" in catalog_on
        assert "dispatch_subagents_parallel" in catalog_on
    finally:
        toolsearch.bind_dispatch(None, None)  # reset, mirrors test_harness_fixes.py


def test_agents_enabled_gate_on_chat_belt_tool():
    """build_default_belt only adds spawn_agents when the character has
    agents_enabled=True -- mirrors how EmailTool is gated on has_email."""
    from pleiades.profiles import Profile
    from pleiades.tools import ToolBelt, ToolContext, build_default_belt

    ctx_off = ToolContext(profile=Profile(name="off", agents_enabled=False),
                          vault=None, settings=None)
    belt_off = build_default_belt(ctx_off)
    assert "spawn_agents" not in belt_off.names()

    ctx_on = ToolContext(profile=Profile(name="on", agents_enabled=True),
                         vault=None, settings=None)
    belt_on = build_default_belt(ctx_on)
    assert "spawn_agents" in belt_on.names()
