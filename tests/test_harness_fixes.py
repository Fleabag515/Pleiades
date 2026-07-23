"""Regression tests for the harness fixes:

  * openai/ollama tool results carry tool_call_id (binds result -> call)
  * subagent roles resolve to defined tiers (no silent fallback)
  * tool-search respects a per-agent allow-list (subagent sandbox)
  * tool schema unwraps Optional[T] to the real JSON type
  * Agent.run()'s self-reflection check and its failed-model-call retry
    notice are synthetic tool rounds, not role:"user" (fix/anamnesis-
    reflection-injection -- see _synthetic_tool_round's docstring in
    pleiades/harness/agent.py for why: this agent's "openai" backend routes
    through a character's Anamnesis proxy when bound via
    identity.bind_character, and Anamnesis persists "the new user turn" by
    taking the last role=='user' message in the request -- a role:"user"
    self-check or retry notice used to get written into the character's
    real long-term memory as if the user had said it)
"""

import time
from typing import Optional

import pleiades.harness.builtins  # noqa: F401  registers builtin tools
from pleiades.agents import _EphemeralMemory
from pleiades.harness import registry
from pleiades.harness.agent import Agent, _synthetic_tool_round
from pleiades.harness.llm import LLM, Reply, ToolCall
from pleiades.harness.config import Config, DEFAULT_TIERS
from pleiades.harness.subagent import ROLE_TOOLS
from pleiades.harness.tools import tool, _build_schema


def test_result_turns_include_tool_call_id():
    calls = [(ToolCall(id="call_abc", name="read_file", args={}), "out", False)]
    for backend in ("openai", "ollama"):
        turns = Agent._result_turns(backend, calls)
        assert turns and turns[0]["role"] == "tool"
        assert turns[0]["tool_call_id"] == "call_abc", backend


def test_anthropic_result_turns_unchanged():
    calls = [(ToolCall(id="tu_1", name="read_file", args={}), "out", True)]
    turns = Agent._result_turns("anthropic", calls)
    block = turns[0]["content"][0]
    assert block["tool_use_id"] == "tu_1"
    assert block.get("is_error") is True


def test_subagent_roles_map_to_defined_tiers():
    for role, spec in ROLE_TOOLS.items():
        assert spec["tier"] in DEFAULT_TIERS, f"role {role} -> undefined tier {spec['tier']}"


def test_tier_resolution_is_not_a_silent_fallback():
    cfg = Config.load()
    # research/fast must resolve to themselves, not the default tier
    assert cfg.tier("research") is cfg.tiers["research"]
    assert cfg.tier("fast") is cfg.tiers["fast"]


def test_toolsearch_allow_list_blocks_out_of_scope_calls():
    from pleiades.harness import toolsearch
    toolsearch.bind_dispatch(lambda *a, **k: True, allowed={"read_file"})
    try:
        # in-scope but harmless: unknown-name path still gated by allow-list
        out = toolsearch.call_tool("run_shell", {"command": "echo hi"})
        assert "not available to this agent" in out
        # discovery is also restricted to the allow-list
        names = {line.split(" — ")[0] for line in toolsearch.list_catalog().splitlines()}
        assert names == {"read_file"}
    finally:
        toolsearch.bind_dispatch(None, None)  # reset


def test_schema_unwraps_optional():
    @tool(safe=True)
    def _probe(count: Optional[int] = None, names: list = None) -> str:
        """Probe.

        count: how many.
        names: the names.
        """
        return ""
    schema = _build_schema(_probe.__wrapped__ if hasattr(_probe, "__wrapped__") else registry.get("_probe").func)
    props = schema["properties"]
    assert props["count"]["type"] == "integer"
    assert props["names"]["type"] == "array"


# --------------------------------------------------------------------------- #
# fix/anamnesis-reflection-injection: self-reflection + retry notices must be
# synthetic tool rounds, never role:"user".
# --------------------------------------------------------------------------- #

def test_synthetic_tool_round_shape_openai_and_ollama():
    """Direct unit test of the harness's own copy of the helper -- pins the
    openai/ollama wire shape (both share this branch in _result_turns too,
    and both can be routed through Anamnesis via cfg.openai_host /
    cfg.ollama_host)."""
    for backend in ("openai", "ollama"):
        msgs = _synthetic_tool_round(backend, "system_reflection", "check in", "reflect_2")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "system_reflection"
        assert msgs[0]["tool_calls"][0]["id"] == "reflect_2"
        assert msgs[1] == {"role": "tool", "tool_call_id": "reflect_2", "content": "check in"}
        assert all(m["role"] != "user" for m in msgs), backend


def test_synthetic_tool_round_shape_anthropic():
    """Anthropic has no "tool" role -- a real tool result there is a
    role:"user" message with a tool_result content block (see
    Agent._result_turns' anthropic branch). The synthetic round matches
    that shape for wire-format correctness even though the Anamnesis bug
    itself can't happen on this backend (_chat_anthropic talks to the
    Anthropic API directly, never through cfg.openai_host/Anamnesis)."""
    msgs = _synthetic_tool_round("anthropic", "system_reflection", "check in", "reflect_9")
    assert len(msgs) == 2
    assert msgs[0] == {"role": "assistant", "content": [
        {"type": "tool_use", "id": "reflect_9", "name": "system_reflection", "input": {}},
    ]}
    assert msgs[1] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "reflect_9", "content": "check in"},
    ]}


def test_agent_reflection_check_is_synthetic_tool_round(monkeypatch):
    """Regression test for the Anamnesis memory-pollution bug in the harness
    loop: Agent.run() used to append the periodic self-check as
    {"role": "user", ...}. When this agent is bound to a character
    (identity.bind_character routes the "openai" backend through that
    character's Anamnesis proxy), that message became indistinguishable
    from real user speech to Anamnesis's "last user turn" dedup/persistence
    logic (proxy.js) once a run went past eval_interval steps. It must now
    be a fabricated assistant tool_calls message + its role:"tool" result."""
    cfg = Config.load()
    cfg.eval_interval = 1
    cfg.max_rounds = 10

    calls = []

    def fake_chat(self, messages, tools, tier, system=None):
        calls.append([dict(m) for m in messages])
        if len(calls) == 1:
            return Reply(text="", tool_calls=[ToolCall(id="call_1", name="noop", args={})],
                        raw_content={"content": "", "tool_calls": [
                            {"id": "call_1", "type": "function",
                             "function": {"name": "noop", "arguments": "{}"}},
                        ]}, backend="openai")
        return Reply(text="All done.", tool_calls=[], raw_content={"content": "All done."},
                    backend="openai")

    monkeypatch.setattr(LLM, "chat", fake_chat)

    agent = Agent(cfg, tools=[], tier_name="fast", memory=_EphemeralMemory())
    result = agent.run("do the thing")

    assert result.answer == "All done."
    assert len(calls) == 2
    # calls[0]: [user(task)] -- no reflection yet (step 0 is falsy, never fires).
    assert len(calls[0]) == 1
    # calls[1]: [user(task), assistant(real tool_calls), tool(real result),
    #            assistant(synthetic system_reflection), tool(reflection text)]
    round2 = calls[1]
    assert round2[-2]["role"] == "assistant"
    assert round2[-2]["tool_calls"][0]["function"]["name"] == "system_reflection"
    assert round2[-1]["role"] == "tool"
    assert round2[-1]["tool_call_id"] == round2[-2]["tool_calls"][0]["id"]
    assert "[Self-check after" in round2[-1]["content"]
    # The crux of the whole fix: the ONLY role:"user" message anywhere in
    # the request is the genuine original task -- the self-check must never
    # add a second one.
    user_msgs = [m for m in round2 if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "do the thing"


def test_agent_model_error_retry_notice_is_synthetic_tool_round(monkeypatch):
    """Same bug, same fix, different call site: the "the previous model call
    failed, retry" notice Agent.run() injects after a transient LLM error
    used to be role:"user" too."""
    cfg = Config.load()
    cfg.eval_interval = 0  # isolate from the reflection-check injection
    cfg.max_rounds = 10

    calls = []

    def fake_chat(self, messages, tools, tier, system=None):
        calls.append([dict(m) for m in messages])
        if len(calls) == 1:
            raise RuntimeError("upstream hiccup")
        return Reply(text="Recovered.", tool_calls=[], raw_content={"content": "Recovered."},
                    backend="openai")

    monkeypatch.setattr(LLM, "chat", fake_chat)
    monkeypatch.setattr(time, "sleep", lambda *_: None)  # skip the real backoff

    agent = Agent(cfg, tools=[], tier_name="fast", memory=_EphemeralMemory())
    result = agent.run("do the thing")

    assert result.answer == "Recovered."
    assert len(calls) == 2
    round2 = calls[1]
    assert round2[-2]["role"] == "assistant"
    assert round2[-2]["tool_calls"][0]["function"]["name"] == "system_notice"
    assert round2[-1]["role"] == "tool"
    assert "previous model call failed" in round2[-1]["content"]
    user_msgs = [m for m in round2 if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "do the thing"


def test_agent_anthropic_backend_reflection_check_uses_tool_use_shape(monkeypatch):
    """Same regression, confirming the anthropic-backend branch: no bare
    role:"tool" message (invalid on that API) and no accidental role:"user"
    self-check either -- content lives in a tool_result block instead."""
    cfg = Config.load()
    cfg.eval_interval = 1
    cfg.max_rounds = 10

    calls = []

    def fake_chat(self, messages, tools, tier, system=None):
        calls.append([dict(m) for m in messages])
        if len(calls) == 1:
            return Reply(text="", tool_calls=[ToolCall(id="call_1", name="noop", args={})],
                        raw_content=[{"type": "tool_use", "id": "call_1", "name": "noop", "input": {}}],
                        backend="anthropic")
        return Reply(text="All done.", tool_calls=[], raw_content=[{"type": "text", "text": "All done."}],
                    backend="anthropic")

    monkeypatch.setattr(LLM, "chat", fake_chat)

    agent = Agent(cfg, tools=[], tier_name="cloud", memory=_EphemeralMemory())
    result = agent.run("do the thing")

    assert result.answer == "All done."
    round2 = calls[1]
    assert round2[-2] == {"role": "assistant", "content": [
        {"type": "tool_use", "id": "reflect_1", "name": "system_reflection", "input": {}},
    ]}
    assert round2[-1]["role"] == "user"
    block = round2[-1]["content"][0]
    assert block["type"] == "tool_result"
    assert "[Self-check after" in block["content"]
    assert not any(m["role"] == "tool" for m in round2)
