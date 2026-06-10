"""Regression tests for the harness fixes:

  * openai/ollama tool results carry tool_call_id (binds result -> call)
  * subagent roles resolve to defined tiers (no silent fallback)
  * tool-search respects a per-agent allow-list (subagent sandbox)
  * tool schema unwraps Optional[T] to the real JSON type
"""

from typing import Optional

import pleiades.harness.builtins  # noqa: F401  registers builtin tools
from pleiades.harness import registry
from pleiades.harness.agent import Agent
from pleiades.harness.llm import ToolCall
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
