"""Smoke tests for the vendored agent harness (pleiades.harness).

These assert the harness imports cleanly, registers its tool belt, and produces
valid Anthropic/OpenAI tool schemas — without requiring any optional deps
(anthropic, camoufox, ollama). Heavy backends are all lazily imported.
"""

import pleiades.harness as harness
import pleiades.harness.builtins  # noqa: F401  registers all builtin tools
from pleiades.harness import registry


def test_harness_public_surface():
    assert hasattr(harness, "Agent")
    assert hasattr(harness, "Config")
    assert hasattr(harness, "tool")
    assert hasattr(harness, "registry")


def test_builtins_register_expected_tools():
    names = {t.name for t in registry.all()}
    # a representative slice across categories
    for expected in (
        "read_file", "write_file", "grep_search", "get_symbols",
        "run_shell", "run_python",
        "git_status", "git_commit",
        "web_search", "http_fetch",
        "remember", "recall", "set_section",
        "find_tools", "call_tool",
        "dispatch_subagent",
    ):
        assert expected in names, f"missing harness tool: {expected}"
    assert len(names) >= 50  # full belt


def test_tool_schemas_are_valid():
    for t in registry.all():
        anth = t.input_schema()
        assert set(anth) == {"name", "description", "input_schema"}
        assert anth["input_schema"]["type"] == "object"
        oai = t.openai_schema()
        assert oai["type"] == "function"
        assert oai["function"]["name"] == t.name


def test_safe_flag_present():
    read = registry.get("read_file")
    assert read is not None and read.safe is True
    write = registry.get("write_file")
    assert write is not None and write.safe is False
