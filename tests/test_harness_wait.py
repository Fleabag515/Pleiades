"""Harness builtin `wait`: registration, cap enforcement, and basic behavior.

This is the `pleiades work` / harness-path counterpart to the chat ToolBelt's
WaitTool (tests/test_wait_tool.py) — Pleiades has two tool surfaces and both
need the pause capability.
"""

import time

import pleiades.harness.builtins  # noqa: F401  registers all builtin tools
from pleiades.harness import registry
from pleiades.harness.builtins.wait import MAX_WAIT_SECONDS, wait


def test_wait_registered_in_harness():
    t = registry.get("wait")
    assert t is not None
    assert t.safe is True


def test_wait_schema_has_seconds_param():
    t = registry.get("wait")
    schema = t.input_schema()
    assert "seconds" in schema["input_schema"]["properties"]


def test_wait_actually_waits_measured_duration():
    start = time.monotonic()
    out = wait(0.05)
    elapsed = time.monotonic() - start
    assert "Waited" in out
    # Sleep granularity is not exact everywhere (measured 0.047 on a Windows
    # runner for a 0.05s request) -- allow a 10% floor instead of >= the
    # requested duration verbatim.
    assert elapsed >= 0.045


def test_wait_rejects_absurd_request_with_clear_cap():
    out = wait(10_000)
    assert "Error" in out
    assert str(MAX_WAIT_SECONDS) in out


def test_wait_rejects_zero_or_negative():
    assert "Error" in wait(0)
    assert "Error" in wait(-5)


def test_wait_rejects_non_numeric():
    assert "Error" in wait("soon")


def test_wait_rejects_nan_and_inf():
    assert "Error" in wait(float("inf"))
    assert "Error" in wait(float("nan"))
