"""WaitTool (chat ToolBelt path): schema, cap enforcement, and cancellation."""

import time

from pleiades.profiles import Profile
from pleiades.tools import ToolBelt, ToolContext, build_default_belt
from pleiades.tools.wait_tool import MAX_WAIT_SECONDS, WaitTool


def _ctx(should_stop=None):
    ctx = ToolContext(profile=Profile(name="testchar"), vault=None, settings=None)
    if should_stop is not None:
        # Set on the instance (not a class dict) so a plain lambda/function is
        # returned as-is on attribute access instead of becoming a bound method
        # (which would silently inject an unwanted `self` arg).
        ctx.should_stop = should_stop
    return ctx


def test_schema_shape():
    tool = WaitTool()
    schema = tool.schema()
    assert schema["function"]["name"] == "wait"
    props = schema["function"]["parameters"]["properties"]
    assert "seconds" in props
    assert schema["function"]["parameters"]["required"] == ["seconds"]


def test_safe_flag_true():
    # Waiting has no side effects — never gated by exec_policy.
    assert WaitTool().safe is True


def test_default_belt_includes_wait():
    belt = build_default_belt(_ctx())
    assert "wait" in belt


def test_wait_actually_waits_measured_duration():
    tool = WaitTool()
    start = time.monotonic()
    out = tool.run(_ctx(), seconds=0.05)
    elapsed = time.monotonic() - start
    assert "Waited" in out
    assert elapsed >= 0.05


def test_wait_rejects_absurd_request_with_clear_cap():
    tool = WaitTool()
    out = tool.run(_ctx(), seconds=10_000)
    assert "[wait error]" in out
    assert str(MAX_WAIT_SECONDS) in out


def test_wait_rejects_zero_or_negative():
    tool = WaitTool()
    assert "[wait error]" in tool.run(_ctx(), seconds=0)
    assert "[wait error]" in tool.run(_ctx(), seconds=-5)


def test_wait_rejects_non_numeric():
    tool = WaitTool()
    out = tool.run(_ctx(), seconds="soon")
    assert "[wait error]" in out


def test_wait_rejects_nan_and_inf():
    tool = WaitTool()
    assert "[wait error]" in tool.run(_ctx(), seconds=float("inf"))
    assert "[wait error]" in tool.run(_ctx(), seconds=float("nan"))


def test_wait_honors_should_stop_and_returns_early():
    tool = WaitTool()
    start = time.monotonic()
    out = tool.run(_ctx(should_stop=lambda: True), seconds=5)
    elapsed = time.monotonic() - start
    assert "cancelled" in out.lower()
    assert elapsed < 1.0  # did not block for anywhere near the full 5s


def test_wait_without_should_stop_attribute_still_works():
    # ctx=None (or any ctx lacking should_stop) must just wait out the duration.
    tool = WaitTool()
    out = tool.run(None, seconds=0.05)
    assert "Waited" in out


def test_dispatch_through_toolbelt():
    belt = ToolBelt([WaitTool()])
    out = belt.dispatch("wait", {"seconds": 0.05}, ctx=None)
    assert "Waited" in out
