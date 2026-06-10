"""Model-callable system tools: registration, dispatch, and guardrails."""

from pleiades import system_ops
from pleiades.profiles import Profile
from pleiades.tools import ToolBelt, ToolContext
from pleiades.tools.system_tools import (CharactersTool, HardwareTool,
                                         ModelsTool, ProfileTool)


def _ctx(name="testchar"):
    return ToolContext(profile=Profile(name=name), vault=None, settings=None)


def test_default_belt_includes_system_tools():
    from pleiades.tools import build_default_belt
    belt = build_default_belt(_ctx())
    for tool in ("hardware", "models", "profile", "characters"):
        assert tool in belt


def test_models_tool_bad_action():
    out = ModelsTool().run(_ctx(), action="explode")
    assert out.startswith("[models error]")


def test_models_fetch_requires_repo():
    out = ModelsTool().run(_ctx(), action="fetch")
    assert "needs 'repo'" in out


def test_characters_delete_requires_confirmation(tmp_path, monkeypatch):
    out = system_ops.characters_op("delete", name="ghost")
    assert "confirm=" in out  # asks for confirmation instead of deleting


def test_characters_cannot_delete_self():
    out = CharactersTool().run(_ctx("zoe"), action="delete", name="zoe",
                               confirm="delete zoe")
    assert "cannot delete yourself" in out


def test_profile_tool_validates_fields():
    out = ProfileTool().run(_ctx(), action="set", field="nonsense", value="x")
    assert "unknown field" in out
    out = ProfileTool().run(_ctx(), action="set", field="imap_port", value="abc")
    assert "must be an integer" in out


def test_profile_get_never_shows_secrets():
    out = system_ops.profile_show(Profile(name="x", email_address="a@b.c"))
    assert "password" not in out.lower()


def test_hardware_tool_returns_report():
    out = HardwareTool().run(_ctx())
    assert "RAM:" in out and "CPU threads:" in out


def test_harness_registers_system_tools():
    from pleiades.harness.tools import registry
    import pleiades.harness.builtins.inference  # noqa: F401  (registers)
    names = {t.name for t in registry.all()}
    assert {"hardware", "models", "profile", "characters"} <= names
    # hardware is read-only; the rest must pass the exec policy
    assert registry.get("hardware").safe
    assert not registry.get("models").safe
    assert not registry.get("characters").safe


def test_dispatch_through_belt_never_raises():
    belt = ToolBelt([ModelsTool()])
    out = belt.dispatch("models", '{"action": "fetch", "repo": ""}', _ctx())
    assert out.startswith("[models error]")
