from types import SimpleNamespace

from pleiades.tools import Tool, ToolBelt


class _FakeProfileAnamnesis:
    """Stand-in for pleiades.anamnesis.Anamnesis: the webui's ProfileManager
    only ever calls .exists()/.create_character()/.delete() on the path these
    tests exercise (profile create/tools/policy — no chat, no proxy). Avoids
    needing a live Anamnesis daemon on :9000 just to test the local-only
    tool-policy plumbing."""

    def __init__(self, *_a, **_k):
        self._names = set()

    def exists(self, name):
        return name in self._names

    def create_character(self, name, **_cfg):
        self._names.add(name)
        return {"name": name}

    def delete(self, name):
        self._names.discard(name)


class EchoTool(Tool):
    name = "echo"
    description = "Echo back the text."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, ctx, text):
        return f"echo: {text}"


def test_openai_schema_shape():
    belt = ToolBelt([EchoTool()])
    schema = belt.openai_schema()[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert "properties" in schema["function"]["parameters"]


def test_dispatch_json_args():
    belt = ToolBelt([EchoTool()])
    assert belt.dispatch("echo", '{"text": "hi"}', ctx=None) == "echo: hi"


def test_dispatch_dict_args():
    belt = ToolBelt([EchoTool()])
    assert belt.dispatch("echo", {"text": "yo"}, ctx=None) == "echo: yo"


def test_dispatch_unknown_tool():
    belt = ToolBelt([EchoTool()])
    assert "unknown tool" in belt.dispatch("nope", {}, ctx=None)


def test_dispatch_bad_args_does_not_raise():
    belt = ToolBelt([EchoTool()])
    out = belt.dispatch("echo", {"wrong": "x"}, ctx=None)
    assert out.startswith("[error]")


def test_cli_has_ui_command():
    """`pleiades ui` must exist (the web UI docs point users at it)."""
    from pleiades.cli import cli as root

    assert "ui" in root.commands


def test_webui_app_builds():
    """create_app() wires every route without needing live services."""
    fastapi = __import__("pytest").importorskip("fastapi")  # noqa: F841
    from pleiades.webui import create_app

    app = create_app()
    paths = {r.path for r in app.routes}
    assert "/api/status" in paths and "/api/profiles" in paths


def test_webui_has_update_and_hardware_routes():
    __import__("pytest").importorskip("fastapi")
    from pleiades.webui import create_app

    paths = {r.path for r in create_app().routes}
    assert {"/api/update/check", "/api/update", "/api/update/status",
            "/api/hardware", "/api/models/fetch"} <= paths


def test_webui_has_email_discord_and_logs_routes():
    __import__("pytest").importorskip("fastapi")
    from pleiades.webui import create_app

    paths = {r.path for r in create_app().routes}
    assert {"/api/profiles/{name}/email/inbox",
            "/api/profiles/{name}/email/message/{mid}",
            "/api/profiles/{name}/email/send",
            "/api/profiles/{name}/discord/info",
            "/api/models/{name}/logs",
            "/api/models/hf-search"} <= paths


def test_webui_has_chat_session_routes():
    __import__("pytest").importorskip("fastapi")
    from pleiades.webui import create_app

    paths = {r.path for r in create_app().routes}
    assert {"/api/chats", "/api/chats/{chat_id}",
            "/api/chats/{chat_id}/message"} <= paths


def test_webui_has_profile_tools_route():
    __import__("pytest").importorskip("fastapi")
    from pleiades.webui import create_app

    paths = {r.path for r in create_app().routes}
    assert "/api/profiles/{name}/tools" in paths


def test_toolbelt_get_returns_tool_or_none():
    belt = ToolBelt([EchoTool()])
    assert belt.get("echo") is not None
    assert belt.get("echo").name == "echo"
    assert belt.get("nope") is None


def test_profile_tools_endpoint_reflects_exec_policy_and_usage(monkeypatch):
    """Full round trip: create a profile, hit GET .../tools, and check that
    status honestly reflects exec_policy (deny -> blocked, ask -> needs
    approval, allow -> available) and that use_count/last_used come from
    real chats.tool_usage() data, not placeholders."""
    pytest = __import__("pytest")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pleiades import chats
    import pleiades.profiles as profiles_mod
    from pleiades.webui import create_app

    monkeypatch.setattr(profiles_mod, "Anamnesis", _FakeProfileAnamnesis)
    app = create_app()
    client = TestClient(app, base_url="http://127.0.0.1")

    resp = client.post("/api/profiles", json={"name": "tools-test-char"})
    assert resp.status_code == 200

    resp = client.put("/api/profiles/tools-test-char",
                      json={"exec_policy": "deny"})
    assert resp.status_code == 200

    resp = client.get("/api/profiles/tools-test-char/tools")
    assert resp.status_code == 200
    body = resp.json()
    assert body["exec_policy"] == "deny"
    by_name = {t["name"]: t for t in body["tools"]}
    # SearchTool.safe=True -> always available, deny gate never applies to it.
    assert by_name["search"]["safe"] is True
    assert by_name["search"]["status"] == "available"
    # VaultTool is not safe -> exec_policy=deny actually blocks it.
    assert by_name["vault"]["safe"] is False
    assert by_name["vault"]["status"] == "blocked"
    assert by_name["search"]["use_count"] == 0
    assert by_name["search"]["last_used"] is None

    # Now record a real completed call for this character and confirm the
    # endpoint picks it up -- not fabricated, mined from chats.tool_usage().
    c = chats.create("tools-test-char")
    chats.append_turn(c["id"], "go", [
        {"t": "tool", "name": "search", "args": "{}", "output": "...",
         "ok": True, "ts": 4242.0},
    ], {"tokens": 1, "seconds": 0.1, "tps": 1.0})

    resp = client.get("/api/profiles/tools-test-char/tools")
    body = resp.json()
    by_name = {t["name"]: t for t in body["tools"]}
    assert by_name["search"]["use_count"] == 1
    assert by_name["search"]["last_used"] == 4242.0

    chats.delete(c["id"])
    client.delete("/api/profiles/tools-test-char")


# --------------------------------------------------------------------------- #
# exec_policy="preset" (per-tool ask/allow override) -- see
# profiles.Profile.tool_policies and ToolBelt.dispatch's gate in
# pleiades/tools/__init__.py.
# --------------------------------------------------------------------------- #

def _ctx(exec_policy=None, tool_policies=None):
    """Minimal stand-in for a ToolContext: dispatch's gate only ever reads
    ctx.profile (and ctx.settings as a fallback it never reaches here, since
    every case below sets an explicit profile.exec_policy)."""
    profile = SimpleNamespace(exec_policy=exec_policy, tool_policies=tool_policies or {})
    return SimpleNamespace(profile=profile, settings=None)


def test_dispatch_preset_defaults_to_allow_for_tool_with_no_override():
    """No approve callback, no tty -- if preset's per-tool fallback weren't
    "allow", this would come back as an explicit needs-approval error
    instead of actually running the tool."""
    belt = ToolBelt([EchoTool()])
    ctx = _ctx(exec_policy="preset")
    assert belt.dispatch("echo", {"text": "hi"}, ctx=ctx) == "echo: hi"


def test_dispatch_preset_per_tool_ask_override_requires_approval():
    belt = ToolBelt([EchoTool()])
    ctx = _ctx(exec_policy="preset", tool_policies={"echo": "ask"})

    # No approve callback, no tty -> explicit error, same shape as the
    # existing blanket exec_policy="ask" non-interactive error.
    out = belt.dispatch("echo", {"text": "hi"}, ctx=ctx)
    assert "needs approval" in out

    assert belt.dispatch("echo", {"text": "hi"}, ctx=ctx, approve=lambda t, a: True) == "echo: hi"
    assert belt.dispatch("echo", {"text": "hi"}, ctx=ctx, approve=lambda t, a: False) == "[cancelled] echo"


def test_dispatch_preset_per_tool_allow_override_runs_immediately():
    belt = ToolBelt([EchoTool()])
    ctx = _ctx(exec_policy="preset", tool_policies={"echo": "allow"})
    assert belt.dispatch("echo", {"text": "hi"}, ctx=ctx) == "echo: hi"


def test_dispatch_blanket_allow_ask_deny_unchanged_by_preset_addition():
    """Regression: preset is purely additive -- the three pre-existing
    blanket exec_policy values must behave exactly as before."""
    belt = ToolBelt([EchoTool()])
    assert belt.dispatch("echo", {"text": "hi"}, ctx=_ctx(exec_policy="allow")) == "echo: hi"
    assert belt.dispatch(
        "echo", {"text": "hi"}, ctx=_ctx(exec_policy="deny")
    ) == "[error] tool 'echo' blocked by exec_policy=deny"
    out = belt.dispatch("echo", {"text": "hi"}, ctx=_ctx(exec_policy="ask"))
    assert "needs approval" in out
    assert belt.dispatch("echo", {"text": "hi"}, ctx=_ctx(exec_policy="ask"), approve=lambda t, a: True) == "echo: hi"


def test_set_tool_policy_endpoint_persists_and_reflected_in_tools_list(monkeypatch):
    """Full round trip: switch a character into preset, set/clear a
    per-tool override via the new endpoint, and confirm GET .../tools
    reflects both the stored tool_policy and the resulting status."""
    pytest = __import__("pytest")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import pleiades.profiles as profiles_mod
    from pleiades.webui import create_app

    monkeypatch.setattr(profiles_mod, "Anamnesis", _FakeProfileAnamnesis)
    app = create_app()
    client = TestClient(app, base_url="http://127.0.0.1")

    resp = client.post("/api/profiles", json={"name": "preset-test-char"})
    assert resp.status_code == 200

    resp = client.put("/api/profiles/preset-test-char", json={"exec_policy": "preset"})
    assert resp.status_code == 200
    assert resp.json()["exec_policy"] == "preset"

    # No override yet -> vault (unsafe) defaults to "allow"/"available".
    resp = client.get("/api/profiles/preset-test-char/tools")
    by_name = {t["name"]: t for t in resp.json()["tools"]}
    assert by_name["vault"]["tool_policy"] == "allow"
    assert by_name["vault"]["status"] == "available"

    resp = client.post("/api/profiles/preset-test-char/tools/vault/policy",
                        json={"policy": "ask"})
    assert resp.status_code == 200
    assert resp.json()["policy"] == "ask"

    resp = client.get("/api/profiles/preset-test-char/tools")
    by_name = {t["name"]: t for t in resp.json()["tools"]}
    assert by_name["vault"]["tool_policy"] == "ask"
    assert by_name["vault"]["status"] == "needs_approval"
    # Safe tools are never gated, preset or otherwise.
    assert by_name["search"]["status"] == "available"

    # Clearing (policy: null) reverts to the "allow" default.
    resp = client.post("/api/profiles/preset-test-char/tools/vault/policy",
                        json={"policy": None})
    assert resp.status_code == 200
    assert resp.json()["policy"] == "allow"
    resp = client.get("/api/profiles/preset-test-char/tools")
    by_name = {t["name"]: t for t in resp.json()["tools"]}
    assert by_name["vault"]["tool_policy"] == "allow"

    resp = client.post("/api/profiles/preset-test-char/tools/not-a-real-tool/policy",
                        json={"policy": "ask"})
    assert resp.status_code == 404

    resp = client.post("/api/profiles/preset-test-char/tools/vault/policy",
                        json={"policy": "sometimes"})
    assert resp.status_code == 400

    client.delete("/api/profiles/preset-test-char")
