from pleiades.tools import Tool, ToolBelt


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


def test_profile_tools_endpoint_reflects_exec_policy_and_usage():
    """Full round trip: create a profile, hit GET .../tools, and check that
    status honestly reflects exec_policy (deny -> blocked, ask -> needs
    approval, allow -> available) and that use_count/last_used come from
    real chats.tool_usage() data, not placeholders."""
    pytest = __import__("pytest")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pleiades import chats
    from pleiades.webui import create_app

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
