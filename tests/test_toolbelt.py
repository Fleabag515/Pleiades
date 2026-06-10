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
