"""External-model escape hatch (Phase 9.4.4, docs/specs/2026-07-21-native-
inference-engine-design.md): `ModelManager.add(..., external_url=...)`
registers an already-running OpenAI-compatible server Pleiades never spawns,
stops, or health-manages via a process -- start()/state()/stop() all
special-case it. Tested against a real HTTP server (no real GGUF, no real
inference), same spirit as test_claude_mcp_integration.py's fake server.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pleiades.models import ModelManager, ModelError


class _FakeOpenAIServer:
    def __init__(self, healthy=True):
        self.healthy = healthy

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/v1/models" and self.server.owner.healthy:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"object":"list","data":[]}')
                else:
                    self.send_response(503)
                    self.end_headers()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.owner = self
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def fake_server():
    srv = _FakeOpenAIServer()
    try:
        yield srv
    finally:
        srv.stop()


def test_add_external_model_requires_no_local_file():
    mm = ModelManager()
    m = mm.add("ext1", external_url="http://10.0.0.5:9999/v1")
    assert m["external_url"] == "http://10.0.0.5:9999/v1"
    assert m["path"] == ""
    mm.remove("ext1", delete_file=False)


def test_base_url_returns_external_url_verbatim():
    mm = ModelManager()
    mm.add("ext2", external_url="http://10.0.0.5:9999/v1")
    assert mm.base_url("ext2") == "http://10.0.0.5:9999/v1"
    mm.remove("ext2", delete_file=False)


def test_state_and_start_reflect_real_reachability(fake_server):
    mm = ModelManager()
    mm.add("ext3", external_url=fake_server.url)
    assert mm.state("ext3") == "running"
    assert mm.is_running("ext3")
    assert mm.start("ext3") == fake_server.url
    mm.remove("ext3", delete_file=False)


def test_start_raises_when_external_server_unreachable():
    mm = ModelManager()
    mm.add("ext4", external_url="http://127.0.0.1:1/v1")  # nothing listens on port 1
    assert mm.state("ext4") == "stopped"
    with pytest.raises(ModelError, match="not reachable"):
        mm.start("ext4")
    mm.remove("ext4", delete_file=False)


def test_stop_is_a_safe_noop_for_external_model(fake_server):
    mm = ModelManager()
    mm.add("ext5", external_url=fake_server.url)
    mm.start("ext5")
    assert mm.stop("ext5") is False  # nothing of ours was ever spawned
    assert mm.state("ext5") == "running"  # the external server is unaffected
    mm.remove("ext5", delete_file=False)


def test_list_and_prune_do_not_drop_external_entries(fake_server):
    mm = ModelManager()
    mm.add("ext6", external_url=fake_server.url)
    names = {m["name"] for m in mm.list()}
    assert "ext6" in names
    # _load() runs _prune_missing() on every call -- confirm a second load
    # still has the entry (would be silently deleted pre-fix, since
    # Path("").is_file() is False).
    assert mm.get("ext6") is not None
    mm.remove("ext6", delete_file=False)


def test_cli_rejects_both_path_and_external_url_and_neither():
    # The mutual-exclusivity rule lives in cli.py's model_add, not
    # ModelManager.add() itself (which would just silently prefer
    # external_url if both were somehow passed) -- test it where it's
    # actually enforced.
    from click.testing import CliRunner

    from pleiades.cli import cli

    runner = CliRunner()
    r = runner.invoke(cli, ["model", "add", "ext8", "/some/path.gguf",
                            "--external-url", "http://x/v1"])
    assert r.exit_code != 0
    assert "exactly one" in r.output.lower()

    r = runner.invoke(cli, ["model", "add", "ext9"])
    assert r.exit_code != 0
    assert "exactly one" in r.output.lower()
