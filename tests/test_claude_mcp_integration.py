"""claude-mcp sidecar integration (services/claude-mcp) — the Python half.

Two integration paths, both tested against an in-process fake of the Node
server (no Node, no Claude subscription, no network beyond loopback):

  1. Backend tier: Settings.load() synthesizes a "claude-mcp" tier whose
     base_url embeds the bearer token as the /t/<token>/ path segment the real
     server accepts — because the "openai" backend posts with no Authorization
     header, the URL is the only per-tier channel that reaches the request.
  2. MCP tool source: Settings.mcp_servers grows an SSE entry (headers carry
     the token) and mount_configured_servers() turns the remote ask_claude
     into a registered Pleiades tool.

The fake mirrors the real server.mjs contract: /sse + /messages MCP transport,
/v1/chat/completions, auth via bearer header OR /t/<token>/ URL prefix.
"""

import json
import queue
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pleiades import config
from pleiades.harness import mcp_client
from pleiades.harness.llm import LLM
from pleiades.harness.mcp_client import (
    SSEMCPServer,
    close_all,
    mount_configured_servers,
    mount_sse_server,
)
from pleiades.harness.tools import registry


TOKEN = "sekrit-token-abc123"


# --------------------------------------------------------------------------- #
# A fake claude-mcp server (mirrors services/claude-mcp/server.mjs)
# --------------------------------------------------------------------------- #
class _FakeClaudeMCP:
    def __init__(self):
        self.sessions = {}        # sessionId -> queue of outbound SSE payloads
        self.chat_requests = []   # (path, body) of every /v1/chat/completions
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep test output clean
                pass

            def _split(self):
                path, _, query = self.path.partition("?")
                token = ""
                if path.startswith("/t/"):
                    token, _, tail = path[3:].partition("/")
                    path = "/" + tail
                return path, query, token

            def _authed(self, token):
                return (token == TOKEN
                        or self.headers.get("Authorization") == f"Bearer {TOKEN}")

            def do_GET(self):
                path, _, token = self._split()
                if path == "/sse":
                    if not self._authed(token):
                        self.send_response(401)
                        self.end_headers()
                        return
                    sid = uuid.uuid4().hex
                    q = queue.Queue()
                    fake.sessions[sid] = q
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    prefix = f"/t/{token}" if token else ""
                    self.wfile.write(
                        f"event: endpoint\ndata: {prefix}/messages?sessionId={sid}\n\n"
                        .encode())
                    self.wfile.flush()
                    try:
                        while True:
                            payload = q.get()
                            if payload is None:
                                break
                            self.wfile.write(
                                f"event: message\ndata: {payload}\n\n".encode())
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        fake.sessions.pop(sid, None)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                path, query, token = self._split()
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if not self._authed(token):
                    self.send_response(401)
                    self.end_headers()
                    return
                if path == "/messages":
                    sid = query.partition("sessionId=")[2].partition("&")[0]
                    q = fake.sessions.get(sid)
                    if q is None:
                        self.send_response(404)
                        self.end_headers()
                        return
                    msg = json.loads(body)
                    self.send_response(202)
                    self.end_headers()
                    self.wfile.write(b"Accepted")
                    if "id" in msg:                       # notifications get no reply
                        q.put(json.dumps(
                            {"jsonrpc": "2.0", "id": msg["id"],
                             "result": fake.rpc(msg)}))
                elif path == "/v1/chat/completions":
                    fake.chat_requests.append((self.path, json.loads(body)))
                    user = [m for m in json.loads(body).get("messages", [])
                            if m.get("role") == "user"]
                    text = "claude says: " + (user[-1]["content"] if user else "")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "id": "chatcmpl-1", "object": "chat.completion",
                        "model": "claude",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant",
                                                 "content": text}}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 5},
                    }).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def rpc(self, msg):
        method = msg.get("method")
        if method == "initialize":
            return {"protocolVersion": "2024-11-05", "capabilities": {},
                    "serverInfo": {"name": "fake-claude-mcp"}}
        if method == "tools/list":
            return {"tools": [{
                "name": "ask_claude",
                "description": "Ask Vern's Claude anything",
                "inputSchema": {"type": "object",
                                "properties": {"prompt": {"type": "string"}},
                                "required": ["prompt"]},
            }]}
        if method == "tools/call":
            prompt = msg["params"]["arguments"].get("prompt", "")
            return {"content": [{"type": "text",
                                 "text": f"claude says: {prompt}"}]}
        return {}

    def stop(self):
        for q in list(self.sessions.values()):
            q.put(None)
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def fake_server():
    srv = _FakeClaudeMCP()
    before = set(registry._tools)
    try:
        yield srv
    finally:
        close_all()
        for name in set(registry._tools) - before:   # unpollute the global registry
            registry._tools.pop(name, None)
        srv.stop()


def _settings(monkeypatch, tmp_path, url, token=TOKEN, tools=False):
    monkeypatch.setenv("PLEIADES_CLAUDE_MCP_URL", url)
    monkeypatch.setenv("PLEIADES_CLAUDE_MCP_TOKEN", token)
    monkeypatch.setenv("PLEIADES_CLAUDE_MCP_TOOLS", "1" if tools else "")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return config.Settings.load(root=tmp_path)


# --------------------------------------------------------------------------- #
# Config: tier synthesis + masking
# --------------------------------------------------------------------------- #
def test_tier_synthesized_from_env(monkeypatch, tmp_path):
    s = _settings(monkeypatch, tmp_path, "http://10.0.0.9:3456")
    t = s.tier("claude-mcp")
    assert t.backend == "openai"
    assert t.model == "claude"
    assert t.base_url == f"http://10.0.0.9:3456/t/{TOKEN}/v1"


def test_url_normalization_tolerates_pasted_v1_and_slash(monkeypatch, tmp_path):
    s = _settings(monkeypatch, tmp_path, "http://10.0.0.9:3456/v1/")
    assert s.claude_mcp_openai_url == f"http://10.0.0.9:3456/t/{TOKEN}/v1"
    assert s.claude_mcp_sse_url == "http://10.0.0.9:3456/sse"


def test_no_tier_without_url(monkeypatch, tmp_path):
    monkeypatch.delenv("PLEIADES_CLAUDE_MCP_URL", raising=False)
    monkeypatch.delenv("PLEIADES_CLAUDE_MCP_TOKEN", raising=False)
    s = config.Settings.load(root=tmp_path)
    assert "claude-mcp" not in s.tiers
    assert s.claude_mcp_openai_url == ""


def test_explicit_config_json_tier_wins(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "claude_mcp_url": "http://10.0.0.9:3456",
        "claude_mcp_token": TOKEN,
        "tiers": {"claude-mcp": {"backend": "openai", "model": "claude-custom",
                                 "base_url": "http://elsewhere/v1"}},
    }), encoding="utf-8")
    monkeypatch.delenv("PLEIADES_CLAUDE_MCP_URL", raising=False)
    monkeypatch.delenv("PLEIADES_CLAUDE_MCP_TOKEN", raising=False)
    s = config.Settings.load(root=tmp_path)
    assert s.tier("claude-mcp").model == "claude-custom"
    assert s.tier("claude-mcp").base_url == "http://elsewhere/v1"


def test_tier_with_unknown_key_survives(tmp_path, monkeypatch):
    # A typo'd extra field used to TypeError the whole tier away, silently
    # rerouting requests to default_tier.
    monkeypatch.delenv("PLEIADES_CLAUDE_MCP_URL", raising=False)
    (tmp_path / "config.json").write_text(json.dumps({
        "tiers": {"mytier": {"backend": "openai", "model": "m",
                             "max_tokenz": 123}},
    }), encoding="utf-8")
    s = config.Settings.load(root=tmp_path)
    assert "mytier" in s.tiers
    assert s.tiers["mytier"].model == "m"


def test_claude_mcp_tools_flag_adds_sse_entry(monkeypatch, tmp_path):
    s = _settings(monkeypatch, tmp_path, "http://10.0.0.9:3456", tools=True)
    entries = [e for e in s.mcp_servers if e.get("name") == "claude"]
    assert len(entries) == 1
    e = entries[0]
    assert e["url"] == "http://10.0.0.9:3456/sse"
    assert e["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert "ask_claude" in e["safe_tools"]
    # load() must be idempotent about it (no duplicate on a fresh load of the
    # same config, and no entry duplication when one already exists on disk).
    s2 = config.Settings.load(root=tmp_path)
    assert len([e for e in s2.mcp_servers if e.get("name") == "claude"]) == 1


def test_to_dict_masks_every_token_carrier(monkeypatch, tmp_path):
    s = _settings(monkeypatch, tmp_path, "http://10.0.0.9:3456", tools=True)
    d = s.to_dict()
    dumped = json.dumps(d)
    assert TOKEN not in dumped
    assert d["claude_mcp_token"] == "***"
    assert d["tiers"]["claude-mcp"]["base_url"] == "http://10.0.0.9:3456/t/***/v1"
    claude_entry = [e for e in d["mcp_servers"] if e.get("name") == "claude"][0]
    assert claude_entry["headers"]["Authorization"] == "***"


# --------------------------------------------------------------------------- #
# Backend tier, end to end: LLM -> fake claude-mcp over HTTP
# --------------------------------------------------------------------------- #
def test_llm_chat_reaches_claude_mcp_via_token_url(monkeypatch, tmp_path, fake_server):
    s = _settings(monkeypatch, tmp_path, fake_server.url)
    reply = LLM(s).chat([{"role": "user", "content": "hi there"}], [],
                        s.tier("claude-mcp"))
    assert reply.text == "claude says: hi there"
    assert reply.backend == "openai"
    assert reply.usage == {"input_tokens": 3, "output_tokens": 5}
    path, body = fake_server.chat_requests[0]
    assert path == f"/t/{TOKEN}/v1/chat/completions"     # token rode in the URL
    assert body["model"] == "claude"


def test_llm_chat_unauthorized_without_token(monkeypatch, tmp_path, fake_server):
    import urllib.error

    s = _settings(monkeypatch, tmp_path, fake_server.url, token="wrong-token")
    with pytest.raises(urllib.error.HTTPError):
        LLM(s).chat([{"role": "user", "content": "hi"}], [], s.tier("claude-mcp"))


# --------------------------------------------------------------------------- #
# MCP tool source: SSE client + configured mounting
# --------------------------------------------------------------------------- #
def test_sse_mount_registers_and_calls_ask_claude(fake_server):
    n = mount_sse_server("claude", f"{fake_server.url}/sse",
                         headers={"Authorization": f"Bearer {TOKEN}"},
                         safe_tools=["ask_claude"])
    assert n == 1
    tool = registry.get("mcp.claude.ask_claude")
    assert tool is not None
    assert tool.safe                                     # whitelisted read-only
    assert tool.func(prompt="ping") == "claude says: ping"


def test_sse_client_rejects_bad_token_with_clear_error(fake_server):
    with pytest.raises(RuntimeError, match="401|bearer token"):
        SSEMCPServer("claude", f"{fake_server.url}/sse",
                     headers={"Authorization": "Bearer nope"}, init_timeout=5)


def test_mount_configured_servers_from_settings(monkeypatch, tmp_path, fake_server):
    s = _settings(monkeypatch, tmp_path, fake_server.url, tools=True)
    # Poison the list with junk entries: a dead server, a malformed entry, and
    # a non-dict — none may abort the good mount.
    s.mcp_servers.insert(0, {"name": "dead", "url": "http://127.0.0.1:1/sse"})
    s.mcp_servers.insert(0, {"name": "no-transport"})
    s.mcp_servers.insert(0, "garbage")
    assert mount_configured_servers(s) == 1
    assert registry.get("mcp.claude.ask_claude") is not None


def test_mount_configured_servers_empty_is_noop():
    s = config.Settings()
    assert mount_configured_servers(s) == 0
