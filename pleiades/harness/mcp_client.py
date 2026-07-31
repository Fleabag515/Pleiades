"""
mcp_client.py — let Pleiades speak MCP as a CLIENT.

This is the multiplier. Instead of hand-writing every capability, Pleiades mounts
existing MCP servers and inherits their tools. The user's NyzKh fleet alone
(Discord, browser-use, deep-research, vision, image-gen, email, github, memory,
keyless web) becomes Pleiades tools the moment they're mounted — and any future
MCP server does too.

Two transports, both minimal, synchronous JSON-RPC 2.0 and dependency-free
(matching Pleiades's stdlib-only posture — the Ollama backend is urllib too):

  * stdio  — a spawned subprocess, newline-delimited messages.
        mount_mcp_server("nyzkh-web", command="python", args=[".../server.py"])
  * SSE    — a remote HTTP server (GET /sse stream + POSTed messages), e.g. the
    services/claude-mcp sidecar that exposes a Claude subscription's ask_claude.
        mount_sse_server("claude", "http://host:3456/sse",
                         headers={"Authorization": "Bearer <token>"})

`mount_configured_servers(cfg)` mounts every entry in Settings.mcp_servers —
dicts with either {"name", "command", "args", ...} or {"name", "url",
"headers", ...} — skipping (with a stderr warning) any that fail, so one dead
server never blocks startup.

Mounted MCP tools are gated by exec_policy by default (external side effects).
Pass safe_tools=[...] to whitelist read-only ones.
"""

from __future__ import annotations

import http.client
import json
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Any
from urllib.parse import urljoin, urlsplit

from .tools import Tool, registry


class _JsonRpcSession:
    """Shared request/response machinery over an inbound message queue.

    Subclasses provide `_transport_send(msg)` (deliver one JSON-RPC dict) and
    `close()`, and feed inbound JSON strings (one message each) into
    `self._out_q`, ending with a `None` sentinel when the transport dies.
    """

    def __init__(self, name: str, call_timeout: float) -> None:
        self.name = name
        self._id = 0
        self._lock = threading.Lock()
        self.call_timeout = call_timeout
        self._out_q: "queue.Queue[str | None]" = queue.Queue()

    # -- transport hook -------------------------------------------------
    def _transport_send(self, msg: dict) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- JSON-RPC plumbing ---------------------------------------------
    def _send(self, method: str, params: dict | None = None,
              notify: bool = False) -> int | None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        rid = None
        if not notify:
            self._id += 1
            rid = self._id
            msg["id"] = rid
        self._transport_send(msg)
        return rid

    def _read_result(self, rid: int, timeout: float) -> dict:
        """Read queued messages until the response with id==rid arrives (skip
        notifications), or raise TimeoutError once `timeout` elapses."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"MCP server '{self.name}' did not respond within {timeout}s")
            try:
                line = self._out_q.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(
                    f"MCP server '{self.name}' did not respond within {timeout}s") from None
            if line is None:
                raise RuntimeError(f"MCP server '{self.name}' closed the connection.")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # server log noise on stdout — ignore
            if msg.get("id") != rid:
                continue  # a notification or an unrelated (e.g. timed-out) response
            if "error" in msg:
                raise RuntimeError(f"MCP error from '{self.name}': {msg['error']}")
            return msg.get("result", {})

    def _request(self, method: str, params: dict | None = None,
                 timeout: float | None = None) -> dict:
        with self._lock:
            rid = self._send(method, params)
            return self._read_result(rid, timeout if timeout is not None  # type: ignore[arg-type]
                                      else self.call_timeout)

    # -- protocol -------------------------------------------------------
    def _handshake(self, timeout: float) -> None:
        self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pleiades", "version": "0.1.0"},
        }, timeout=timeout)
        self._send("notifications/initialized", notify=True)

    def list_tools(self) -> list[dict]:
        return self._request("tools/list").get("tools", [])

    def call(self, tool_name: str, arguments: dict) -> str:
        result = self._request("tools/call",
                               {"name": tool_name, "arguments": arguments})
        # MCP returns content blocks; flatten text for the agent loop
        parts = []
        for block in result.get("content", []):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(json.dumps(block))
        text = "\n".join(parts) if parts else json.dumps(result)
        if result.get("isError"):
            return f"[tool error] {text}"
        return text


class MCPServer(_JsonRpcSession):
    """A spawned MCP stdio server; one subprocess, request/response JSON-RPC."""

    def __init__(self, name: str, command: str, args: list[str] | None = None,
                 cwd: str | None = None, env: dict | None = None,
                 init_timeout: float = 30.0, call_timeout: float = 60.0) -> None:
        super().__init__(name, call_timeout)
        self.proc = subprocess.Popen(
            [command, *(args or [])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
            bufsize=1, cwd=cwd, env=env,
        )
        # drain stderr so the pipe never blocks the server
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        # stdout.readline() itself has no timeout parameter, and select()
        # doesn't work on pipes on Windows -- so a plain blocking read on
        # the caller's own thread can never be bounded. Instead a
        # dedicated thread does the blocking read and hands lines off via
        # a queue, which DOES support a timeout on get() cross-platform.
        # This is what makes a hung/deadlocked MCP server (bad interpreter
        # path, a missing dep that prints to stderr and stalls, etc.) a
        # bounded, recoverable error instead of hanging this call --
        # and mount_mcp_server(), which calls the handshake below -- forever.
        threading.Thread(target=self._read_stdout_loop, daemon=True).start()
        try:
            self._handshake(init_timeout)
        except BaseException:
            # A failed/timed-out handshake must not leak the child process.
            self.close()
            raise

    def _drain_stderr(self) -> None:
        try:
            for _ in self.proc.stderr:  # type: ignore[union-attr]
                pass
        except Exception:
            pass

    def _read_stdout_loop(self) -> None:
        try:
            for line in self.proc.stdout:  # type: ignore[union-attr]
                self._out_q.put(line)
        except Exception:
            pass
        finally:
            self._out_q.put(None)  # sentinel: pipe closed / process exited

    def _transport_send(self, msg: dict) -> None:
        line = json.dumps(msg) + "\n"
        self.proc.stdin.write(line)      # type: ignore[union-attr]
        self.proc.stdin.flush()          # type: ignore[union-attr]

    def close(self) -> None:
        try:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()   # SIGTERM ignored — escalate so it can't linger
                self.proc.wait(timeout=1)
        except Exception:
            pass


class SSEMCPServer(_JsonRpcSession):
    """A remote MCP server over the HTTP+SSE transport (e.g. services/claude-mcp).

    GET <url> opens the event stream; the server's first `endpoint` event names
    where to POST JSON-RPC messages, and responses come back as `message`
    events on the stream. Auth (e.g. a bearer token) travels in `headers` on
    both the stream request and every POST.
    """

    def __init__(self, name: str, url: str, headers: dict | None = None,
                 init_timeout: float = 30.0, call_timeout: float = 60.0) -> None:
        super().__init__(name, call_timeout)
        self.url = url
        self._headers = dict(headers or {})
        self._endpoint: str | None = None
        self._endpoint_evt = threading.Event()
        self._closed = False

        parts = urlsplit(url)
        conn_cls = (http.client.HTTPSConnection if parts.scheme == "https"
                    else http.client.HTTPConnection)
        # http.client (not urlopen) for the stream: urlopen's timeout applies to
        # EVERY blocking read, and an SSE stream legitimately idles between
        # events — it would kill the connection mid-session. Here the timeout
        # bounds only connect + the first response, then the socket timeout is
        # cleared for the long-lived read.
        self._conn = conn_cls(parts.hostname or "", parts.port,
                              timeout=init_timeout)
        path = parts.path or "/"
        if parts.query:
            path += f"?{parts.query}"
        try:
            self._conn.request("GET", path, headers={
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                **self._headers,
            })
            self._resp = self._conn.getresponse()
            if self._resp.status != 200:
                hint = " — check the bearer token" if self._resp.status in (401, 403) else ""
                raise RuntimeError(
                    f"MCP SSE server '{name}' refused the stream: "
                    f"HTTP {self._resp.status}{hint}")
            if self._conn.sock is not None:
                self._conn.sock.settimeout(None)
            threading.Thread(target=self._read_sse_loop, daemon=True).start()
            if not self._endpoint_evt.wait(init_timeout):
                raise TimeoutError(
                    f"MCP SSE server '{name}' sent no endpoint event within {init_timeout}s")
            self._handshake(init_timeout)
        except BaseException:
            self.close()
            raise

    def _read_sse_loop(self) -> None:
        event, data = "", []
        try:
            while True:
                raw = self._resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if not line:                      # blank line = dispatch event
                    self._dispatch(event or "message", "\n".join(data))
                    event, data = "", []
                elif line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())
                # ":" comment/keepalive lines and unknown fields are ignored
        except Exception:
            pass
        finally:
            self._out_q.put(None)  # sentinel: stream closed

    def _dispatch(self, event: str, data: str) -> None:
        if event == "endpoint":
            self._endpoint = urljoin(self.url, data)
            self._endpoint_evt.set()
        elif event == "message" and data:
            self._out_q.put(data)

    def _transport_send(self, msg: dict) -> None:
        if not self._endpoint:
            raise RuntimeError(f"MCP SSE server '{self.name}' has no message endpoint yet")
        req = urllib.request.Request(
            self._endpoint, data=json.dumps(msg).encode("utf-8"),
            headers={"Content-Type": "application/json", **self._headers},
            method="POST",
        )
        # POSTs are short-lived (the server just enqueues the frame); the
        # actual result arrives on the SSE stream and is awaited separately.
        with urllib.request.urlopen(req, timeout=self.call_timeout) as r:
            r.read()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for obj in (getattr(self, "_resp", None), getattr(self, "_conn", None)):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass


# track live servers so they aren't garbage-collected / can be closed
_SERVERS: dict[str, _JsonRpcSession] = {}


def _adopt(server: _JsonRpcSession) -> None:
    """Register a live server, closing any previous mount under the same name."""
    old = _SERVERS.pop(server.name, None)
    if old is not None:
        old.close()
    _SERVERS[server.name] = server


def _register_tools(server: _JsonRpcSession, safe_tools: list[str] | None,
                    prefix: bool) -> int:
    """Register each of a mounted server's tools into the Pleiades registry.

    Tool names are namespaced as `mcp.<server>.<tool>` (prefix=True) to avoid
    collisions across servers. Returns the number of tools registered.
    """
    safe = set(safe_tools or [])
    count = 0
    for spec in server.list_tools():
        tname = spec["name"]
        full = f"mcp.{server.name}.{tname}" if prefix else tname

        def _make(_server: _JsonRpcSession, _tname: str):
            def _call(**kwargs):
                return _server.call(_tname, kwargs)
            return _call

        tool = Tool(
            name=full,
            description=spec.get("description", tname),
            func=_make(server, tname),
            schema=spec.get("inputSchema",
                            {"type": "object", "properties": {}}),
            safe=tname in safe,
            tags=("mcp", server.name),
        )
        registry.add(tool)
        count += 1
    return count


def mount_mcp_server(name: str, command: str, args: list[str] | None = None,
                     cwd: str | None = None, env: dict | None = None,
                     safe_tools: list[str] | None = None,
                     prefix: bool = True) -> int:
    """Spawn a stdio MCP server and register its tools. Returns the tool count."""
    server = MCPServer(name, command, args, cwd=cwd, env=env)
    _adopt(server)
    return _register_tools(server, safe_tools, prefix)


def mount_sse_server(name: str, url: str, headers: dict | None = None,
                     safe_tools: list[str] | None = None,
                     prefix: bool = True) -> int:
    """Connect to a remote SSE MCP server and register its tools.

    e.g. the claude-mcp sidecar:
        mount_sse_server("claude", "http://<tailscale-ip>:3456/sse",
                         headers={"Authorization": "Bearer <token>"},
                         safe_tools=["ask_claude"])
    """
    server = SSEMCPServer(name, url, headers=headers)
    _adopt(server)
    return _register_tools(server, safe_tools, prefix)


def mount_configured_servers(cfg) -> int:
    """Mount every entry in Settings.mcp_servers; returns total tools mounted.

    Entries are dicts: {"name", "command", "args"?, "cwd"?, "env"?} for stdio,
    {"name", "url", "headers"?} for SSE; both accept "safe_tools" and "prefix".
    A failing or malformed entry is skipped with a warning — one dead server
    must never block startup.
    """
    total = 0
    for entry in getattr(cfg, "mcp_servers", None) or []:
        if not isinstance(entry, dict):
            print(f"[mcp] skipping malformed mcp_servers entry: {entry!r}",
                  file=sys.stderr)
            continue
        name = entry.get("name") or ""
        try:
            if entry.get("url"):
                total += mount_sse_server(
                    name or "sse", entry["url"], headers=entry.get("headers"),
                    safe_tools=entry.get("safe_tools"),
                    prefix=entry.get("prefix", True))
            elif entry.get("command"):
                total += mount_mcp_server(
                    name or entry["command"], entry["command"],
                    args=entry.get("args"), cwd=entry.get("cwd"),
                    env=entry.get("env"), safe_tools=entry.get("safe_tools"),
                    prefix=entry.get("prefix", True))
            else:
                print(f"[mcp] mcp_servers entry {name or entry!r} has neither "
                      "'url' nor 'command' — skipped", file=sys.stderr)
        except Exception as e:
            print(f"[mcp] could not mount '{name}': {e}", file=sys.stderr)
    return total


def close_all() -> None:
    for s in _SERVERS.values():
        s.close()
    _SERVERS.clear()
