"""
mcp_client.py — let Pleiades speak MCP as a CLIENT.

This is the multiplier. Instead of hand-writing every capability, Pleiades mounts
existing MCP servers and inherits their tools. The user's NyzKh fleet alone
(Discord, browser-use, deep-research, vision, image-gen, email, github, memory,
keyless web) becomes Pleiades tools the moment they're mounted — and any future
MCP server does too.

Implemented as a minimal, synchronous JSON-RPC 2.0 client over a subprocess's
stdio (newline-delimited messages — the MCP stdio transport). No extra deps; it
matches Pleiades's stdlib-only posture (the Ollama backend is urllib too).

    n = mount_mcp_server("nyzkh-web",
                         command="python",
                         args=["C:/.../nyzkh_web_server.py"])
    # -> registers mcp.nyzkh-web.web_search, mcp.nyzkh-web.web_fetch

Mounted MCP tools are gated by exec_policy by default (external side effects).
Pass safe_tools=[...] to whitelist read-only ones.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from typing import Any

from .tools import Tool, registry


class MCPServer:
    """A spawned MCP stdio server; one subprocess, request/response JSON-RPC."""

    def __init__(self, name: str, command: str, args: list[str] | None = None,
                 cwd: str | None = None, env: dict | None = None,
                 init_timeout: float = 30.0, call_timeout: float = 60.0) -> None:
        self.name = name
        self._id = 0
        self._lock = threading.Lock()
        self.call_timeout = call_timeout
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
        self._out_q: "queue.Queue[str | None]" = queue.Queue()
        threading.Thread(target=self._read_stdout_loop, daemon=True).start()
        self._handshake(init_timeout)

    # -- JSON-RPC plumbing ---------------------------------------------
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
        line = json.dumps(msg) + "\n"
        self.proc.stdin.write(line)      # type: ignore[union-attr]
        self.proc.stdin.flush()          # type: ignore[union-attr]
        return rid

    def _read_result(self, rid: int, timeout: float) -> dict:
        """Read queued lines until the response with id==rid arrives (skip
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

    def close(self) -> None:
        try:
            self.proc.terminate()
        except Exception:
            pass


# track live servers so they aren't garbage-collected / can be closed
_SERVERS: dict[str, MCPServer] = {}


def mount_mcp_server(name: str, command: str, args: list[str] | None = None,
                     cwd: str | None = None, env: dict | None = None,
                     safe_tools: list[str] | None = None,
                     prefix: bool = True) -> int:
    """Spawn an MCP server and register each of its tools into Pleiades.

    Tool names are namespaced as `mcp.<server>.<tool>` (prefix=True) to avoid
    collisions across servers. Returns the number of tools mounted.
    """
    server = MCPServer(name, command, args, cwd=cwd, env=env)
    _SERVERS[name] = server
    safe = set(safe_tools or [])
    count = 0
    for spec in server.list_tools():
        tname = spec["name"]
        full = f"mcp.{name}.{tname}" if prefix else tname

        def _make(_server: MCPServer, _tname: str):
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
            tags=("mcp", name),
        )
        registry.add(tool)
        count += 1
    return count


def close_all() -> None:
    for s in _SERVERS.values():
        s.close()
    _SERVERS.clear()
