"""MCPServer's init_timeout/call_timeout must actually be enforced -- a real
subprocess, not mocked innards, since the whole point of the fix is bounding
a genuinely blocking cross-thread read. See HANDOFF's 2026-07-25 correctness
audit, finding #5: a hung MCP server used to wedge mount_mcp_server()/any
tool call forever, with no way to recover short of killing the process."""

import sys
import time

import pytest

from pleiades.harness.mcp_client import MCPServer

# A "server" that reads one line off stdin (so it doesn't exit immediately,
# which would just be a closed-connection error, not a timeout) and then
# hangs forever without ever writing a response.
_HANGING_SERVER = "import sys; sys.stdin.readline(); import time; time.sleep(60)"

# A minimal real JSON-RPC responder: replies to "initialize" and "tools/list",
# enough for MCPServer's own handshake + list_tools() to succeed normally.
_ECHO_SERVER = r"""
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except Exception:
        continue
    if "id" not in msg:
        continue  # a notification, no response expected
    method = msg.get("method")
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "echo"}}
    elif method == "tools/list":
        result = {"tools": []}
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}) + "\n")
    sys.stdout.flush()
"""


def test_handshake_times_out_instead_of_hanging_forever():
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        MCPServer("hangtest", sys.executable, ["-c", _HANGING_SERVER], init_timeout=0.5)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, (
        f"took {elapsed:.1f}s -- init_timeout=0.5 should have fired, not the "
        "old unbounded readline()"
    )


def test_request_response_roundtrip_still_works():
    server = MCPServer("echotest", sys.executable, ["-c", _ECHO_SERVER],
                       init_timeout=5.0, call_timeout=5.0)
    try:
        assert server.list_tools() == []
    finally:
        server.close()


_ECHO_THEN_STALL_SERVER = r"""
import json, sys, time
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except Exception:
        continue
    if "id" not in msg:
        continue
    method = msg.get("method")
    if method == "tools/call":
        time.sleep(60)  # alive, just never answers this one
        continue
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "echo"}}
    elif method == "tools/list":
        result = {"tools": []}
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}) + "\n")
    sys.stdout.flush()
"""


def test_call_times_out_on_a_request_the_server_never_answers():
    # A server that completes the handshake (so construction succeeds) but
    # never answers tools/call -- the request-level timeout, not just the
    # handshake one, must also be enforced.
    server = MCPServer("stalltest", sys.executable, ["-c", _ECHO_THEN_STALL_SERVER],
                       init_timeout=5.0, call_timeout=0.5)
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            server.call("whatever", {})
        assert time.monotonic() - start < 5.0
    finally:
        server.close()
