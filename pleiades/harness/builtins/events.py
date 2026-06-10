"""Event tools — receive webhooks and react to them.

Turns the agent from poll-only into reactive: GitHub pushes, CI results,
production alerts, or any `curl -X POST` can wake it up.

    start_webhook_listener(port=8765, secret="hunter2")
    wait_for_event("ci.failed", timeout=600)   # blocks until it arrives
    list_events()                              # peek at the buffer

Pure stdlib: ThreadingHTTPServer in a daemon thread + an in-process queue.
POST any JSON to http://host:port/ — a "type" field enables filtered waits.
If a secret is set, requests must carry it in the X-Webhook-Secret header.
"""

from __future__ import annotations

import json
import queue
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..tools import tool

_STATE: dict = {"server": None, "thread": None, "secret": "",
                "queue": queue.Queue(), "log": []}
_LOG_CAP = 200


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        secret = _STATE["secret"]
        if secret and self.headers.get("X-Webhook-Secret") != secret:
            self.send_response(403)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            body = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            body = {"raw": raw}
        event = {
            "type": body.get("type", "") if isinstance(body, dict) else "",
            "path": self.path,
            "body": body,
            "received": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _STATE["queue"].put(event)
        _STATE["log"].append(event)
        del _STATE["log"][:-_LOG_CAP]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *args):  # silence default stderr noise
        pass


@tool(tags=("events",))
def start_webhook_listener(port: int = 8765, secret: str = "") -> str:
    """Start a local HTTP listener that buffers incoming webhook events. POST JSON to it; include a "type" field to enable filtered waits.

    port: TCP port to listen on (localhost-reachable; expose at your own risk).
    secret: if set, requests must send it in the X-Webhook-Secret header.
    """
    if _STATE["server"] is not None:
        return f"Error: listener already running on port {_STATE['port']}; stop it first."
    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    except OSError as e:
        return f"Error: cannot bind port {port}: {e}"
    _STATE.update(server=server, secret=secret, port=port)
    t = threading.Thread(target=server.serve_forever, daemon=True,
                         name=f"pleiades-webhook-{port}")
    _STATE["thread"] = t
    t.start()
    guard = " (secret required)" if secret else ""
    return f"Webhook listener running on port {port}{guard}. POST JSON to receive events."


@tool(tags=("events",))
def stop_webhook_listener() -> str:
    """Stop the webhook listener. Buffered events remain readable via list_events."""
    server = _STATE["server"]
    if server is None:
        return "No listener running."
    server.shutdown()
    server.server_close()
    _STATE.update(server=None, thread=None)
    return "Webhook listener stopped."


@tool(safe=True, tags=("events", "read"))
def wait_for_event(event_type: str = "", timeout: int = 300) -> str:
    """Block until a matching event arrives (or timeout). Consumes the event.

    event_type: only return events whose "type" field equals this (empty = any).
    timeout: max seconds to wait.
    """
    if _STATE["server"] is None:
        return "Error: no listener running — call start_webhook_listener first."
    import time
    deadline = time.monotonic() + timeout
    skipped = []
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return f"Timeout: no '{event_type or 'any'}' event within {timeout}s."
            try:
                ev = _STATE["queue"].get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if not event_type or ev["type"] == event_type:
                return json.dumps(ev, indent=2)[:20_000]
            skipped.append(ev)
    finally:
        for ev in skipped:           # put non-matching events back
            _STATE["queue"].put(ev)


@tool(safe=True, tags=("events", "read"))
def list_events(limit: int = 20) -> str:
    """Show the most recent received events without consuming them.

    limit: max number of events to show.
    """
    log = _STATE["log"]
    if not log:
        status = "listener running" if _STATE["server"] else "no listener"
        return f"(no events received — {status})"
    out = [f"- [{e['received']}] type={e['type'] or '?'} path={e['path']} "
           f"body={json.dumps(e['body'])[:120]}" for e in log[-limit:]]
    return "\n".join(out)
