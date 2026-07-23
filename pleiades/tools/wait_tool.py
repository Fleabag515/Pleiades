"""Wait tool: pause a fixed number of seconds before continuing the turn.

For real load latency only — e.g. a form was just submitted and the result
page needs a moment to render, or a background process needs a beat to
finish — not a general-purpose delay/sleep hack. Hard-capped so a bad or
manipulated `seconds` value can't stall a whole turn.
"""

from __future__ import annotations

import time
from typing import Any

from . import Tool, ToolContext

MAX_WAIT_SECONDS = 120
_POLL_INTERVAL = 1.0  # seconds between should_stop checks, so an emergency
                      # stop only ever adds up to ~1s of extra latency


class WaitTool(Tool):
    safe = True  # no side effects — just consumes wall-clock time
    name = "wait"
    description = (
        "Pause for a fixed number of seconds before continuing your turn. Use this "
        "ONLY when you already know something needs real wall-clock time to happen in "
        "the background — e.g. you just submitted a form and the result page is still "
        "loading, or a process you started needs a moment to finish. Do NOT use it as a "
        "generic delay, to 'slow down', or as a substitute for actually checking whether "
        f"something is ready. Capped at {MAX_WAIT_SECONDS} seconds per call; call it "
        "again if you need to wait longer."
    )
    parameters = {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "number",
                "description": f"How long to pause, in seconds (1-{MAX_WAIT_SECONDS}).",
            },
        },
        "required": ["seconds"],
    }

    def run(self, ctx: ToolContext, seconds: Any = None) -> str:
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return f"[wait error] 'seconds' must be a number, got {seconds!r}."
        if seconds != seconds or seconds in (float("inf"), float("-inf")):  # NaN/inf
            return "[wait error] 'seconds' must be a finite number."
        if seconds <= 0:
            return "[wait error] 'seconds' must be positive."
        if seconds > MAX_WAIT_SECONDS:
            return (f"[wait error] {seconds:g}s exceeds the {MAX_WAIT_SECONDS}s cap per "
                     f"call. Request at most {MAX_WAIT_SECONDS}s — call wait again if you "
                     "need longer.")

        # Cooperative cancellation: honored if ctx exposes a should_stop() callable
        # (e.g. a chat's emergency-stop button), but not required — plain ctx=None
        # or a ctx without the attribute just waits out the full duration.
        should_stop = getattr(ctx, "should_stop", None)
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return f"Waited {seconds:g}s."
            if callable(should_stop) and should_stop():
                elapsed = seconds - remaining
                return f"Wait cancelled after {elapsed:.1f}s of {seconds:g}s (stop requested)."
            time.sleep(min(_POLL_INTERVAL, remaining))
