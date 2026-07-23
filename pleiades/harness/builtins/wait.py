"""wait.py — pause a fixed number of seconds before continuing.

For real load latency only — e.g. you just submitted a form and the result
page needs a moment to render, or a process you started needs a beat to
finish — not a general-purpose delay/sleep hack. Hard-capped so a bad value
can't stall the agent loop indefinitely. See also wait_for_event (events.py)
for blocking on an external signal instead of a clock.
"""

from __future__ import annotations

import time

from ..tools import tool

MAX_WAIT_SECONDS = 120
_POLL_INTERVAL = 1.0  # sleep in small chunks rather than one long call


@tool(safe=True, tags=("system", "read"))
def wait(seconds: float) -> str:
    """Pause for a fixed number of seconds before continuing. Use ONLY when you already know something needs real wall-clock time to happen in the background — e.g. you just submitted a form and the page is still loading, or a process you started needs a moment to finish. Do NOT use it as a generic delay or as a substitute for actually checking whether something is ready.

    seconds: how long to pause, in seconds (1-120; capped to prevent an indefinite hang — call it again if you need to wait longer).
    """
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return f"Error: 'seconds' must be a number, got {seconds!r}."
    if seconds != seconds or seconds in (float("inf"), float("-inf")):  # NaN/inf
        return "Error: 'seconds' must be a finite number."
    if seconds <= 0:
        return "Error: 'seconds' must be positive."
    if seconds > MAX_WAIT_SECONDS:
        return (f"Error: {seconds:g}s exceeds the {MAX_WAIT_SECONDS}s cap per call. "
                f"Request at most {MAX_WAIT_SECONDS}s — call wait again if you need "
                "longer.")

    remaining = seconds
    while remaining > 0:
        chunk = min(_POLL_INTERVAL, remaining)
        time.sleep(chunk)
        remaining -= chunk
    return f"Waited {seconds:g}s."
