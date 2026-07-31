"""
process.py — background process management.

`run_shell` and `run_python` block until the command finishes. That's fine for
quick scripts but useless for long-running things: dev servers, watchers, builds
that take minutes, concurrent tasks. This module adds a persistent process table:

    pid = start_process("python -m http.server 8080")
    read_process_output(pid)        # stream stdout/stderr so far
    read_process_output(pid)        # poll again later
    kill_process(pid)

Processes are identified by an integer PID key that the agent holds in working
memory. The table is in-process; processes survive across agent turns but not
across interpreter restarts (by design — no orphaned subprocesses on crash).

Design constraint: all output is buffered in a background thread into a capped
deque so read_process_output is instant and non-blocking. The buffer holds the
last `_BUFFER_LINES` lines; older output is discarded.

Sandbox note: start_process is the same risk class as run_shell (harness/
builtins/shell.py) — an arbitrary command, possibly running unattended for a
long time. It gets the same hard, unconditional destructive-command floor and
the same memory watchdog from harness/sandbox.py. It does not get a wall-clock
timeout (by design — that's the point of a background process), so the memory
ceiling is the main backstop against a runaway background job.
"""

from __future__ import annotations

import collections
import os
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Deque

from ..tools import tool
from .. import sandbox as _sb

_BUFFER_LINES = 2000   # lines of stdout+stderr to keep per process


@dataclass
class _Proc:
    pid: int                          # our handle (== proc.pid for real processes)
    command: str
    proc: subprocess.Popen
    buffer: Deque[str] = field(default_factory=lambda: collections.deque(maxlen=_BUFFER_LINES))
    read_pos: int = 0                 # lines already returned by read_process_output
    done: bool = False
    returncode: int | None = None
    watchdog: "_sb._MemWatchdog | None" = None
    _lock: threading.Lock = field(default_factory=threading.Lock)


_TABLE: dict[int, _Proc] = {}
_TABLE_LOCK = threading.Lock()


def _reader(p: _Proc) -> None:
    """Background thread: drain stdout+stderr into the ring buffer."""
    try:
        for line in p.proc.stdout:  # type: ignore[union-attr]
            with p._lock:
                p.buffer.append(line.rstrip("\n"))
    finally:
        p.proc.wait()
        with p._lock:
            p.done = True
            p.returncode = p.proc.returncode
        if p.watchdog is not None:
            p.watchdog.stop()


@tool(tags=("process",))
def start_process(command: str, cwd: str = "") -> str:
    """Start a command in the background and return its process ID.

    The process runs independently; use read_process_output to see its output,
    kill_process to stop it, and list_processes to see all running processes.
    stdout and stderr are merged.

    Runs inside the Pleiades sandbox (see sandbox.py): whole-system-destructive
    patterns are blocked unconditionally, and the process is killed if it
    exceeds the configured memory ceiling (there is no wall-clock timeout for
    background processes — that's the point of start_process).

    command: shell command to run (e.g. "python -m http.server 8080").
    cwd: working directory for the process (default: current directory).
    """
    policy = _sb.load_policy()
    try:
        _sb.check_command_safety(command, policy)
    except _sb.SandboxViolation as e:
        return f"Error: {e}"

    # shell=True means `proc` is the shell wrapper (/bin/sh -c … or cmd /c …),
    # not the command itself. Detach it into its own process group so
    # kill_process can take down the whole tree — terminate() on the wrapper
    # alone orphans anything the shell spawned (pipelines, `npm run dev`,
    # `a && b`), which keeps running while we report a clean stop.
    group_kwargs: dict = {}
    if os.name == "nt":
        group_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        group_kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(
            command, shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            cwd=cwd or None,
            **group_kwargs,
        )
    except Exception as e:
        return f"Error starting process: {e}"

    p = _Proc(pid=proc.pid, command=command, proc=proc)
    if policy.enabled and policy.mem_mb:
        p.watchdog = _sb.attach_watchdog(proc.pid, policy)
    t = threading.Thread(target=_reader, args=(p,), daemon=True)
    t.start()

    with _TABLE_LOCK:
        _TABLE[proc.pid] = p

    return f"Started process PID={proc.pid}: {command!r}"


@tool(safe=True, tags=("process", "read"))
def read_process_output(pid: int, tail: int = 80) -> str:
    """Read new output from a background process since the last call.

    Returns lines not yet returned (incremental), up to `tail` lines. If the
    process has exited, reports the exit code.

    pid: the process ID returned by start_process.
    tail: max lines to return (default 80; use 0 for all buffered since last read).
    """
    with _TABLE_LOCK:
        p = _TABLE.get(pid)
    if p is None:
        return f"Error: no such process PID={pid}. Use list_processes."

    with p._lock:
        buf = list(p.buffer)
        new_lines = buf[p.read_pos:]
        p.read_pos = len(buf)
        done = p.done
        rc = p.returncode
        killed_for_memory = p.watchdog.triggered if p.watchdog else False

    if tail > 0:
        new_lines = new_lines[-tail:]

    out = "\n".join(new_lines) if new_lines else "(no new output)"
    if killed_for_memory:
        out += "\n[killed: exceeded sandbox memory ceiling]"
    elif done:
        out += f"\n[process exited with code {rc}]"
    return out


@tool(safe=True, tags=("process", "read"))
def list_processes() -> str:
    """List all background processes managed by Pleiades (running and recently exited)."""
    with _TABLE_LOCK:
        procs = list(_TABLE.values())
    if not procs:
        return "(no background processes)"
    rows = []
    for p in procs:
        status = f"exited({p.returncode})" if p.done else "running"
        rows.append(f"  PID={p.pid}  [{status}]  {p.command[:80]!r}")
    return "\n".join(rows)


@tool(tags=("process",))
def kill_process(pid: int) -> str:
    """Terminate a background process. GATED — requires approval.

    pid: the process ID returned by start_process.
    """
    with _TABLE_LOCK:
        p = _TABLE.get(pid)
    if p is None:
        return f"Error: no such process PID={pid}."
    if p.done:
        return f"Process PID={pid} already exited (code {p.returncode})."
    # Signal the whole process group/tree, not just the shell wrapper (see
    # start_process: shell=True + its own group). While p.done is False the
    # wrapper is unreaped, so its pid — the group id on POSIX — can't have
    # been recycled and the group kill can't hit an unrelated process.
    try:
        if os.name == "nt":
            # taskkill /T walks the tree; TerminateProcess on cmd.exe wouldn't.
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.proc.pid)],
                           capture_output=True)
        else:
            os.killpg(p.proc.pid, signal.SIGTERM)
        p.proc.wait(timeout=5)
    except Exception:
        try:
            if os.name == "nt":
                p.proc.kill()
            else:
                os.killpg(p.proc.pid, signal.SIGKILL)
            p.proc.wait(timeout=5)
        except Exception:
            pass
    if p.watchdog is not None:
        p.watchdog.stop()
    return f"Terminated PID={pid}."


@tool(tags=("process",))
def send_stdin(pid: int, text: str) -> str:
    """Send text to a background process's stdin — for interactive programs.

    pid: the process ID.
    text: text to send (a newline is appended automatically).
    """
    with _TABLE_LOCK:
        p = _TABLE.get(pid)
    if p is None:
        return f"Error: no such process PID={pid}."
    if p.done:
        return f"Error: process PID={pid} has already exited."
    try:
        p.proc.stdin.write(text + "\n")  # type: ignore[union-attr]
        p.proc.stdin.flush()             # type: ignore[union-attr]
    except Exception as e:
        return f"Error writing to stdin: {e}"
    return f"Sent to PID={pid}."
