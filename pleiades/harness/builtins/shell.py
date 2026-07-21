"""Execution tools — run shell commands and Python. Always gated (side effects).

Layered safety, in order:
  1. The agent's approval gate (exec_policy: ask/allow/deny, harness/agent.py).
     Mandatory, unaffected by anything below -- "should this run at all?"
  2. This module's sandbox (harness/sandbox.py): a hard, unconditional floor
     against whole-system-destructive commands, plus a memory ceiling and
     (opt-in) filesystem/network scoping -- "if it runs, how much damage can
     it do?" See sandbox.py's module docstring for exactly what's real vs.
     best-effort on this platform.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from ..tools import tool
from .. import sandbox as _sb


@tool(tags=("shell",))
def run_shell(command: str, timeout: int = 120) -> str:
    """Run a shell command and return combined stdout/stderr.

    Runs inside the Pleiades sandbox (see sandbox.py): whole-system-destructive
    patterns are blocked unconditionally, and the process is killed if it
    exceeds the configured memory ceiling or this timeout.

    command: the command line to execute (runs in the system shell).
    timeout: seconds before the command is killed.
    """
    policy = _sb.load_policy()
    try:
        proc = _sb.run_sandboxed(command, shell=True, timeout=timeout, policy=policy)
    except _sb.SandboxViolation as e:
        return f"Error: {e}"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s."
    out = proc.stdout or ""
    tag = f"[exit {proc.returncode}]\n"
    return tag + (out[:40000] if out else "(no output)")


@tool(tags=("shell",))
def run_python(code: str, timeout: int = 120) -> str:
    """Execute a Python snippet in a fresh subprocess and return its output.

    Runs inside the Pleiades sandbox (see sandbox.py): destructive file ops
    and (if configured) network access outside the workspace root are denied
    at the interpreter level before your code runs, in addition to the
    memory/timeout limits applied to run_shell.

    code: Python source to run (use print() to produce output).
    timeout: seconds before the process is killed.
    """
    policy = _sb.load_policy()
    try:
        _sb.check_command_safety(code, policy)
    except _sb.SandboxViolation as e:
        return f"Error: {e}"
    wrapped = _sb.wrap_python_source(code, policy)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(wrapped)
        tmp = f.name
    try:
        proc = _sb.run_sandboxed([sys.executable, tmp], shell=False,
                                 timeout=timeout, policy=policy)
        out = proc.stdout or ""
        return f"[exit {proc.returncode}]\n" + (out[:40000] if out else "(no output)")
    except _sb.SandboxViolation as e:
        return f"Error: {e}"
    except subprocess.TimeoutExpired:
        return f"Error: code timed out after {timeout}s."
    finally:
        Path(tmp).unlink(missing_ok=True)
