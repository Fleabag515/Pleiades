"""Execution tools — run shell commands and Python. Always gated (side effects)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from ..tools import tool


@tool(tags=("shell",))
def run_shell(command: str, timeout: int = 120) -> str:
    """Run a shell command and return combined stdout/stderr.

    command: the command line to execute (runs in the system shell).
    timeout: seconds before the command is killed.
    """
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s."
    out = (proc.stdout or "") + (proc.stderr or "")
    tag = f"[exit {proc.returncode}]\n"
    return tag + (out[:40000] if out else "(no output)")


@tool(tags=("shell",))
def run_python(code: str, timeout: int = 120) -> str:
    """Execute a Python snippet in a fresh subprocess and return its output.

    code: Python source to run (use print() to produce output).
    timeout: seconds before the process is killed.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp], capture_output=True, text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return f"[exit {proc.returncode}]\n" + (out[:40000] if out else "(no output)")
    except subprocess.TimeoutExpired:
        return f"Error: code timed out after {timeout}s."
    finally:
        Path(tmp).unlink(missing_ok=True)
