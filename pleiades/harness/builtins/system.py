"""
system.py — OS integration and self-awareness tools.

  notify(title, message)    — OS notification when a long task finishes.
  context_status()          — token budget: how much context I've used so far.
  write_clipboard(text)     — put something on the clipboard (GATED).
  read_clipboard()          — read current clipboard contents.
  get_env(key)              — read an environment variable safely.
  open_file(path)           — open a file in the OS default application.

These are the "agent connecting to the world around the task" tools.
context_status is especially important: without it, the agent doesn't know
when it's approaching the context limit and should start summarizing.
"""

from __future__ import annotations

import os
import platform
import subprocess

from ..tools import tool


# ---------------------------------------------------------------------------
# Notification
# ---------------------------------------------------------------------------

@tool(safe=True, tags=("system",))
def notify(title: str, message: str) -> str:
    """Send an OS desktop notification — use this when a long background task finishes so the user knows without watching the terminal.

    title: notification title (e.g. "Pleiades — task complete").
    message: body text (one or two sentences).
    """
    system = platform.system()
    try:
        if system == "Darwin":
            esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
            script = (f'display notification "{esc(message)}" '
                      f'with title "{esc(title)}"')
            subprocess.run(["osascript", "-e", script], timeout=5)
        elif system == "Windows":
            # Pass text via the environment ($env:PL_*) so quotes/semicolons in
            # title/message can't break out of (or inject into) the command.
            ps = (
                'Add-Type -AssemblyName System.Windows.Forms;'
                '$n = New-Object System.Windows.Forms.NotifyIcon;'
                '$n.Icon = [System.Drawing.SystemIcons]::Information;'
                '$n.Visible = $true;'
                '$n.ShowBalloonTip(5000, $env:PL_NOTIFY_TITLE, '
                '$env:PL_NOTIFY_MSG, [System.Windows.Forms.ToolTipIcon]::Info);'
                'Start-Sleep -Seconds 6; $n.Dispose()'
            )
            env = {**os.environ, "PL_NOTIFY_TITLE": title, "PL_NOTIFY_MSG": message}
            subprocess.Popen(
                ["powershell", "-WindowStyle", "Hidden", "-Command", ps],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
            )
        elif system == "Linux":
            # notify-send (libnotify) — available on most desktop distros
            subprocess.run(["notify-send", title, message], timeout=5)
        else:
            return f"Notification not supported on {system} (title: {title!r})"
    except Exception as e:
        return f"Notification failed: {e}"
    return f"Notification sent: {title!r}"


# ---------------------------------------------------------------------------
# Context / token budget
# ---------------------------------------------------------------------------

@tool(safe=True, tags=("system", "read"))
def context_status() -> str:
    """Report how many tokens have been used in this session and an estimate of remaining budget.

    Use this before a long task to check if you have room, and after heavy
    tool use to decide whether to summarize working memory or switch tiers.
    """
    from ..llm import get_session_usage
    from ..config import Config
    usage = get_session_usage()
    calls = usage["calls"]
    window = usage.get("last_input_tokens", 0)   # actual current-window size
    budget = Config.load().context_budget
    pct = window / budget * 100 if budget else 0.0

    lines = [
        f"Context window: ~{window:,} of {budget:,} tokens ({pct:.1f}% full)",
        f"Session totals ({calls} LLM call{'s' if calls != 1 else ''}): "
        f"input {usage['input_tokens']:,} / output {usage['output_tokens']:,}",
    ]
    if pct > 70:
        lines.append("⚠  Context is getting full — summarize findings into "
                     "remember/set_section; old tool outputs will be "
                     "auto-compressed at 75%.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Clipboard
# ---------------------------------------------------------------------------

@tool(safe=True, tags=("system", "read"))
def read_clipboard() -> str:
    """Read the current contents of the system clipboard."""
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=5)
            return result.stdout or "(clipboard is empty)"
        elif system == "Windows":
            result = subprocess.run(
                ["powershell", "-Command", "Get-Clipboard"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() or "(clipboard is empty)"
        elif system == "Linux":
            for cmd in [["xclip", "-selection", "clipboard", "-o"],
                        ["xsel", "--clipboard", "--output"]]:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        return result.stdout or "(clipboard is empty)"
                except FileNotFoundError:
                    continue
            return "Error: install xclip or xsel for clipboard access on Linux."
    except Exception as e:
        return f"Error reading clipboard: {e}"
    return "Clipboard read not supported on this platform."


@tool(tags=("system",))
def write_clipboard(text: str) -> str:
    """Write text to the system clipboard. GATED — puts content in the user's clipboard.

    text: the text to copy to the clipboard.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            proc = subprocess.run(["pbcopy"], input=text, text=True, timeout=5)
            return "Copied to clipboard." if proc.returncode == 0 else "pbcopy failed."
        elif system == "Windows":
            proc = subprocess.run(
                ["powershell", "-Command", "$input | Set-Clipboard"],
                input=text, text=True, timeout=5,
            )
            return "Copied to clipboard." if proc.returncode == 0 else "Set-Clipboard failed."
        elif system == "Linux":
            for cmd, inp_flag in [
                (["xclip", "-selection", "clipboard"], True),
                (["xsel", "--clipboard", "--input"], True),
            ]:
                try:
                    proc = subprocess.run(cmd, input=text, text=True, timeout=5)
                    if proc.returncode == 0:
                        return "Copied to clipboard."
                except FileNotFoundError:
                    continue
            return "Error: install xclip or xsel for clipboard write on Linux."
    except Exception as e:
        return f"Error writing clipboard: {e}"
    return "Clipboard write not supported on this platform."


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

@tool(safe=True, tags=("system", "read"))
def get_env(key: str) -> str:
    """Read an environment variable. Returns the value or '(not set)'.

    key: the environment variable name (e.g. "PATH", "HOME").
    """
    val = os.environ.get(key)
    return val if val is not None else f"(not set — {key} is not in the environment)"


@tool(tags=("system",))
def open_file(path: str) -> str:
    """Open a file or URL in the OS default application (like double-clicking it). GATED.

    path: file path or URL to open.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", path])
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        return f"Error opening {path!r}: {e}"
    return f"Opened {path!r}."
