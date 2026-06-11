"""
notebook.py — a shared, human-editable notebook for the character.

ONE plain-Markdown file that BOTH the human and the agent read and write. Unlike
the Anamnesis-backed memory tools (internal stores the human can't see), this is
an ordinary file the human can open in any editor — the place to keep running
notes the two of you maintain together: tool reminders, preferences, "here's how
we do X", open questions. The agent reads it to orient, and appends to it as it
learns; the human edits the same file directly.

Path is pinned per-character by bind_notebook() (the character's profile dir, as
NOTEBOOK.md); falls back to ./NOTEBOOK.md when no character is bound.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from ..tools import tool

_NB: dict[str, str | None] = {"path": None}


def bind_notebook(path: str) -> None:
    """Pin the notebook file for this run (e.g. a character's profile dir)."""
    _NB["path"] = path or None


def _path() -> Path:
    p = Path(_NB["path"] or (Path.cwd() / "NOTEBOOK.md"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@tool(safe=True, tags=("memory", "read", "notebook"))
def notebook_read(max_chars: int = 8000) -> str:
    """Read the shared notebook — the Markdown file you and your human both keep. Check it when you need a reminder of how you two work, your tool notes, or running context. It is the human-visible companion to your private memory.

    max_chars: cap on returned characters (default 8000).
    """
    p = _path()
    if not p.exists():
        return (f"(notebook is empty — it will be created at {p} on first write. "
                "Use notebook_append to add the first entry.)")
    text = p.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return f"(notebook at {p} is empty)"
    head = f"# Shared notebook ({p})\n\n"
    return head + (text[:max_chars] + ("\n... [truncated]" if len(text) > max_chars else ""))


@tool(tags=("memory", "notebook"))
def notebook_append(text: str) -> str:
    """Append a dated entry to the shared notebook WITHOUT erasing anything. This is the safe, everyday way to add a note your human will also see.

    text: the note to add (one or more lines).
    """
    p = _path()
    new = not p.exists() or not p.read_text(encoding="utf-8", errors="replace").strip()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with p.open("a", encoding="utf-8") as f:
        if new:
            f.write("# Shared notebook\n\n*Maintained together by the character and the human.*\n")
        f.write(f"\n- _{stamp}_ — {text.strip()}\n")
    return f"Appended to the shared notebook ({p})."


@tool(tags=("memory", "notebook"))
def notebook_write(content: str) -> str:
    """REPLACE the whole shared notebook with new content. Use this only to reorganise or rewrite the file; for routine additions use notebook_append so nothing is lost.

    content: the full new contents of the notebook.
    """
    p = _path()
    p.write_text(content, encoding="utf-8")
    return f"Rewrote the shared notebook ({p})."
