"""
Memory tools — two tiers of anamnesis:

  working memory (live scratchpad):  note_to_self / read_working_memory /
                                     set_section / add_finding /
                                     clear_working_memory
  long-term memory (durable facts):  remember / recall / forget / list_memory

Recall is hybrid: semantic (local Ollama embeddings) blended with keyword
overlap, degrading gracefully to pure keyword matching when no embedder is
available.

Working memory is auto-injected into the system prompt at the start of every run,
so the agent always resumes with its own state in front of it.

SECTION-BASED WORKFLOW (recommended for multi-step tasks):
  Use set_section to maintain named blocks — Goal, Plan, Findings, Open — that
  you can update independently. This is much cleaner than note_to_self for
  structured tasks. note_to_self remains available for quick ad-hoc notes.

  Example:
      set_section("Goal", "Find and fix the auth bug in user.py")
      set_section("Plan", "1. Read user.py\\n2. Run tests\\n3. Fix\\n4. Verify")
      add_finding("JWT expiry check uses wrong timezone on line 142")
      set_section("Open", "Does this affect refresh tokens too?")
"""

from __future__ import annotations

from ..tools import tool
from ..anamnesis import Anamnesis

_MEM: dict[str, Anamnesis] = {"store": None}  # bound at startup by run.py


def bind_memory(store: Anamnesis) -> None:
    _MEM["store"] = store


def _store() -> Anamnesis | None:
    return _MEM["store"]


# ---- working memory: section-based (preferred) ----------------------------

@tool(tags=("memory",))
def set_section(section: str, content: str) -> str:
    """Set a named section in working memory, replacing it if it already exists. Use this to maintain structured state: Goal, Plan, Findings, Open, etc.

    section: section name (e.g. "Goal", "Plan", "Findings", "Open").
    content: the full content for this section (multi-line ok).
    """
    store = _store()
    if store is None:
        return "Error: memory not bound."
    store.scratch_set_section(section, content)
    return f"Set [{section}] in working memory."


@tool(tags=("memory",))
def add_finding(finding: str) -> str:
    """Append a bullet to the Findings section of working memory. Call this whenever you discover something relevant during a task.

    finding: a single finding or observation to record.
    """
    store = _store()
    if store is None:
        return "Error: memory not bound."
    store.scratch_append_section("Findings", finding)
    return "Added to Findings."


# ---- working memory: flat log (quick notes) --------------------------------

@tool(tags=("memory",))
def note_to_self(text: str) -> str:
    """Append a timestamped line to working memory — the live scratchpad shown to you at the start of every run. Use it to track the goal, plan, findings, and open questions as you work. For structured tasks prefer set_section + add_finding.

    text: a short note to your future self (one line).
    """
    store = _store()
    if store is None:
        return "Error: memory not bound."
    store.scratch_append(text)
    return "Noted to working memory."


@tool(safe=True, tags=("memory", "read"))
def read_working_memory() -> str:
    """Read your current working-memory scratchpad in full."""
    store = _store()
    if store is None:
        return "Error: memory not bound."
    return store.working() or "(working memory is empty)"


@tool(tags=("memory",))
def clear_working_memory() -> str:
    """Wipe your working memory — do this when a task is finished or context changes."""
    store = _store()
    if store is None:
        return "Error: memory not bound."
    store.scratch_clear()
    return "Working memory cleared."


# ---- checkpoints: pause/resume across restarts ------------------------------

@tool(tags=("memory",))
def save_checkpoint(label: str, notes: str = "") -> str:
    """Snapshot current working memory under a label so a long task can be resumed after a restart. Working memory itself is untouched.

    label: short name for this checkpoint (e.g. "auth-refactor-step3").
    notes: optional extra context to store alongside (what's done, what's next).
    """
    store = _store()
    if store is None:
        return "Error: memory not bound."
    slug = store.checkpoint_save(label, extra=notes)
    return f"Checkpoint '{slug}' saved."


@tool(tags=("memory",))
def load_checkpoint(label: str) -> str:
    """Restore working memory from a checkpoint (replaces current working memory). Returns the restored content.

    label: the checkpoint name as shown by list_checkpoints.
    """
    store = _store()
    if store is None:
        return "Error: memory not bound."
    body = store.checkpoint_load(label)
    if body is None:
        return f"Error: no checkpoint '{label}'. Use list_checkpoints."
    return f"Restored working memory from '{label}':\n\n{body}"


@tool(safe=True, tags=("memory", "read"))
def list_checkpoints() -> str:
    """List saved checkpoints (newest first) with their timestamps."""
    store = _store()
    if store is None:
        return "Error: memory not bound."
    cps = store.checkpoint_list()
    if not cps:
        return "(no checkpoints saved)"
    return "\n".join(f"- {label}  (saved {ts})" for label, ts in cps)


# ---- long-term memory ------------------------------------------------------

@tool(tags=("memory",))
def remember(name: str, fact: str, kind: str = "note") -> str:
    """Save a durable fact to long-term memory (anamnesis).

    name: short title/slug for the memory.
    fact: the information to remember (one self-contained fact).
    kind: category — "user", "project", "reference", or "note".
    """
    store = _store()
    if store is None:
        return "Error: memory not bound."
    slug = store.write(name, fact, kind=kind)
    return f"Saved memory '{slug}'."


@tool(safe=True, tags=("memory", "read"))
def recall(query: str) -> str:
    """Search long-term memory for facts relevant to a query. Recall is semantic when a local embedding model is available (synonyms and paraphrases work), with keyword fallback — phrase the query naturally.

    query: what you want to remember about.
    """
    store = _store()
    if store is None:
        return "Error: memory not bound."
    return store.recall(query) or "(no relevant memories)"


@tool(tags=("memory",))
def forget(name: str) -> str:
    """Delete a note from long-term memory by name. Use when a saved fact is wrong or obsolete (e.g. before re-saving a corrected version).

    name: the note's name/slug as shown by list_memory.
    """
    store = _store()
    if store is None:
        return "Error: memory not bound."
    if store.forget(name):
        return f"Forgot '{name}'."
    return f"No memory named '{name}'."


@tool(safe=True, tags=("memory", "read"))
def list_memory() -> str:
    """List all notes currently saved in long-term memory (names only). Use recall to retrieve the content of specific notes."""
    store = _store()
    if store is None:
        return "Error: memory not bound."
    names = store.list()
    if not names:
        return "(long-term memory is empty)"
    return "\n".join(f"- {n}" for n in names)
