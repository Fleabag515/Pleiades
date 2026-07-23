"""Task-list tools — the harness's own progress tracker, same shape as
Cowork's own TaskCreate/TaskUpdate/TaskList: a character lays out a short
plan at the start of a multi-step job, marks each step in_progress then
completed as it actually does the work, and a human watching the job (the
desktop app's Progress panel) sees that list update live.

    create_task("read config.py", description="check the tier defaults",
                activeForm="Reading config.py")
    list_tasks()
    update_task(task_id, status="in_progress")
    update_task(task_id, status="completed")
    delete_task(task_id)

Scoping: a task list belongs to the current *work session* — the same job a
`pleiades work` run or a webui POST /api/work call represents — not to a
character or to the whole process. bind_job(job_id) is called once, at the
top of that run (pleiades/webui/server.py's work-job runner and `pleiades
work` in cli.py both do this, the same way they already call bind_memory /
bind_character), and every task tool call for the rest of that run reads and
writes that job's list only. There is deliberately no cross-job/global task
list: a task list describes one in-flight job's plan, not a permanent record
— exactly mirroring how pleiades.webui.server's work_jobs dict itself is
in-memory-only and scoped to the process, not persisted to disk.

Unlike bind_character/bind_memory/bind_context (plain module-level globals,
fine for those because only one identity/memory store is "active" at a
time), work jobs genuinely run concurrently — the webui launches every job
on its own background thread, and a scheduled task can fire while a
human-triggered job is still running. bind_job therefore uses thread-local
storage: each job's own thread — and the same thread's sequential tool
dispatch inside Agent.run — sees its own job id, so two jobs running at once
never cross-write each other's task list. dispatch_subagents_parallel (which
fans a job's own work out across a ThreadPoolExecutor) re-binds the parent's
job id into each worker thread so subagent-created tasks still land on the
right list; see subagent.py.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Optional

from ..tools import tool

STATUSES = ("pending", "in_progress", "completed")

# job_id -> ordered list of task dicts. Process-lifetime, in-memory only —
# same persistence model as pleiades.webui.server's work_jobs.
_TASKS: dict[str, list[dict]] = {}

# Which job id the CURRENT THREAD's tool calls should read/write. Thread-local
# (not a plain dict like bind_character/bind_memory) because work jobs run
# concurrently on their own threads — see module docstring.
_local = threading.local()

_UNBOUND = "_unbound"  # fallback bucket for tool calls with no bound job
                       # (bare Agent() in tests/scripts, or a bridged chat call).


def bind_job(job_id: str) -> None:
    """Scope this thread's task tool calls to `job_id`. Called once at the
    start of a work run by the webui job runner and `pleiades work`."""
    _local.job_id = job_id
    _TASKS.setdefault(job_id, [])


def current_job_id() -> Optional[str]:
    """The job id bound to the calling thread, if any."""
    return getattr(_local, "job_id", None)


def get_tasks(job_id: str) -> list[dict]:
    """Webui-facing read: a copy of the live task list for a job id (empty
    list, not an error, if the job never created any tasks)."""
    return [dict(t) for t in _TASKS.get(job_id, [])]


def clear_job(job_id: str) -> None:
    """Drop a job's task list (e.g. when the job record itself is cleared)."""
    _TASKS.pop(job_id, None)


def _current_list() -> list[dict]:
    job_id = current_job_id() or _UNBOUND
    return _TASKS.setdefault(job_id, [])


def _find(tasks: list[dict], task_id: str) -> Optional[dict]:
    for t in tasks:
        if t["id"] == task_id:
            return t
    return None


@tool(tags=("tasks",))
def create_task(subject: str, description: str = "", activeForm: str = "") -> str:
    """Add a task to the current job's live task list. Use this at the start of
    any non-trivial multi-step job to lay out your plan, and append more as new
    steps become clear as you work.

    subject: short imperative label, e.g. "Fix the auth bug".
    description: optional extra detail on what the step involves.
    activeForm: present-continuous form shown while this task is in_progress,
                e.g. "Fixing the auth bug". Defaults to subject if left blank.
    """
    tasks = _current_list()
    t = {
        "id": uuid.uuid4().hex[:8],
        "subject": subject,
        "description": description,
        "activeForm": activeForm or subject,
        "status": "pending",
        "created": time.time(),
        "updated": time.time(),
    }
    tasks.append(t)
    return f"Created task {t['id']}: {subject!r} (pending)."


@tool(safe=True, tags=("tasks", "read"))
def list_tasks() -> str:
    """List the current job's tasks in creation order, with status and, for
    the in_progress one, its activeForm."""
    tasks = _current_list()
    if not tasks:
        return "(no tasks yet)"
    marker = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
    lines = []
    for t in tasks:
        extra = f" — {t['activeForm']}" if t["status"] == "in_progress" else ""
        lines.append(f"{marker[t['status']]} [{t['id']}] {t['subject']}{extra}")
    return "\n".join(lines)


@tool(tags=("tasks",))
def update_task(task_id: str, status: str = "", subject: str = "",
                description: str = "", activeForm: str = "") -> str:
    """Update a task's status and/or details. Keep at most one task
    in_progress at a time — mark the previous one completed before starting
    the next, so the live list always shows what's actually happening now.

    task_id: id returned by create_task / shown by list_tasks.
    status: "pending", "in_progress", or "completed". Blank = unchanged.
    subject, description, activeForm: new values; blank = unchanged.
    """
    if status and status not in STATUSES:
        return f"Error: status must be one of {STATUSES}, got {status!r}."
    t = _find(_current_list(), task_id)
    if t is None:
        return f"Error: no such task: {task_id}"
    if status:
        t["status"] = status
    if subject:
        t["subject"] = subject
    if description:
        t["description"] = description
    if activeForm:
        t["activeForm"] = activeForm
    t["updated"] = time.time()
    return f"Updated task {t['id']}: {t['subject']} ({t['status']})."


@tool(tags=("tasks",))
def delete_task(task_id: str) -> str:
    """Remove a task from the current job's list (e.g. it turned out to be
    unnecessary). Prefer marking a task "completed" over deleting it when the
    step was actually done — delete is for plan changes, not progress."""
    tasks = _current_list()
    for i, t in enumerate(tasks):
        if t["id"] == task_id:
            tasks.pop(i)
            return f"Deleted task {task_id}."
    return f"Error: no such task: {task_id}"
