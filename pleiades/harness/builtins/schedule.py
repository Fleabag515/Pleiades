"""Scheduled-task tools — a character can create, inspect, edit, and cancel
its own scheduled or recurring work, the same shape as Cowork's own
scheduled-tasks tool.

    create_scheduled_task("check overnight logs and summarize", cron="0 7 * * *")
    create_scheduled_task("remind Fleagle to stretch", fire_at="2026-07-22 15:00")
    list_scheduled_tasks()
    update_scheduled_task(task_id, enabled=False)
    delete_scheduled_task(task_id)

A scheduled task fires by starting a real work job (the same thing
POST /api/work runs) for the given — or, if left blank, the currently
active — character, at the given time or on the given cron schedule. Cron is
the standard 5-field format (minute hour day month weekday); fire_at accepts
any ISO-8601-ish datetime string.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from .. import identity
from ..tools import tool
from ...scheduler import ScheduledTaskManager, ScheduleError


def _parse_when(s: str) -> float:
    s = s.strip()
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        raise ScheduleError(
            f"couldn't parse {s!r} as a datetime — use an ISO-8601-ish format "
            f"like '2026-07-22 15:00' or '2026-07-22T15:00:00'")


def _default_character(character: str) -> str:
    if character:
        return character
    p = identity.active_profile()
    return p.name if p else ""


def _describe_when(t: dict) -> str:
    if t.get("cron"):
        return f"cron {t['cron']}"
    if t.get("next_run"):
        return datetime.fromtimestamp(t["next_run"]).isoformat(sep=" ", timespec="minutes")
    return "(already fired, one-time)"


@tool(tags=("schedule",))
def create_scheduled_task(task: str, cron: str = "", fire_at: str = "",
                          character: str = "") -> str:
    """Schedule a task to run once at a specific time, or repeatedly on a cron schedule.

    task: the instruction to run as a work job when it fires.
    cron: 5-field cron ("minute hour day month weekday", e.g. "0 7 * * *" = every day at 7am). Leave blank for a one-time task.
    fire_at: ISO-8601-ish datetime for a one-time task (e.g. "2026-07-22 15:00"). Leave blank for a recurring task.
    character: which character this runs as. Blank = whichever character is calling this tool.
    """
    try:
        fa = _parse_when(fire_at) if fire_at else 0.0
        t = ScheduledTaskManager().create(
            task, character=_default_character(character), cron=cron, fire_at=fa)
    except ScheduleError as e:
        return f"Error: {e}"
    return (f"Scheduled task {t['id']} created — runs {_describe_when(t)} "
           f"as {t['character'] or '(default character)'}.")


@tool(safe=True, tags=("schedule", "read"))
def list_scheduled_tasks() -> str:
    """List all scheduled tasks (recurring and one-time), soonest-first."""
    tasks = ScheduledTaskManager().list()
    if not tasks:
        return "(no scheduled tasks)"
    lines = []
    for t in tasks:
        status = "enabled" if t["enabled"] else "disabled"
        lines.append(f"- [{t['id']}] {t['task']!r} — {_describe_when(t)} — "
                     f"{t['character'] or '(default)'} — {status}")
    return "\n".join(lines)


@tool(tags=("schedule",))
def update_scheduled_task(task_id: str, task: str = "", cron: str = "",
                          fire_at: str = "", enabled: Optional[bool] = None) -> str:
    """Edit a scheduled task's instruction, schedule, or enabled state.

    task_id: id from list_scheduled_tasks.
    task: new instruction text. Leave blank to keep unchanged.
    cron: new cron schedule — setting this switches the task to recurring. Leave blank to keep unchanged (unless fire_at is set).
    fire_at: new one-time ISO-8601-ish datetime — setting this switches the task to one-time. Leave blank to keep unchanged (unless cron is set).
    enabled: pass true/false to enable or disable without changing the schedule. Omit to leave unchanged.
    """
    patch: dict = {}
    if task:
        patch["task"] = task
    try:
        if cron:
            patch["cron"] = cron
            patch["fire_at"] = 0.0
        elif fire_at:
            patch["fire_at"] = _parse_when(fire_at)
            patch["cron"] = ""
        if enabled is not None:
            patch["enabled"] = enabled
        t = ScheduledTaskManager().update(task_id, **patch)
    except ScheduleError as e:
        return f"Error: {e}"
    return f"Updated {t['id']} — now runs {_describe_when(t)}."


@tool(tags=("schedule",))
def delete_scheduled_task(task_id: str) -> str:
    """Cancel and delete a scheduled task.

    task_id: id from list_scheduled_tasks.
    """
    ok = ScheduledTaskManager().delete(task_id)
    return f"Deleted {task_id}." if ok else f"Error: no such scheduled task: {task_id}"
