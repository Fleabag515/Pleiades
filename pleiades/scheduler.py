"""Scheduled tasks — recurring or one-time work jobs a character runs on its own.

Pleiades is an inference engine + connector, not a training platform — this
module just lets a character (or the desktop app) schedule an existing
capability (a work job, the same thing `POST /api/work` runs) to fire at a
future time or on a recurring schedule. No third-party scheduling library:
cron matching is a small hand-rolled 5-field matcher (minute hour day month
weekday) — enough for a background loop that only needs "does now match",
not general-purpose calendar arithmetic.

State (under ~/.pleiades): scheduled_tasks.json — id -> ScheduledTask fields.

The background runner (`start_scheduler_thread`) wakes every `interval`
seconds (30s by default), fires any due task via a job-launcher callback the
caller supplies (this module has no import-time dependency on the webui
server — see `set_job_launcher`), and recomputes next_run. One-time
(`fire_at`) tasks disable themselves after firing; recurring (`cron`) tasks
compute their next occurrence and keep going.

Concurrency: the background thread and any number of webui/harness-tool
callers can all touch scheduled_tasks.json at the same time (create from the
desktop app while the tick thread is mid-mark_fired, etc). `_registry_lock`
below serializes every read-modify-write cycle against that one file so
concurrent writers can't stomp each other's changes (see
test_concurrent_creates_do_not_lose_writes), and `_save` writes to a temp
file + os.replace so a concurrent reader never sees a half-written file.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from . import config


def _tasks_json() -> Path:
    return config.PLEIADES_HOME / "scheduled_tasks.json"


# Guards every load->mutate->save cycle against scheduled_tasks.json. Module
# level (not per-instance) because the background tick thread, each webui
# request, and each harness-tool call all construct their own
# ScheduledTaskManager() but share the one file on disk.
_registry_lock = threading.RLock()


@dataclass
class ScheduledTask:
    id: str
    task: str                    # the prompt/instruction to run as a work job
    character: str = ""          # blank = process-wide default character
    cron: str = ""               # 5-field cron ("" if one-time)
    fire_at: float = 0.0         # epoch seconds (0 if recurring)
    tier: str = ""
    policy: str = ""
    enabled: bool = True
    created: float = field(default_factory=time.time)
    last_run: float = 0.0
    next_run: float = 0.0
    last_job_id: str = ""


class ScheduleError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Minimal cron: '*', '*/N', 'a,b,c', 'a-b', and combinations — no named
# months/weekdays, no '?', no 'L'/'W'. Weekday follows the standard cron
# convention (0 and 7 both mean Sunday, 1=Monday ... 6=Saturday).
# --------------------------------------------------------------------------- #
_CRON_FIELD_NAMES = ("minute", "hour", "day", "month", "weekday")
_CRON_RANGES = {"minute": (0, 59), "hour": (0, 23), "day": (1, 31),
                "month": (1, 12), "weekday": (0, 7)}


def _parse_cron_field(spec: str, lo: int, hi: int) -> set:
    out: set = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            part, step_s = part.split("/", 1)
            try:
                step = int(step_s)
            except ValueError:
                raise ScheduleError(f"bad step in cron field {spec!r}")
            if step <= 0:
                raise ScheduleError(f"step must be positive in cron field {spec!r}")
        if part == "*":
            base_lo, base_hi = lo, hi
        elif "-" in part:
            a, b = part.split("-", 1)
            try:
                base_lo, base_hi = int(a), int(b)
            except ValueError:
                raise ScheduleError(f"bad range in cron field {spec!r}")
        else:
            try:
                base_lo = base_hi = int(part)
            except ValueError:
                raise ScheduleError(f"bad value in cron field {spec!r}")
        if not (lo <= base_lo <= hi) or not (lo <= base_hi <= hi) or base_lo > base_hi:
            raise ScheduleError(f"cron field {spec!r} out of range [{lo}, {hi}]")
        out.update(v for v in range(base_lo, base_hi + 1) if (v - base_lo) % step == 0)
    if not out:
        raise ScheduleError(f"empty cron field {spec!r}")
    return out


def parse_cron(expr: str):
    """Parse a 5-field cron string into (minutes, hours, days, months, weekdays) sets.

    Raises ScheduleError on anything malformed — callers should validate at
    create/update time, not discover a bad expression only when it should fire.
    """
    fields = expr.strip().split()
    if len(fields) != 5:
        raise ScheduleError(
            f"cron must have 5 fields (minute hour day month weekday), got {expr!r}")
    return tuple(_parse_cron_field(f, *_CRON_RANGES[name])
                 for f, name in zip(fields, _CRON_FIELD_NAMES))


def cron_matches(expr: str, at: datetime) -> bool:
    minutes, hours, days, months, weekdays = parse_cron(expr)
    wd = at.isoweekday() % 7  # Sunday -> 0, Monday -> 1, ... Saturday -> 6
    return (at.minute in minutes and at.hour in hours and at.day in days
            and at.month in months and (wd in weekdays or (wd == 0 and 7 in weekdays)))


def next_cron_fire(expr: str, after: datetime, *, horizon_days: int = 366) -> Optional[datetime]:
    """Next match strictly after `after` (minute-truncated). Scans forward
    minute-by-minute — simple and correct, not meant for microsecond precision;
    a once-a-minute scheduler loop never needs more than this."""
    parse_cron(expr)  # validate eagerly
    t = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    limit = t + timedelta(days=horizon_days)
    while t <= limit:
        if cron_matches(expr, t):
            return t
        t += timedelta(minutes=1)
    return None  # malformed-but-parseable (e.g. Feb 30) — caller treats as "never"


class ScheduledTaskManager:
    """Create, list, update, delete, and fire scheduled tasks."""

    def __init__(self) -> None:
        config.ensure_home()

    def _load(self) -> dict:
        """Read the registry. Callers that mutate it must hold
        `_registry_lock` for the whole load-mutate-save cycle -- see each
        public method below."""
        p = _tasks_json()
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self, data: dict) -> None:
        """Write the registry atomically: a reader that races this write
        (list()/get() called without the lock, or a future out-of-process
        reader) always sees either the old file or the new one in full,
        never a JSON parse error from a half-written file."""
        p = _tasks_json()
        tmp = p.with_suffix(p.suffix + f".tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, p)

    def _compute_next_run(self, t: "ScheduledTask", after: Optional[datetime] = None) -> float:
        after = after or datetime.now()
        if t.cron:
            nxt = next_cron_fire(t.cron, after)
            return nxt.timestamp() if nxt else 0.0
        return t.fire_at  # one-time: next_run IS fire_at until it fires

    def create(self, task: str, *, character: str = "", cron: str = "",
               fire_at: float = 0.0, tier: str = "", policy: str = "") -> dict:
        task = (task or "").strip()
        if not task:
            raise ScheduleError("task must not be empty")
        if bool(cron) == bool(fire_at):
            raise ScheduleError("specify exactly one of cron or fire_at")
        if cron:
            parse_cron(cron)
        elif fire_at <= time.time():
            raise ScheduleError("fire_at must be in the future")
        with _registry_lock:
            reg = self._load()
            tid = uuid.uuid4().hex[:12]
            t = ScheduledTask(id=tid, task=task, character=character, cron=cron,
                              fire_at=fire_at, tier=tier, policy=policy)
            t.next_run = self._compute_next_run(t)
            reg[tid] = asdict(t)
            self._save(reg)
            return reg[tid]

    def get(self, task_id: str) -> Optional[dict]:
        with _registry_lock:
            return self._load().get(task_id)

    def list(self) -> list:
        with _registry_lock:
            return sorted(self._load().values(), key=lambda t: t.get("next_run") or 0)

    def update(self, task_id: str, **patch) -> dict:
        allowed = {"task", "character", "cron", "fire_at", "tier", "policy", "enabled"}
        bad = set(patch) - allowed
        if bad:
            raise ScheduleError(f"unknown field(s): {', '.join(sorted(bad))}")
        with _registry_lock:
            reg = self._load()
            if task_id not in reg:
                raise ScheduleError(f"no such scheduled task: {task_id}")
            d = reg[task_id]
            d.update({k: v for k, v in patch.items() if v is not None})
            if "cron" in patch or "fire_at" in patch:
                if bool(d.get("cron")) == bool(d.get("fire_at")):
                    raise ScheduleError("specify exactly one of cron or fire_at")
                if d.get("cron"):
                    parse_cron(d["cron"])
                t = ScheduledTask(**d)
                d["next_run"] = self._compute_next_run(t)
            reg[task_id] = d
            self._save(reg)
            return d

    def delete(self, task_id: str) -> bool:
        with _registry_lock:
            reg = self._load()
            if task_id not in reg:
                return False
            del reg[task_id]
            self._save(reg)
            return True

    def due_now(self, *, at: Optional[float] = None) -> list:
        now = at if at is not None else time.time()
        with _registry_lock:
            return [t for t in self._load().values()
                    if t.get("enabled") and t.get("next_run") and t["next_run"] <= now]

    def mark_fired(self, task_id: str, *, job_id: str = "", now: Optional[float] = None) -> None:
        with _registry_lock:
            reg = self._load()
            if task_id not in reg:
                return
            d = reg[task_id]
            now = now if now is not None else time.time()
            d["last_run"] = now
            d["last_job_id"] = job_id
            if d.get("cron"):
                t = ScheduledTask(**d)
                d["next_run"] = self._compute_next_run(t, datetime.fromtimestamp(now))
            else:
                d["enabled"] = False  # one-time task: fired once, done
                d["next_run"] = 0.0
            reg[task_id] = d
            self._save(reg)

    def run_now(self, task_id: str) -> dict:
        """Manually fire a task immediately, out of band from its schedule
        (the desktop UI's "Run now" button / a run_scheduled_task_now harness
        tool call). Uses the same job-launcher the background tick thread
        uses, then records last_run/last_job_id and advances next_run exactly
        like a normal firing would -- a one-time task still disables itself,
        a recurring task's schedule is untouched (next_run is recomputed from
        its cron, not shifted by the manual run)."""
        with _registry_lock:
            reg = self._load()
            if task_id not in reg:
                raise ScheduleError(f"no such scheduled task: {task_id}")
            t = dict(reg[task_id])
        job_id = ""
        if _launcher is not None:
            job_id = _launcher(t["task"], t.get("character", ""), t.get("tier", ""),
                              t.get("policy", ""))
        self.mark_fired(task_id, job_id=job_id)
        return self.get(task_id)


# --------------------------------------------------------------------------- #
# Background runner — decoupled from the webui server via a settable
# job-launcher callback, so this module stays a plain, testable library with
# no import-time dependency on the FastAPI app.
# --------------------------------------------------------------------------- #
JobLauncher = Callable[[str, str, str, str], str]  # (task, character, tier, policy) -> job_id

_launcher: Optional[JobLauncher] = None
_runner_thread: Optional[threading.Thread] = None
_runner_stop: Optional[threading.Event] = None


def set_job_launcher(fn: JobLauncher) -> None:
    """Wire in the function that actually starts a work job (webui's
    /api/work POST handler, refactored to be callable directly)."""
    global _launcher
    _launcher = fn


def tick(mgr: Optional[ScheduledTaskManager] = None) -> list:
    """Fire every currently-due task once. Returns the list of task ids fired.
    Safe to call with no launcher wired (e.g. from tests) — does nothing."""
    mgr = mgr or ScheduledTaskManager()
    fired = []
    for t in mgr.due_now():
        job_id = ""
        if _launcher is not None:
            try:
                job_id = _launcher(t["task"], t.get("character", ""),
                                   t.get("tier", ""), t.get("policy", ""))
            except Exception:
                job_id = ""  # best-effort: still advance next_run, don't wedge the loop
        mgr.mark_fired(t["id"], job_id=job_id)
        fired.append(t["id"])
    return fired


def start_scheduler_thread(*, interval: float = 30.0) -> threading.Thread:
    """Start (once) the background thread that ticks the scheduler every
    `interval` seconds. Idempotent — calling twice is a no-op after the first."""
    global _runner_thread, _runner_stop
    if _runner_thread is not None and _runner_thread.is_alive():
        return _runner_thread
    _runner_stop = threading.Event()
    stop = _runner_stop

    def loop() -> None:
        while not stop.is_set():
            try:
                tick()
            except Exception:
                pass  # never let a bad task/expression kill the whole loop
            stop.wait(interval)

    _runner_thread = threading.Thread(target=loop, daemon=True, name="pleiades-scheduler")
    _runner_thread.start()
    return _runner_thread


def stop_scheduler_thread() -> None:
    if _runner_stop is not None:
        _runner_stop.set()
