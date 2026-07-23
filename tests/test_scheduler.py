"""pleiades.scheduler: cron matching + ScheduledTaskManager CRUD + tick()."""

import threading
import time
from datetime import datetime

import pytest

from pleiades.scheduler import (ScheduledTaskManager, ScheduleError, cron_matches,
                                next_cron_fire, parse_cron, set_job_launcher, tick)


# --------------------------------------------------------------------------- #
# cron parsing / matching
# --------------------------------------------------------------------------- #
def test_parse_cron_every_minute():
    minutes, hours, days, months, weekdays = parse_cron("* * * * *")
    assert len(minutes) == 60 and len(hours) == 24
    assert len(days) == 31 and len(months) == 12 and len(weekdays) == 8  # 0..7


def test_parse_cron_step_and_range():
    minutes, hours, *_ = parse_cron("*/15 9-17 * * *")
    assert minutes == {0, 15, 30, 45}
    assert hours == set(range(9, 18))


def test_parse_cron_list():
    _, hours, days, *_ = parse_cron("0 6,18 1,15 * *")
    assert hours == {6, 18}
    assert days == {1, 15}


def test_parse_cron_wrong_field_count():
    with pytest.raises(ScheduleError):
        parse_cron("* * * *")


def test_parse_cron_out_of_range():
    with pytest.raises(ScheduleError):
        parse_cron("99 * * * *")


def test_cron_matches_exact_time():
    at = datetime(2026, 7, 22, 6, 0)  # a Wednesday
    assert cron_matches("0 6 * * *", at)
    assert not cron_matches("0 7 * * *", at)


def test_cron_matches_weekday_sunday_0_or_7():
    sunday = datetime(2026, 7, 26, 9, 0)  # confirmed Sunday
    assert cron_matches("0 9 * * 0", sunday)
    assert cron_matches("0 9 * * 7", sunday)
    assert not cron_matches("0 9 * * 1", sunday)


def test_next_cron_fire_daily():
    after = datetime(2026, 7, 22, 6, 30)
    nxt = next_cron_fire("0 6 * * *", after)
    assert nxt.day == 23 and nxt.hour == 6 and nxt.minute == 0


def test_next_cron_fire_later_same_day():
    after = datetime(2026, 7, 22, 5, 0)
    nxt = next_cron_fire("0 6 * * *", after)
    assert nxt.day == 22 and nxt.hour == 6


# --------------------------------------------------------------------------- #
# ScheduledTaskManager CRUD
# --------------------------------------------------------------------------- #
def test_create_one_time_task():
    mgr = ScheduledTaskManager()
    future = time.time() + 3600
    t = mgr.create("say hello", character="Mark", fire_at=future)
    assert t["cron"] == "" and t["fire_at"] == future
    assert t["next_run"] == future
    assert mgr.get(t["id"])["task"] == "say hello"


def test_create_recurring_task():
    mgr = ScheduledTaskManager()
    t = mgr.create("morning briefing", character="Mark", cron="0 6 * * *")
    assert t["next_run"] > time.time()


def test_create_requires_exactly_one_schedule_kind():
    mgr = ScheduledTaskManager()
    with pytest.raises(ScheduleError):
        mgr.create("bad", cron="0 6 * * *", fire_at=time.time() + 60)
    with pytest.raises(ScheduleError):
        mgr.create("bad")


def test_create_rejects_past_fire_at():
    mgr = ScheduledTaskManager()
    with pytest.raises(ScheduleError):
        mgr.create("too late", fire_at=time.time() - 60)


def test_create_rejects_empty_task():
    mgr = ScheduledTaskManager()
    with pytest.raises(ScheduleError):
        mgr.create("   ", fire_at=time.time() + 60)


def test_create_rejects_bad_cron():
    mgr = ScheduledTaskManager()
    with pytest.raises(ScheduleError):
        mgr.create("bad cron", cron="not a cron")


def test_list_sorted_by_next_run():
    mgr = ScheduledTaskManager()
    a = mgr.create("later", fire_at=time.time() + 7200)
    b = mgr.create("sooner", fire_at=time.time() + 60)
    ids_in_order = [t["id"] for t in mgr.list()]
    assert ids_in_order.index(b["id"]) < ids_in_order.index(a["id"])


def test_update_task_text():
    mgr = ScheduledTaskManager()
    t = mgr.create("original", fire_at=time.time() + 60)
    updated = mgr.update(t["id"], task="revised")
    assert updated["task"] == "revised"


def test_update_reschedule_from_onetime_to_cron():
    mgr = ScheduledTaskManager()
    t = mgr.create("task", fire_at=time.time() + 60)
    updated = mgr.update(t["id"], cron="0 6 * * *", fire_at=0.0)
    assert updated["cron"] == "0 6 * * *"
    assert updated["next_run"] > time.time()


def test_update_unknown_task_raises():
    mgr = ScheduledTaskManager()
    with pytest.raises(ScheduleError):
        mgr.update("does-not-exist", task="x")


def test_update_unknown_field_raises():
    mgr = ScheduledTaskManager()
    t = mgr.create("task", fire_at=time.time() + 60)
    with pytest.raises(ScheduleError):
        mgr.update(t["id"], not_a_real_field="x")


def test_delete_task():
    mgr = ScheduledTaskManager()
    t = mgr.create("to delete", fire_at=time.time() + 60)
    assert mgr.delete(t["id"]) is True
    assert mgr.get(t["id"]) is None
    assert mgr.delete(t["id"]) is False  # already gone


# --------------------------------------------------------------------------- #
# due_now / mark_fired / tick
# --------------------------------------------------------------------------- #
def test_due_now_finds_overdue_task():
    mgr = ScheduledTaskManager()
    t = mgr.create("due", fire_at=time.time() + 3600)
    assert t["id"] not in [d["id"] for d in mgr.due_now()]
    assert t["id"] in [d["id"] for d in mgr.due_now(at=time.time() + 7200)]


def test_mark_fired_one_time_disables():
    mgr = ScheduledTaskManager()
    t = mgr.create("fire once", fire_at=time.time() + 60)
    mgr.mark_fired(t["id"], job_id="job-abc")
    d = mgr.get(t["id"])
    assert d["enabled"] is False
    assert d["last_job_id"] == "job-abc"
    assert d["next_run"] == 0.0


def test_mark_fired_recurring_advances_next_run():
    mgr = ScheduledTaskManager()
    t = mgr.create("daily", cron="0 6 * * *")
    first_next = t["next_run"]
    mgr.mark_fired(t["id"], job_id="job-1", now=first_next)
    d = mgr.get(t["id"])
    assert d["enabled"] is True
    assert d["next_run"] > first_next


def test_tick_fires_due_tasks_and_calls_launcher():
    mgr = ScheduledTaskManager()
    fired_calls = []

    def fake_launcher(task, character, tier, policy):
        fired_calls.append((task, character))
        return "job-xyz"

    set_job_launcher(fake_launcher)
    try:
        t = mgr.create("ping Mark", character="Mark", fire_at=time.time() + 0.05)
        time.sleep(0.1)  # let it become due
        fired = tick(mgr)
        assert t["id"] in fired
        assert ("ping Mark", "Mark") in fired_calls
        assert mgr.get(t["id"])["enabled"] is False
    finally:
        set_job_launcher(None)


def test_tick_with_no_launcher_still_advances_state():
    mgr = ScheduledTaskManager()
    set_job_launcher(None)
    t = mgr.create("no launcher", fire_at=time.time() + 0.05)
    time.sleep(0.1)
    fired = tick(mgr)
    assert t["id"] in fired
    assert mgr.get(t["id"])["enabled"] is False


# --------------------------------------------------------------------------- #
# run_now: manual "fire immediately" trigger, independent of the schedule
# --------------------------------------------------------------------------- #
def test_run_now_fires_one_time_task_and_disables_it():
    mgr = ScheduledTaskManager()
    fired_calls = []

    def fake_launcher(task, character, tier, policy):
        fired_calls.append(task)
        return "job-manual-1"

    set_job_launcher(fake_launcher)
    try:
        t = mgr.create("run me manually", fire_at=time.time() + 3600)
        d = mgr.run_now(t["id"])
        assert "run me manually" in fired_calls
        assert d["last_job_id"] == "job-manual-1"
        assert d["enabled"] is False       # one-time: still disables after firing
        assert d["last_run"] > 0
    finally:
        set_job_launcher(None)


def test_run_now_recurring_task_keeps_its_schedule():
    mgr = ScheduledTaskManager()
    set_job_launcher(lambda *a: "job-manual-2")
    try:
        t = mgr.create("recurring manual", cron="0 6 * * *")
        original_next_run = t["next_run"]
        d = mgr.run_now(t["id"])
        assert d["enabled"] is True
        assert d["last_job_id"] == "job-manual-2"
        # next_run is recomputed from the cron (not shifted by the manual
        # run), so it may equal or advance past the original -- but the task
        # must still be scheduled to run again.
        assert d["next_run"] >= original_next_run - 1
    finally:
        set_job_launcher(None)


def test_run_now_unknown_task_raises():
    mgr = ScheduledTaskManager()
    with pytest.raises(ScheduleError):
        mgr.run_now("does-not-exist")


# --------------------------------------------------------------------------- #
# concurrency: background tick thread + API/harness calls hit the same
# scheduled_tasks.json with no locking -- read-then-write is vulnerable to a
# classic lost-update race under concurrent access.
# --------------------------------------------------------------------------- #
def test_concurrent_creates_do_not_lose_writes(monkeypatch):
    """Reproduces the race: ScheduledTaskManager._save() overwrites the whole
    file with no lock guarding the load-mutate-save cycle, so N threads
    calling .create() around the same time can each read the same
    pre-mutation snapshot and stomp each other's writes on the way out."""
    mgr = ScheduledTaskManager()
    orig_save = ScheduledTaskManager._save

    def slow_save(self, data):
        time.sleep(0.05)  # widen the window between read and write
        orig_save(self, data)

    monkeypatch.setattr(ScheduledTaskManager, "_save", slow_save)

    n = 8
    created_ids: list = []
    ids_lock = threading.Lock()

    def make(i):
        t = mgr.create(f"race-task-{i}", fire_at=time.time() + 3600 + i)
        with ids_lock:
            created_ids.append(t["id"])

    threads = [threading.Thread(target=make, args=(i,)) for i in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    reg = mgr._load()
    survived = [tid for tid in created_ids if tid in reg]
    assert len(survived) == n, (
        f"lost {n - len(survived)} of {n} concurrently-created scheduled "
        "tasks to a read-modify-write race (no lock around "
        "load+mutate+save in ScheduledTaskManager)"
    )
