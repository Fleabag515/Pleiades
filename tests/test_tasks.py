"""Tests for the harness's live task-list tools (pleiades.harness.builtins.tasks),
the create_task/update_task/list_tasks/delete_task belt that mirrors Cowork's own
TaskCreate/TaskUpdate/TaskList shape. Pure in-memory, offline."""

import threading

import pytest

from pleiades.harness.builtins import tasks as tasks_mod
from pleiades.harness.builtins.tasks import (
    bind_job, clear_job, create_task, current_job_id, delete_task, get_tasks,
    list_tasks, update_task,
)


@pytest.fixture(autouse=True)
def _isolated_job():
    """Every test gets its own job id so tests never see each other's tasks,
    and thread-local state never leaks across tests."""
    job_id = f"test-{id(object())}"
    bind_job(job_id)
    yield job_id
    clear_job(job_id)
    tasks_mod._local.job_id = None


def test_create_task_starts_pending():
    out = create_task("read config.py")
    assert "pending" in out
    listed = list_tasks()
    assert "read config.py" in listed
    assert listed.startswith("[ ]")


def test_create_task_defaults_activeform_to_subject(_isolated_job):
    create_task("fix bug")
    t = get_tasks(_isolated_job)[0]
    assert t["activeForm"] == "fix bug"


def test_create_task_custom_activeform_and_description(_isolated_job):
    create_task("fix bug", description="in auth.py", activeForm="Fixing the bug")
    t = get_tasks(_isolated_job)[0]
    assert t["description"] == "in auth.py"
    assert t["activeForm"] == "Fixing the bug"


def test_list_tasks_empty():
    assert list_tasks() == "(no tasks yet)"


def test_update_task_transitions_status(_isolated_job):
    create_task("step one")
    tid = get_tasks(_isolated_job)[0]["id"]

    update_task(tid, status="in_progress")
    assert get_tasks(_isolated_job)[0]["status"] == "in_progress"
    assert "[~]" in list_tasks()

    out = update_task(tid, status="completed")
    assert "completed" in out
    assert get_tasks(_isolated_job)[0]["status"] == "completed"


def test_update_task_shows_activeform_while_in_progress(_isolated_job):
    create_task("step one", activeForm="Doing step one")
    tid = get_tasks(_isolated_job)[0]["id"]
    update_task(tid, status="in_progress")
    assert "Doing step one" in list_tasks()


def test_update_task_rejects_bad_status(_isolated_job):
    create_task("step one")
    tid = get_tasks(_isolated_job)[0]["id"]
    out = update_task(tid, status="done")
    assert out.startswith("Error")
    assert get_tasks(_isolated_job)[0]["status"] == "pending"  # unchanged


def test_update_task_unknown_id():
    out = update_task("nope", status="completed")
    assert out.startswith("Error")


def test_update_task_edits_fields(_isolated_job):
    create_task("original subject")
    tid = get_tasks(_isolated_job)[0]["id"]
    update_task(tid, subject="revised subject", description="more detail")
    t = get_tasks(_isolated_job)[0]
    assert t["subject"] == "revised subject"
    assert t["description"] == "more detail"


def test_delete_task(_isolated_job):
    create_task("throwaway")
    tid = get_tasks(_isolated_job)[0]["id"]
    out = delete_task(tid)
    assert "Deleted" in out
    assert get_tasks(_isolated_job) == []


def test_delete_unknown_task():
    out = delete_task("nope")
    assert out.startswith("Error")


def test_tasks_scoped_per_job_not_global():
    bind_job("job-a")
    create_task("a's task")
    bind_job("job-b")
    create_task("b's task")

    a_tasks = get_tasks("job-a")
    b_tasks = get_tasks("job-b")
    assert [t["subject"] for t in a_tasks] == ["a's task"]
    assert [t["subject"] for t in b_tasks] == ["b's task"]

    clear_job("job-a")
    clear_job("job-b")


def test_get_tasks_returns_copies_not_live_refs(_isolated_job):
    create_task("step")
    snapshot = get_tasks(_isolated_job)
    snapshot[0]["status"] = "completed"
    assert get_tasks(_isolated_job)[0]["status"] == "pending"


def test_current_job_id_is_thread_local():
    """Two threads binding different job ids must not see each other's --
    this is the whole reason bind_job uses threading.local instead of a
    plain module dict (unlike bind_character/bind_memory)."""
    seen = {}

    def worker(name):
        bind_job(name)
        # yield the GIL so the other thread can run and (if this were a
        # plain shared dict) clobber our binding
        import time as _t
        _t.sleep(0.05)
        seen[name] = current_job_id()

    t1 = threading.Thread(target=worker, args=("thread-a",))
    t2 = threading.Thread(target=worker, args=("thread-b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert seen == {"thread-a": "thread-a", "thread-b": "thread-b"}
    clear_job("thread-a")
    clear_job("thread-b")


def test_unbound_thread_falls_back_to_unbound_bucket():
    def worker():
        # a fresh thread that never called bind_job
        assert current_job_id() is None
        create_task("orphan")

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert any(x["subject"] == "orphan" for x in get_tasks("_unbound"))
    clear_job("_unbound")


def test_registered_as_harness_tools():
    import pleiades.harness.builtins  # noqa: F401  registers all builtin tools
    from pleiades.harness import registry

    names = {t.name for t in registry.all()}
    for expected in ("create_task", "list_tasks", "update_task", "delete_task"):
        assert expected in names, f"missing harness tool: {expected}"

    ct = registry.get("create_task")
    assert ct.safe is False  # side-effecting (mutates state) -> gated like schedule tools
    lt = registry.get("list_tasks")
    assert lt.safe is True   # read-only
