"""GET /api/work/{job_id}/tasks -- the desktop app's Progress panel polls this
for a job's live task list. The store itself (pleiades.harness.builtins.tasks)
is unit-tested in test_tasks.py; this file only exercises the HTTP surface and
that it reflects the same module a running job's create_task/update_task
tool calls would write to."""

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from pleiades.webui import create_app  # noqa: E402
from pleiades.harness.builtins.tasks import bind_job, clear_job, create_task, update_task  # noqa: E402


def _client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


def _start_job(c: TestClient) -> str:
    r = c.post("/api/work", json={"task": "noop", "policy": "deny"})
    assert r.status_code == 200
    return r.json()["id"]


def test_get_tasks_for_unknown_job_is_404():
    c = _client()
    r = c.get("/api/work/does-not-exist/tasks")
    assert r.status_code == 404


def test_get_tasks_empty_list_for_job_with_no_tasks_yet():
    c = _client()
    job_id = _start_job(c)
    try:
        r = c.get(f"/api/work/{job_id}/tasks")
        assert r.status_code == 200
        assert r.json() == {"tasks": []}
    finally:
        clear_job(job_id)


def test_get_tasks_reflects_live_tool_calls_for_that_job():
    """Simulates what the harness tools do mid-job: bind_job(job_id) then
    create_task/update_task -- the same module the webui endpoint reads."""
    c = _client()
    job_id = _start_job(c)
    try:
        bind_job(job_id)
        create_task("read config.py", activeForm="Reading config.py")

        body = c.get(f"/api/work/{job_id}/tasks").json()
        assert len(body["tasks"]) == 1
        t = body["tasks"][0]
        assert t["subject"] == "read config.py"
        assert t["status"] == "pending"

        update_task(t["id"], status="in_progress")
        body = c.get(f"/api/work/{job_id}/tasks").json()
        assert body["tasks"][0]["status"] == "in_progress"
        assert body["tasks"][0]["activeForm"] == "Reading config.py"

        update_task(t["id"], status="completed")
        body = c.get(f"/api/work/{job_id}/tasks").json()
        assert body["tasks"][0]["status"] == "completed"
    finally:
        clear_job(job_id)


def test_tasks_for_one_job_do_not_leak_into_another():
    c = _client()
    job_a = _start_job(c)
    job_b = _start_job(c)
    try:
        bind_job(job_a)
        create_task("a's task")
        bind_job(job_b)
        create_task("b's task")

        tasks_a = c.get(f"/api/work/{job_a}/tasks").json()["tasks"]
        tasks_b = c.get(f"/api/work/{job_b}/tasks").json()["tasks"]
        assert [t["subject"] for t in tasks_a] == ["a's task"]
        assert [t["subject"] for t in tasks_b] == ["b's task"]
    finally:
        clear_job(job_a)
        clear_job(job_b)
