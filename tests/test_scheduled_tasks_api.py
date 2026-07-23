"""GET/POST/PATCH/DELETE /api/scheduled-tasks -- the desktop app's Scheduled
tasks panel and the create_scheduled_task/etc. harness tools share this one
store (pleiades.scheduler.ScheduledTaskManager); this file only exercises the
HTTP surface."""

import time

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from pleiades.webui import create_app  # noqa: E402


def _client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


def test_create_list_patch_delete_cron_task():
    c = _client()
    r = c.post("/api/scheduled-tasks", json={"task": "daily summary", "cron": "0 7 * * *"})
    assert r.status_code == 200
    body = r.json()
    assert body["cron"] == "0 7 * * *" and body["next_run"] > time.time()
    tid = body["id"]

    listed = c.get("/api/scheduled-tasks").json()["tasks"]
    assert any(t["id"] == tid for t in listed)

    patched = c.patch(f"/api/scheduled-tasks/{tid}", json={"enabled": False})
    assert patched.status_code == 200 and patched.json()["enabled"] is False

    deleted = c.delete(f"/api/scheduled-tasks/{tid}")
    assert deleted.status_code == 200
    assert c.delete(f"/api/scheduled-tasks/{tid}").status_code == 404


def test_create_one_time_task():
    c = _client()
    r = c.post("/api/scheduled-tasks",
              json={"task": "remind me", "fire_at": time.time() + 3600,
                    "character": "Mark"})
    assert r.status_code == 200
    body = r.json()
    assert body["character"] == "Mark" and body["fire_at"] > 0


def test_create_rejects_bad_cron():
    c = _client()
    r = c.post("/api/scheduled-tasks", json={"task": "bad", "cron": "99 * * * *"})
    assert r.status_code == 400


def test_create_rejects_both_cron_and_fire_at():
    c = _client()
    r = c.post("/api/scheduled-tasks",
              json={"task": "bad", "cron": "0 7 * * *", "fire_at": time.time() + 60})
    assert r.status_code == 400


def test_patch_unknown_task_404_via_manager_error():
    c = _client()
    r = c.patch("/api/scheduled-tasks/does-not-exist", json={"task": "x"})
    assert r.status_code == 400  # ScheduleError -> 400, not a route-level 404


def test_delete_unknown_task_is_404():
    c = _client()
    r = c.delete("/api/scheduled-tasks/does-not-exist")
    assert r.status_code == 404


def test_run_now_fires_task_immediately():
    c = _client()
    created = c.post("/api/scheduled-tasks",
                     json={"task": "manual run via api", "fire_at": time.time() + 3600})
    tid = created.json()["id"]

    r = c.post(f"/api/scheduled-tasks/{tid}/run")
    assert r.status_code == 200
    body = r.json()
    assert body["last_run"] > 0
    assert body["enabled"] is False   # one-time task fired -> disabled


def test_run_now_unknown_task_is_404():
    c = _client()
    r = c.post("/api/scheduled-tasks/does-not-exist/run")
    assert r.status_code == 404
