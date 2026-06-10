"""The control-panel API must defend localhost against DNS-rebinding and CSRF.

POST /api/work runs the full agent tool belt (shell/files), so a page the user
merely visits must not be able to drive it. The loopback guard validates Host
and Origin against a loopback allow-list.
"""

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from pleiades.webui import create_app  # noqa: E402


def _client() -> TestClient:
    # base_url sets a loopback Host by default; individual requests override it.
    return TestClient(create_app(), base_url="http://127.0.0.1")


def test_same_origin_loopback_request_allowed():
    c = _client()
    r = c.get("/api/email/presets", headers={"host": "127.0.0.1:8750"})
    assert r.status_code == 200


def test_localhost_host_allowed():
    c = _client()
    r = c.get("/api/email/presets", headers={"host": "localhost:8750"})
    assert r.status_code == 200


def test_rebinding_foreign_host_blocked():
    # DNS rebinding: attacker.example resolves to 127.0.0.1, so the request
    # arrives with the attacker's hostname in Host. Must be rejected.
    c = _client()
    r = c.get("/api/email/presets", headers={"host": "attacker.example:8750"})
    assert r.status_code == 403


def test_cross_origin_post_blocked():
    # A cross-site page POSTing to the agent runner must be refused.
    c = _client()
    r = c.post("/api/work",
               headers={"host": "127.0.0.1:8750", "origin": "http://evil.example"},
               json={"task": "rm -rf ~", "policy": "allow"})
    assert r.status_code == 403


def test_cross_origin_referer_blocked():
    c = _client()
    r = c.get("/api/email/presets",
              headers={"host": "127.0.0.1:8750", "referer": "http://evil.example/x"})
    assert r.status_code == 403


def test_guard_can_be_disabled_for_intentional_exposure():
    # Wildcard binds opt out (operator's explicit choice); no Host filtering.
    from pleiades.webui import create_app as ca
    c = TestClient(ca(enforce_guard=False), base_url="http://127.0.0.1")
    r = c.get("/api/email/presets", headers={"host": "anything.example"})
    assert r.status_code == 200
