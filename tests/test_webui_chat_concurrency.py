"""Mid-turn interjection substrate (owner brief: message an in-flight chat
turn like a Claude Code agent, without aborting it) + the per-chat in-flight
lock that fixes a real pre-existing bug: nothing used to stop two concurrent
POST /message calls for the SAME chat_id from clobbering each other's
chat_stops/chat_approvals entries and interleaving two turns into one
transcript.

These tests replace Engine.stream_events with small controlled fakes --
exercising the real webui/server.py route/lock/inbox/persistence code, not a
real llama.cpp round trip."""

import threading

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from pleiades import chats, config  # noqa: E402
from pleiades.engine import Engine  # noqa: E402
from pleiades.profiles import Profile, ProfileManager  # noqa: E402
from pleiades.webui import create_app  # noqa: E402


def _client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


def _make_chat(character: str) -> dict:
    # Write the profile straight to disk -- bypasses ProfileManager.create()'s
    # real Anamnesis character-creation call, which would need a live daemon.
    pm = ProfileManager(config.Settings.load(), anamnesis=object())
    pm._save(Profile(name=character, exec_policy="allow"))
    return chats.create(character)


def _lines(resp) -> list[dict]:
    import json
    out = []
    for line in resp.text.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def test_concurrent_message_posts_same_chat_get_409(monkeypatch):
    """The real bug the council found: two POSTs racing for the same chat_id
    used to both drive the engine and both write to chat_stops/chat_approvals.
    The per-chat lock must turn the second one into a clean 409 instead."""
    chat = _make_chat("lock-test-char")
    started = threading.Event()
    release = threading.Event()

    def fake_stream_events(self, profile, user_message, *, system=None,
                           history=None, should_stop=None, poll_injections=None):
        started.set()
        release.wait(timeout=5)
        yield {"type": "token", "text": "ok"}
        yield {"type": "done", "tokens": 1, "seconds": 0.01, "tps": 100.0}

    monkeypatch.setattr(Engine, "stream_events", fake_stream_events)

    c = _client()
    result = {}

    def _run_first():
        result["resp"] = c.post(f"/api/chats/{chat['id']}/message", json={"message": "hi"})

    t = threading.Thread(target=_run_first)
    t.start()
    assert started.wait(timeout=5), "first request never reached the engine"

    # Second concurrent POST for the SAME chat -- must be rejected, not race.
    second = c.post(f"/api/chats/{chat['id']}/message", json={"message": "hi again"})
    assert second.status_code == 409

    release.set()
    t.join(timeout=5)
    assert result["resp"].status_code == 200
    events = _lines(result["resp"])
    assert any(e["type"] == "done" for e in events)


def test_lock_released_after_turn_next_message_succeeds(monkeypatch):
    """Once a turn finishes (lock released in gen()'s finally), a brand-new
    POST /message for the same chat must succeed normally -- the lock must
    not leak across turns."""
    chat = _make_chat("lock-test-char-2")

    def fake_stream_events(self, profile, user_message, *, system=None,
                           history=None, should_stop=None, poll_injections=None):
        yield {"type": "token", "text": "ok"}
        yield {"type": "done", "tokens": 1, "seconds": 0.01, "tps": 100.0}

    monkeypatch.setattr(Engine, "stream_events", fake_stream_events)
    c = _client()

    first = c.post(f"/api/chats/{chat['id']}/message", json={"message": "one"})
    assert first.status_code == 200
    second = c.post(f"/api/chats/{chat['id']}/message", json={"message": "two"})
    assert second.status_code == 200


def test_interject_without_in_flight_turn_is_409():
    chat = _make_chat("interject-test-char")
    c = _client()
    r = c.post(f"/api/chats/{chat['id']}/interject", json={"message": "hello?"})
    assert r.status_code == 409


def test_interject_delivers_into_in_flight_turn_and_persists(monkeypatch):
    """POST .../interject while a turn is in flight must reach the engine's
    poll_injections() (not start a second turn), surface as a `user_injected`
    NDJSON event, and persist into the saved transcript so a reload/replay
    sees it."""
    chat = _make_chat("interject-test-char-2")
    ready = threading.Event()
    delivered = threading.Event()

    def fake_stream_events(self, profile, user_message, *, system=None,
                           history=None, should_stop=None, poll_injections=None):
        yield {"type": "token", "text": "working on it"}
        ready.set()
        # Poll for up to a couple seconds for the interjection to land --
        # mirrors polling at a safe round boundary in the real engine.
        import time as _t
        deadline = _t.time() + 5
        found = []
        while _t.time() < deadline and not found:
            found = poll_injections() if poll_injections else []
            if not found:
                _t.sleep(0.02)
        for text in found:
            yield {"type": "user_injected", "text": text}
        delivered.set()
        yield {"type": "token", "text": "done with the update"}
        yield {"type": "done", "tokens": 2, "seconds": 0.01, "tps": 100.0}

    monkeypatch.setattr(Engine, "stream_events", fake_stream_events)
    c = _client()
    result = {}

    def _run():
        result["resp"] = c.post(f"/api/chats/{chat['id']}/message", json={"message": "start task"})

    t = threading.Thread(target=_run)
    t.start()
    assert ready.wait(timeout=5), "turn never started"

    r = c.post(f"/api/chats/{chat['id']}/interject", json={"message": "actually make it shorter"})
    assert r.status_code == 200

    assert delivered.wait(timeout=5), "interjection never reached poll_injections"
    t.join(timeout=5)

    events = _lines(result["resp"])
    injected = [e for e in events if e["type"] == "user_injected"]
    assert injected and injected[0]["text"] == "actually make it shorter"

    loaded = chats.load(chat["id"])
    assistant_msg = loaded["messages"][-1]
    assert assistant_msg["role"] == "assistant"
    assert any(it.get("t") == "user_injected" and it.get("text") == "actually make it shorter"
              for it in assistant_msg["items"])


def test_undelivered_injection_surfaced_not_silently_dropped(monkeypatch):
    """Edge case: an interjection arrives but the engine never gets a chance
    to poll_injections() again before the turn ends (e.g. it lands right as
    the final answer is produced). It must NOT be silently eaten -- gen()'s
    finally block drains anything left in the inbox and yields an
    `undelivered` event so the frontend can offer to resend it."""
    chat = _make_chat("undelivered-test-char")
    ready = threading.Event()
    finish = threading.Event()

    def fake_stream_events(self, profile, user_message, *, system=None,
                           history=None, should_stop=None, poll_injections=None):
        # Deliberately never calls poll_injections() again after this point --
        # simulates the turn ending before it gets a chance to drain.
        yield {"type": "token", "text": "here's the final answer"}
        ready.set()
        finish.wait(timeout=5)
        yield {"type": "done", "tokens": 1, "seconds": 0.01, "tps": 100.0}

    monkeypatch.setattr(Engine, "stream_events", fake_stream_events)
    c = _client()
    result = {}

    def _run():
        result["resp"] = c.post(f"/api/chats/{chat['id']}/message", json={"message": "hi"})

    t = threading.Thread(target=_run)
    t.start()
    assert ready.wait(timeout=5), "turn never started"

    r = c.post(f"/api/chats/{chat['id']}/interject", json={"message": "wait, one more thing"})
    assert r.status_code == 200

    finish.set()
    t.join(timeout=5)

    events = _lines(result["resp"])
    undelivered = [e for e in events if e["type"] == "undelivered"]
    assert undelivered and undelivered[0]["text"] == "wait, one more thing"
    # And it must not have been silently folded into the persisted transcript
    # as if the model actually saw it.
    loaded = chats.load(chat["id"])
    assistant_msg = loaded["messages"][-1]
    assert not any(it.get("t") == "user_injected" for it in assistant_msg["items"])
