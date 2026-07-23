"""Per-chat attachment cache (pleiades/attachments.py) + its wiring into
webui/server.py and engine.py's path-injection note.

Covers: the upload endpoint (saves file, returns correct metadata, rejects
path-traversal filenames), chat_id-scoping (two chats with same-named
uploads don't collide), the delete-cascade (deleting a chat removes its
cache dir), and the attachments-note actually reaching a turn's messages.
"""

from pathlib import Path

import pytest

from pleiades import attachments, chats, config
from pleiades.engine import Engine

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from pleiades.profiles import Profile, ProfileManager  # noqa: E402
from pleiades.webui import create_app  # noqa: E402


def _client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


def _make_chat(character: str) -> dict:
    pm = ProfileManager(config.Settings.load(), anamnesis=object())
    pm._save(Profile(name=character, exec_policy="allow"))
    return chats.create(character)


# --------------------------------------------------------------------------- #
# sanitize_filename / save_attachment -- pure unit tests, no server involved
# --------------------------------------------------------------------------- #

def test_sanitize_filename_strips_traversal_and_absolute_paths():
    assert attachments.sanitize_filename("../../etc/passwd") == "passwd"
    assert attachments.sanitize_filename("/etc/passwd") == "passwd"
    assert attachments.sanitize_filename("..\\..\\windows\\system32\\evil.exe") == "evil.exe"
    assert attachments.sanitize_filename("..") == "file"  # bare '..' -> fallback
    assert attachments.sanitize_filename("") == "file"


def test_sanitize_filename_replaces_hostile_characters():
    safe = attachments.sanitize_filename("weird name;rm -rf $HOME.png")
    assert "/" not in safe and ";" not in safe and " " not in safe
    assert safe.endswith(".png")


def test_save_attachment_returns_correct_metadata_and_writes_real_file():
    chat = _make_chat("attach-meta-char")
    info = attachments.save_attachment(chat["character"], chat["id"], "logo.png", b"\x89PNGDATA")
    assert info["id"] == "logo.png"
    assert info["filename"] == "logo.png"
    assert info["mime"] == "image/png"
    assert info["size"] == len(b"\x89PNGDATA")
    saved = Path(info["path"])
    assert saved.is_file()
    assert saved.read_bytes() == b"\x89PNGDATA"
    # Lives under this chat's own scoped cache dir, not a flat shared one.
    assert saved.parent == config.profile_dir(chat["character"]) / "chat-cache" / chat["id"]


def test_save_attachment_sanitizes_traversal_filename_and_stays_inside_cache_dir():
    chat = _make_chat("attach-traversal-char")
    info = attachments.save_attachment(chat["character"], chat["id"], "../../../etc/passwd", b"nope")
    saved = Path(info["path"])
    assert saved.parent == config.profile_dir(chat["character"]) / "chat-cache" / chat["id"]
    assert saved.name == "passwd"
    assert not (Path("/etc/passwd").read_bytes() == b"nope")  # real /etc/passwd untouched


def test_same_filename_twice_in_same_chat_does_not_collide():
    chat = _make_chat("attach-dup-char")
    first = attachments.save_attachment(chat["character"], chat["id"], "image.png", b"AAAA")
    second = attachments.save_attachment(chat["character"], chat["id"], "image.png", b"BBBB")
    assert first["path"] != second["path"]
    assert Path(first["path"]).read_bytes() == b"AAAA"
    assert Path(second["path"]).read_bytes() == b"BBBB"


def test_chat_id_scoping_two_chats_same_filename_do_not_collide():
    chat_a = _make_chat("attach-scope-char-a")
    chat_b = _make_chat("attach-scope-char-b")
    info_a = attachments.save_attachment(chat_a["character"], chat_a["id"], "shared-name.txt", b"from A")
    info_b = attachments.save_attachment(chat_b["character"], chat_b["id"], "shared-name.txt", b"from B")
    assert info_a["path"] != info_b["path"]
    assert Path(info_a["path"]).read_bytes() == b"from A"
    assert Path(info_b["path"]).read_bytes() == b"from B"
    assert info_a["id"] == "shared-name.txt"
    assert info_b["id"] == "shared-name.txt"  # same id string is fine -- scoped by chat_id


def test_resolve_attachments_skips_traversal_ids_and_missing_files():
    chat = _make_chat("attach-resolve-char")
    saved = attachments.save_attachment(chat["character"], chat["id"], "real.txt", b"hi")
    ids = [saved["id"], "../../etc/passwd", "does-not-exist.txt"]
    resolved = attachments.resolve_attachments(chat["character"], chat["id"], ids)
    assert len(resolved) == 1
    assert resolved[0]["name"] == "real.txt"
    assert resolved[0]["path"] == saved["path"]


def test_delete_chat_cache_removes_directory():
    chat = _make_chat("attach-delete-char")
    attachments.save_attachment(chat["character"], chat["id"], "keep-me.png", b"data")
    cache_dir = config.profile_dir(chat["character"]) / "chat-cache" / chat["id"]
    assert cache_dir.is_dir()
    assert attachments.delete_chat_cache(chat["character"], chat["id"]) is True
    assert not cache_dir.exists()
    # Deleting an already-gone/never-existed cache is a harmless no-op.
    assert attachments.delete_chat_cache(chat["character"], chat["id"]) is False


# --------------------------------------------------------------------------- #
# Upload endpoint (through the real FastAPI app)
# --------------------------------------------------------------------------- #

def test_upload_endpoint_saves_file_and_returns_metadata():
    chat = _make_chat("attach-api-char")
    c = _client()
    resp = c.post(
        f"/api/chats/{chat['id']}/attachments",
        files={"file": ("photo.jpg", b"fakejpegbytes", "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "photo.jpg"
    assert body["mime"] == "image/jpeg"
    assert body["size"] == len(b"fakejpegbytes")
    assert Path(body["path"]).read_bytes() == b"fakejpegbytes"
    assert Path(body["path"]).parent == config.profile_dir("attach-api-char") / "chat-cache" / chat["id"]


def test_upload_endpoint_rejects_path_traversal_filename_stays_in_cache_dir():
    chat = _make_chat("attach-api-traversal-char")
    c = _client()
    resp = c.post(
        f"/api/chats/{chat['id']}/attachments",
        files={"file": ("../../../../tmp/evil.txt", b"payload", "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    saved = Path(body["path"])
    assert saved.name == "evil.txt"
    assert saved.parent == config.profile_dir("attach-api-traversal-char") / "chat-cache" / chat["id"]
    assert not Path("/tmp/evil.txt").exists()


def test_upload_endpoint_404s_for_unknown_chat():
    c = _client()
    resp = c.post(
        "/api/chats/nonexistent12/attachments",
        files={"file": ("a.txt", b"x", "text/plain")},
    )
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Serving endpoint: GET /api/chats/{id}/attachments/{filename} -- streams a
# previously-uploaded attachment's raw bytes back (added so the desktop UI
# can render a real <img>/<audio> against persisted attachments -- see
# MessageBubble.tsx's AttachmentChip + lib/api.ts's attachmentUrl).
# --------------------------------------------------------------------------- #

def test_get_attachment_endpoint_streams_the_real_bytes_back():
    chat = _make_chat("attach-get-char")
    c = _client()
    upload = c.post(
        f"/api/chats/{chat['id']}/attachments",
        files={"file": ("photo.png", b"\x89PNGrealbytes", "image/png")},
    )
    assert upload.status_code == 200
    filename = upload.json()["filename"]

    resp = c.get(f"/api/chats/{chat['id']}/attachments/{filename}")
    assert resp.status_code == 200
    assert resp.content == b"\x89PNGrealbytes"
    assert resp.headers["content-type"] == "image/png"


def test_get_attachment_endpoint_404s_for_unknown_chat():
    c = _client()
    resp = c.get("/api/chats/nonexistent12/attachments/photo.png")
    assert resp.status_code == 404


def test_get_attachment_endpoint_404s_for_missing_file():
    chat = _make_chat("attach-get-missing-char")
    c = _client()
    resp = c.get(f"/api/chats/{chat['id']}/attachments/does-not-exist.png")
    assert resp.status_code == 404


def test_get_attachment_endpoint_rejects_path_traversal_filename():
    """Mirrors the upload endpoint's own traversal test: a hand-crafted
    `../../etc/passwd`-style filename must 404, not escape this chat's
    cache dir and read an arbitrary file off disk."""
    chat = _make_chat("attach-get-traversal-char")
    c = _client()
    c.post(
        f"/api/chats/{chat['id']}/attachments",
        files={"file": ("real.txt", b"hello", "text/plain")},
    )

    resp = c.get(f"/api/chats/{chat['id']}/attachments/..%2f..%2f..%2fetc%2fpasswd")
    assert resp.status_code == 404

    # A same-named file that legitimately exists elsewhere on disk (e.g. a
    # sibling chat's cache) must never leak through a traversal attempt
    # either -- attachment_path() re-sanitizes and compares, it doesn't
    # trust the incoming filename at all.
    other_chat = _make_chat("attach-get-traversal-other-char")
    attachments.save_attachment(other_chat["character"], other_chat["id"], "secret.txt", b"not yours")
    resp2 = c.get(f"/api/chats/{chat['id']}/attachments/../{other_chat['id']}/secret.txt")
    assert resp2.status_code == 404


def test_attachment_path_helper_rejects_traversal_and_missing_files():
    chat = _make_chat("attach-path-helper-char")
    saved = attachments.save_attachment(chat["character"], chat["id"], "real.txt", b"hi")

    assert attachments.attachment_path(chat["character"], chat["id"], saved["id"]) == Path(saved["path"])
    assert attachments.attachment_path(chat["character"], chat["id"], "../../etc/passwd") is None
    assert attachments.attachment_path(chat["character"], chat["id"], "does-not-exist.txt") is None


# --------------------------------------------------------------------------- #
# Delete-cascade: DELETE /api/chats/{id} removes that chat's cache dir
# --------------------------------------------------------------------------- #

def test_delete_chat_cascades_to_cache_dir():
    chat = _make_chat("attach-cascade-char")
    c = _client()
    c.post(
        f"/api/chats/{chat['id']}/attachments",
        files={"file": ("cascade.png", b"bytes", "image/png")},
    )
    cache_dir = config.profile_dir("attach-cascade-char") / "chat-cache" / chat["id"]
    assert cache_dir.is_dir()

    resp = c.delete(f"/api/chats/{chat['id']}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert not cache_dir.exists()


def test_new_chat_does_not_touch_a_different_chats_cache():
    """Regression guard for the "cleared on new-chat" misreading: creating a
    brand-new chat for the same character must never delete or otherwise
    touch an existing chat's attachment cache."""
    old_chat = _make_chat("attach-newchat-char")
    attachments.save_attachment(old_chat["character"], old_chat["id"], "still-here.png", b"data")
    old_cache = config.profile_dir("attach-newchat-char") / "chat-cache" / old_chat["id"]
    assert old_cache.is_dir()

    new_chat = chats.create("attach-newchat-char")
    assert new_chat["id"] != old_chat["id"]
    new_cache = config.profile_dir("attach-newchat-char") / "chat-cache" / new_chat["id"]
    assert not new_cache.exists()  # fresh chat -> fresh, empty cache dir
    assert old_cache.is_dir()      # old chat's cache untouched
    assert (old_cache / "still-here.png").is_file()


# --------------------------------------------------------------------------- #
# Path-injection note -- engine.py
# --------------------------------------------------------------------------- #

def test_attachments_note_lists_real_paths():
    note = Engine._attachments_note([
        {"name": "logo.png", "path": "/home/fleabag/.pleiades/profiles/zoe/chat-cache/abc/logo.png",
         "mime": "image/png"},
    ])
    assert "logo.png" in note
    assert "/home/fleabag/.pleiades/profiles/zoe/chat-cache/abc/logo.png" in note
    assert "REAL files" in note


def test_attachments_note_folds_into_base_messages_system_prompt():
    env_note = "ENV-MARKER-LINE"
    attach_note = Engine._attachments_note(
        [{"name": "logo.png", "path": "/tmp/logo.png", "mime": "image/png"}])
    combined = f"{env_note}\n\n{attach_note}"
    msgs = Engine._base_messages("use it on the site", None, combined)
    sys_text = msgs[0]["content"]
    assert "ENV-MARKER-LINE" in sys_text
    assert "/tmp/logo.png" in sys_text
    assert "logo.png" in sys_text


def test_chats_message_passes_attachments_note_into_stream_events(monkeypatch):
    """End-to-end through the real POST /message route: an `attachments`
    id list on the request must resolve to a real path and reach
    Engine.stream_events as `attachments_note`, containing that real path."""
    chat = _make_chat("attach-turn-char")
    saved = attachments.save_attachment(chat["character"], chat["id"], "logo.png", b"pngdata")

    captured = {}

    def fake_stream_events(self, profile, user_message, *, system=None,
                           history=None, attachments_note=None,
                           should_stop=None, poll_injections=None,
                           attachments=None):
        captured["attachments_note"] = attachments_note
        captured["attachments"] = attachments
        yield {"type": "token", "text": "ok"}
        yield {"type": "done", "tokens": 1, "seconds": 0.01, "tps": 1.0}

    monkeypatch.setattr(Engine, "stream_events", fake_stream_events)
    c = _client()
    resp = c.post(f"/api/chats/{chat['id']}/message",
                  json={"message": "use it on the site", "attachments": [saved["id"]]})
    assert resp.status_code == 200
    assert captured["attachments_note"] is not None
    assert saved["path"] in captured["attachments_note"]
    assert "logo.png" in captured["attachments_note"]

    # And the persisted transcript records the attachment on the user message.
    loaded = chats.load(chat["id"])
    user_msg = loaded["messages"][0]
    assert user_msg["role"] == "user"
    assert user_msg["attachments"][0]["name"] == "logo.png"
    assert user_msg["attachments"][0]["path"] == saved["path"]


def test_chats_message_without_attachments_does_not_pass_attachments_note(monkeypatch):
    """Back-compat guard: when there are no attachments, `attachments_note`
    must not even be passed as a kwarg (existing tests monkeypatch
    Engine.stream_events with a narrower fake signature that doesn't accept
    it -- see test_webui_chat_concurrency.py)."""
    chat = _make_chat("attach-noattach-char")

    def fake_stream_events(self, profile, user_message, *, system=None,
                           history=None, should_stop=None, poll_injections=None):
        yield {"type": "token", "text": "ok"}
        yield {"type": "done", "tokens": 1, "seconds": 0.01, "tps": 1.0}

    monkeypatch.setattr(Engine, "stream_events", fake_stream_events)
    c = _client()
    resp = c.post(f"/api/chats/{chat['id']}/message", json={"message": "hi"})
    assert resp.status_code == 200

    loaded = chats.load(chat["id"])
    user_msg = loaded["messages"][0]
    assert "attachments" not in user_msg
