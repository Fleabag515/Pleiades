"""Persistent chat sessions for the web UI.

A chat is a transcript file under ~/.pleiades/chats/<id>.json bound to one
character. The transcript is presentation state for the UI; the character's
actual long-term memory lives in Anamnesis and is unaffected by deleting chats.

Message shape:
    {"role": "user", "content": "..."}
    {"role": "assistant", "items": [
        {"t": "text", "text": "..."},                       # may contain <think> tags
        {"t": "tool", "name": "...", "args": "...",
         "output": "...", "ok": true},
     ], "meta": {"tokens": 123, "seconds": 4.2, "tps": 29.3}}
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from . import config


def chats_dir() -> Path:
    d = config.PLEIADES_HOME / "chats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(chat_id: str) -> Path:
    if not chat_id.replace("-", "").isalnum():
        raise ValueError(f"invalid chat id {chat_id!r}")
    return chats_dir() / f"{chat_id}.json"


def create(character: str) -> dict:
    chat = {"id": uuid.uuid4().hex[:12], "character": character, "title": "",
            "created": time.time(), "updated": time.time(), "messages": []}
    save(chat)
    return chat


def save(chat: dict) -> None:
    chat["updated"] = time.time()
    _path(chat["id"]).write_text(json.dumps(chat, ensure_ascii=False, indent=1),
                                 encoding="utf-8")


def load(chat_id: str) -> dict:
    p = _path(chat_id)
    if not p.is_file():
        raise FileNotFoundError(f"no chat {chat_id}")
    return json.loads(p.read_text(encoding="utf-8"))


def delete(chat_id: str) -> bool:
    p = _path(chat_id)
    if p.is_file():
        p.unlink()
        return True
    return False


def list_chats() -> list[dict]:
    out = []
    for f in chats_dir().glob("*.json"):
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
            out.append({"id": c["id"], "character": c.get("character", ""),
                        "title": c.get("title") or "(new chat)",
                        "updated": c.get("updated", 0),
                        "messages": len(c.get("messages", []))})
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    return sorted(out, key=lambda c: -c["updated"])


def append_turn(chat_id: str, user_text: str, assistant_items: list[dict],
                meta: dict) -> None:
    chat = load(chat_id)
    chat["messages"].append({"role": "user", "content": user_text})
    chat["messages"].append({"role": "assistant", "items": assistant_items,
                             "meta": meta})
    if not chat.get("title"):
        chat["title"] = user_text[:60] + ("…" if len(user_text) > 60 else "")
    save(chat)
