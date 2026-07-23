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


def recent_messages(chat: dict, max_turns: int = 8) -> list[dict]:
    """Flatten the last `max_turns` user/assistant turns into plain
    {"role","content"} messages, for resending to Anamnesis alongside the
    new user message.

    Why this exists: Anamnesis's own "recency window" (selector.js
    recencyMsgs/recencyTurns, default 8 turns) is built from the CURRENT
    request's own `messages` array, not from its persisted turn DB. Pleiades'
    engine.py used to send ONLY the bare new user message every turn, so that
    recency floor had nothing to work with — a short pronoun-heavy follow-up
    ("message him again") could miss Anamnesis's semantic-similarity floor
    entirely and lose the subject. Resending the last few real turns here
    lets Anamnesis's existing, already-tested mechanism engage as designed.
    Confirmed safe against proxy.js: only the LAST user message in an
    incoming request is ever persisted as a new turn (deduped via
    history.lastUserTurnContent), so replaying older turns here does not
    create duplicate history rows.

    Tool-call detail is summarized inline ("[used tool X]") rather than
    reconstructed as OpenAI tool_calls/tool message pairs — this transcript
    format doesn't preserve exact tool_call_id linkage across turns, and a
    mismatched pairing can make some chat templates error out. A short
    inline note is enough context for continuity purposes.
    """
    msgs = chat.get("messages", [])
    tail = msgs[-(max_turns * 2):] if max_turns else msgs
    out: list[dict] = []
    for m in tail:
        role = m.get("role")
        if role == "user":
            content = m.get("content", "")
            if content:
                out.append({"role": "user", "content": content})
        elif role == "assistant":
            parts = []
            for item in m.get("items", []):
                if item.get("t") == "text" and item.get("text"):
                    parts.append(item["text"])
                elif item.get("t") == "tool":
                    parts.append(f"[used tool {item.get('name', '?')}]")
            content = "\n".join(parts).strip()
            if content:
                out.append({"role": "assistant", "content": content})
    return out


def append_turn(chat_id: str, user_text: str, assistant_items: list[dict],
                meta: dict) -> None:
    chat = load(chat_id)
    chat["messages"].append({"role": "user", "content": user_text})
    chat["messages"].append({"role": "assistant", "items": assistant_items,
                             "meta": meta})
    if not chat.get("title"):
        chat["title"] = user_text[:60] + ("…" if len(user_text) > 60 else "")
    save(chat)
