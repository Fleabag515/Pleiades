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
        {"t": "user_injected", "text": "..."},               # see below
     ], "meta": {"tokens": 123, "seconds": 4.2, "tps": 29.3}}

A `user_injected` item is a message the user sent WHILE this assistant turn
was still running (see webui/server.py's chats_interject endpoint and
engine.py's Engine.stream_events poll_injections param) — the model kept
working on the ORIGINAL user_text above, was interrupted between rounds/tool
calls with this extra message, and kept going with it woven in, all inside
one turn. It lives inside the assistant message's own `items` list (not as a
second top-level {"role":"user"} entry) because append_turn only ever
persists ONE user_text per turn; `recent_messages` below splits the
assistant's `items` back into separate role:user/assistant entries around
each `user_injected` item when reconstructing history for a future turn.
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


def validate_chat_id(chat_id: str) -> str:
    """Reject chat ids that aren't the plain hex/dash tokens `create()`
    generates -- shared with pleiades/attachments.py, which uses chat_id as
    a cache directory name and needs the exact same guarantee (no path
    separators, no traversal) that this module already relies on for the
    transcript file name below."""
    if not chat_id or not chat_id.replace("-", "").isalnum():
        raise ValueError(f"invalid chat id {chat_id!r}")
    return chat_id


def _path(chat_id: str) -> Path:
    validate_chat_id(chat_id)
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


def tool_usage(character: str | None = None) -> dict[str, dict]:
    """Real per-tool usage stats mined from persisted chat transcripts.

    Scans every chat file (optionally scoped to one character) and tallies
    completed tool calls ({"t": "tool", ..., "ok": true/false}) by tool
    name, returning {tool_name: {"count": int, "last_used": float|None}}.

    `last_used` is only ever a real wall-clock time recorded at the moment
    the tool was actually invoked (server.py's streaming loop stamps each
    tool item with "ts" as it's created -- see the comment there). Chats
    written before that stamp existed have no "ts" on their tool items;
    those calls still count toward `count` but never move `last_used`,
    because the only other timestamp available -- the chat file's
    `updated` field -- reflects whenever the chat was LAST saved, not when
    that specific historical tool call happened, and reporting it as
    "last used" would be reporting a guess as a fact. A tool that has
    `count > 0` but `last_used == None` was used before tracking existed;
    the UI should say so rather than inventing a time.
    """
    stats: dict[str, dict] = {}
    for f in chats_dir().glob("*.json"):
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if character is not None and c.get("character") != character:
            continue
        for m in c.get("messages", []):
            if m.get("role") != "assistant":
                continue
            for item in m.get("items", []):
                if item.get("t") != "tool" or item.get("ok") is None:
                    continue  # still in flight / malformed -- not a completed call
                name = item.get("name") or "?"
                s = stats.setdefault(name, {"count": 0, "last_used": None})
                s["count"] += 1
                ts = item.get("ts")
                if isinstance(ts, (int, float)) and (s["last_used"] is None or ts > s["last_used"]):
                    s["last_used"] = ts
    return stats


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
            # A user_injected item splits this ONE persisted assistant turn
            # back into the two-or-more-role sequence the model actually saw
            # live (assistant partial work, then the interjection landing as
            # its own user-role message, then whatever the assistant did
            # after it) -- see the user_injected note in this module's
            # docstring. Flush whatever assistant text/tool summary has
            # accumulated SO FAR whenever one is hit, emit it as its own
            # user-role entry with the same "[message from the user,
            # mid-task]" marker Engine.stream_events actually prepended (so a
            # future turn's resent history matches word-for-word what the
            # live turn contained), then keep accumulating anything the
            # assistant said/did after it into a fresh piece.
            parts: list[str] = []

            def _flush_assistant_parts() -> None:
                nonlocal parts
                content = "\n".join(parts).strip()
                if content:
                    out.append({"role": "assistant", "content": content})
                parts = []

            for item in m.get("items", []):
                t = item.get("t")
                if t == "text" and item.get("text"):
                    parts.append(item["text"])
                elif t == "tool":
                    parts.append(f"[used tool {item.get('name', '?')}]")
                elif t == "user_injected" and item.get("text"):
                    _flush_assistant_parts()
                    out.append({"role": "user",
                               "content": f"[message from the user, mid-task] {item['text']}"})
            _flush_assistant_parts()
    return out


def append_turn(chat_id: str, user_text: str, assistant_items: list[dict],
                meta: dict, attachments: "list[dict] | None" = None) -> None:
    """`attachments`, if given, is `[{"name", "path", "mime"}, ...]` for real
    files the user attached to THIS message (see pleiades/attachments.py and
    webui/server.py's chats_message) -- persisted on the user message so a
    later reload/replay (and eventually MessageBubble.tsx) can show what was
    attached. Only added to the stored message when non-empty, so chats with
    no attachments keep the exact same on-disk shape as before this existed.
    """
    chat = load(chat_id)
    user_msg: dict = {"role": "user", "content": user_text}
    if attachments:
        user_msg["attachments"] = attachments
    chat["messages"].append(user_msg)
    chat["messages"].append({"role": "assistant", "items": assistant_items,
                             "meta": meta})
    if not chat.get("title"):
        chat["title"] = user_text[:60] + ("…" if len(user_text) > 60 else "")
    save(chat)
