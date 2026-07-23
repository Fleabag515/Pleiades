"""Per-chat attachment cache.

Gives the character a REAL file it can point its existing file/shell tools
at (see engine.py's `_attachments_note`, injected the same way as
`_environment_note`) -- e.g. drop a logo image into the composer, then ask
the model to "use it on the site": the model gets a real absolute path, not
a text description of the image. Deliberately does none of the "can a model
see/hear this" work (vision content-parts, image captioning, audio
transcription are separate, parallel efforts) -- this module only answers
"where is the file", which is useful even for a text-only model.

Cache layout: PROFILES_DIR/<character>/chat-cache/<chat_id>/<sanitized-name>

Scoped by chat_id, NOT a flat shared directory, on purpose: profile_dir()'s
existing `workspace/` directory is already shared across every chat a
character has AND every background/scheduled-task job for that character
(see pleiades/scheduler.py) -- two different chats, or a chat and a
concurrently-running scheduled task, dropping same-named files there would
silently collide/overwrite. Namespacing by chat_id avoids that: each chat
gets its own directory, for free, the moment it's created.

Lifecycle -- deliberately NOT wiped when a new chat starts. Fleagle's literal
ask was "cleared on new-chat", but `startNewChat` (desktop's chatStore.tsx)
does not delete the outgoing chat: it archives it into History, where it
stays viewable. If this cache were wiped on new-chat, any image the user
referenced earlier in that now-archived chat would 404 the moment you
started a fresh conversation -- breaking a still-visible chat to satisfy a
literal reading of a request whose actual intent was "don't let old chats'
attachments leak into new ones". Scoping by chat_id already gives a brand
new chat an empty, private cache directory with zero extra code, which IS
"cleared" from that new chat's point of view. Actual deletion happens when a
chat is genuinely deleted (`DELETE /api/chats/{id}`, see
`pleiades/webui/server.py`'s `chats_delete`), via `delete_chat_cache()`
below. Do not "fix" this back to wiping on new-chat -- that reintroduces the
broken-History-image bug this design avoids.
"""

from __future__ import annotations

import mimetypes
import re
import shutil
from pathlib import Path

from . import config
from .chats import validate_chat_id

_NAME_BAD = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_filename(name: str) -> str:
    """Reduce an arbitrary (possibly hostile) client-supplied filename to a
    safe basename.

    `Path(...).name` alone already defeats both `../../etc/passwd`-style
    traversal and absolute paths (`/etc/passwd`, `C:\\Windows\\...`) by
    keeping only the last path segment; everything left that isn't
    alnum/dot/dash/underscore is replaced with `_`. Leading dots are
    stripped too (blocks hidden-file names and a bare `..` surviving as
    itself). Never returns an empty string.
    """
    base = Path(name.replace("\\", "/")).name
    base = _NAME_BAD.sub("_", base)
    base = base.lstrip(".") or "file"
    return base[:200]


def _cache_root(character: str, chat_id: str) -> Path:
    validate_chat_id(chat_id)
    return config.profile_dir(character) / "chat-cache" / chat_id


def chat_cache_dir(character: str, chat_id: str) -> Path:
    """This chat's own attachment cache directory, created on demand."""
    d = _cache_root(character, chat_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _unique_path(d: Path, filename: str) -> Path:
    """The same filename uploaded twice into the SAME chat (e.g. two pasted
    clipboard screenshots both literally called "image.png") must not
    silently clobber each other -- append a short numeric suffix instead of
    overwriting."""
    path = d / filename
    if not path.exists():
        return path
    stem, dot, ext = filename.partition(".")
    n = 1
    while True:
        candidate_name = f"{stem}-{n}{dot}{ext}" if dot else f"{filename}-{n}"
        candidate = d / candidate_name
        if not candidate.exists():
            return candidate
        n += 1


def save_attachment(character: str, chat_id: str, filename: str, data: bytes) -> dict:
    """Save one uploaded file into this chat's cache dir.

    Returns {id, filename, path, mime, size}. `id` doubles as the on-disk
    filename -- it's already unique within this chat's own cache directory
    (see `_unique_path`), and that directory is already namespaced by
    chat_id, so `(chat_id, id)` is a complete address and no separate
    id -> path registry/database is needed.
    """
    d = chat_cache_dir(character, chat_id)
    safe = sanitize_filename(filename)
    path = _unique_path(d, safe)
    path.write_bytes(data)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {
        "id": path.name,
        "filename": path.name,
        "path": str(path),
        "mime": mime,
        "size": len(data),
    }


def resolve_attachments(character: str, chat_id: str, ids: list[str]) -> list[dict]:
    """Turn attachment ids (see `save_attachment`) back into real, existing
    file info -- `{name, path, mime}` -- for path-injection into the turn's
    context and for persisting on the user message.

    Ids that don't round-trip through `sanitize_filename` unchanged (i.e.
    someone hand-crafted a traversal attempt in the request body, like
    `../../etc/passwd`) or that don't exist on disk are silently skipped
    rather than raising -- a stale or bad id shouldn't fail the whole turn.
    """
    d = chat_cache_dir(character, chat_id)
    out: list[dict] = []
    for raw_id in ids:
        safe = sanitize_filename(raw_id)
        if safe != raw_id:
            continue
        path = d / safe
        if not path.is_file():
            continue
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        out.append({"name": path.name, "path": str(path), "mime": mime})
    return out


def attachment_path(character: str, chat_id: str, filename: str) -> Path | None:
    """Resolve one attachment's filename back to its real on-disk path, for
    serving the raw bytes back over HTTP (see webui/server.py's
    `chats_attachment_get`, used by the desktop UI to render a persisted
    attachment chip's thumbnail/`<audio>` player against a real URL).

    Same traversal guard as `resolve_attachments`: a filename that doesn't
    round-trip through `sanitize_filename` unchanged (a hand-crafted
    `../../etc/passwd`-style id), or that doesn't exist on disk, resolves to
    None rather than raising -- the route turns that into a plain 404.
    """
    safe = sanitize_filename(filename)
    if safe != filename:
        return None
    path = chat_cache_dir(character, chat_id) / safe
    if not path.is_file():
        return None
    return path


def delete_chat_cache(character: str, chat_id: str) -> bool:
    """Delete this chat's entire attachment cache directory.

    Called from `DELETE /api/chats/{id}` (a chat actually being deleted),
    never from new-chat -- see this module's docstring for why.
    """
    d = _cache_root(character, chat_id)
    if d.is_dir():
        shutil.rmtree(d)
        return True
    return False
