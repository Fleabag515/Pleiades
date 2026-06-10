"""
anamnesis.py — the harness's IN-PROCESS memory tier (working scratchpad + long-term
notes + semantic recall). It AUGMENTS, and does not replace, the canonical Anamnesis
context manager: the cross-session memory daemon/proxy lives in
``pleiades.anamnesis`` (a thin client to the external Anamnesis service that injects
and retrieves memory around every model turn). When a character is bound
(``pleiades.harness.identity.bind_character``) this tier is scoped to that character's
own ``~/.pleiades/profiles/<name>/memory`` dir, so a profile's working memory is part
of its identity. The embedding/hybrid-recall logic here is the "improvement" folded
into the Anamnesis subsystem.

----
Two tiers, one store.

  WORKING memory  (the agent's live scratchpad)
      A single rolling markdown doc the agent reads and writes *during* a task —
      current goal, plan, findings so far, open questions. Injected into the
      system prompt at the start of every run, so the agent always resumes with
      its own state in front of it. Persists across runs (continuity), capped so
      it can't grow without bound. This is the "anamnesis = working memory" model.

  LONG-TERM memory  (durable facts)
      A flat directory of small markdown notes, one fact each, with keyword
      recall. Stable knowledge the agent chooses to keep (user prefs, project
      facts, references). Surfaced on demand via recall().

Working memory is what the agent thinks *with*; long-term memory is what it
chooses to remember. Both are plain files — no DB — so the core stays
independent and offline. Recall is hybrid: if a local embedder is available
(Ollama + nomic-embed-text), notes are ranked by cosine similarity blended
with keyword overlap; otherwise it degrades to pure keyword recall. Vectors
live in a sidecar cache (.vectors.json) and are recomputed only when a note's
content changes.

    mem = Anamnesis("memory")
    mem.scratch_append("Goal: find the largest .py file and summarise it.")
    mem.working()                     # -> the live scratchpad
    mem.write("ollama-model", "Local model is qwen2.5:7b on port 11434")
    mem.recall("which local model")   # -> the note body
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .embeddings import Embedder, VectorCache, content_hash, cosine


_SLUG = re.compile(r"[^a-z0-9]+")
_WORD = re.compile(r"[a-z0-9]{3,}")
_WORKING_MAX_LINES = 200          # trim oldest entries past this


def _slug(s: str) -> str:
    return _SLUG.sub("-", s.lower()).strip("-")[:64] or "note"


class Anamnesis:
    def __init__(self, directory: str, embedder: Embedder | None = None) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = self.dir / "INDEX.md"
        if not self.index.exists():
            self.index.write_text("# Anamnesis — memory index\n\n", encoding="utf-8")
        self.work_path = self.dir / "WORKING.md"
        self.embedder = embedder
        self.vectors = (VectorCache(self.dir / ".vectors.json", embedder.model)
                        if embedder else None)

    @classmethod
    def from_config(cls, cfg) -> "Anamnesis":
        """Build with semantic recall wired to the configured Ollama embedder."""
        emb = (Embedder(cfg.ollama_host, cfg.embed_model)
               if getattr(cfg, "embed_enabled", True) else None)
        return cls(cfg.memory_dir, embedder=emb)

    # ===================================================================
    # WORKING MEMORY — the agent's live scratchpad
    # ===================================================================
    def working(self) -> str:
        """Return the current working-memory scratchpad (may be empty)."""
        if not self.work_path.exists():
            return ""
        return self.work_path.read_text(encoding="utf-8").strip()

    def scratch_append(self, text: str) -> None:
        """Append a timestamped line to working memory (trimmed if too long)."""
        text = (text or "").strip()
        if not text:
            return
        stamp = datetime.now().strftime("%m-%d %H:%M")
        existing = self.work_path.read_text(encoding="utf-8").splitlines() \
            if self.work_path.exists() else []
        existing.append(f"- [{stamp}] {text}")
        if len(existing) > _WORKING_MAX_LINES:           # keep the most recent
            existing = existing[-_WORKING_MAX_LINES:]
        self.work_path.write_text("\n".join(existing) + "\n", encoding="utf-8")

    def scratch_set(self, text: str) -> None:
        """Replace working memory wholesale (e.g. a fresh plan)."""
        self.work_path.write_text((text or "").strip() + "\n", encoding="utf-8")

    def scratch_clear(self) -> None:
        """Wipe working memory (task done / new context)."""
        if self.work_path.exists():
            self.work_path.unlink()

    def scratch_set_section(self, section: str, content: str) -> None:
        """Set a named section (## Section) in working memory.

        Creates the section if absent; replaces it if it already exists.
        Sections let the agent maintain Goal / Plan / Findings / Open as
        independent blocks instead of a single chronological log.
        """
        text = self.work_path.read_text(encoding="utf-8") if self.work_path.exists() else ""
        lines = text.splitlines()
        header = f"## {section}"

        # Find the section's extent
        start_idx = None
        end_idx = len(lines)
        for i, line in enumerate(lines):
            if line.rstrip() == header:
                start_idx = i
            elif start_idx is not None and line.startswith("## "):
                end_idx = i
                break

        new_block = [header] + (content.strip().splitlines() if content.strip() else [""])

        if start_idx is not None:
            new_lines = lines[:start_idx] + new_block + lines[end_idx:]
        else:
            # Append a blank separator then the new section
            sep = [] if not lines or not lines[-1].strip() else [""]
            new_lines = lines + sep + new_block

        if len(new_lines) > _WORKING_MAX_LINES:
            new_lines = new_lines[-_WORKING_MAX_LINES:]
        self.work_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    def scratch_append_section(self, section: str, text: str) -> None:
        """Append a bullet to a named section, creating the section if needed."""
        raw = self.work_path.read_text(encoding="utf-8") if self.work_path.exists() else ""
        lines = raw.splitlines()
        header = f"## {section}"
        bullet = f"- {text.strip()}"

        start_idx = None
        insert_idx = None
        for i, line in enumerate(lines):
            if line.rstrip() == header:
                start_idx = i
            elif start_idx is not None and line.startswith("## "):
                insert_idx = i
                break

        if start_idx is None:
            sep = [] if not lines or not lines[-1].strip() else [""]
            lines = lines + sep + [header, bullet]
        elif insert_idx is not None:
            lines.insert(insert_idx, bullet)
        else:
            lines.append(bullet)

        if len(lines) > _WORKING_MAX_LINES:
            lines = lines[-_WORKING_MAX_LINES:]
        self.work_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ===================================================================
    # CHECKPOINTS — freeze/restore working memory across restarts
    # ===================================================================
    def checkpoint_save(self, label: str, extra: str = "") -> str:
        """Snapshot current working memory under a label. Returns the slug."""
        slug = _slug(label)
        cdir = self.dir / "checkpoints"
        cdir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = self.working()
        if extra.strip():
            body += ("\n\n## Checkpoint notes\n" + extra.strip())
        (cdir / f"{slug}.md").write_text(
            f"---\nlabel: {slug}\nsaved: {stamp}\n---\n\n{body}\n",
            encoding="utf-8")
        return slug

    def checkpoint_load(self, label: str) -> str | None:
        """Replace working memory with a checkpoint's content. None if absent."""
        path = self.dir / "checkpoints" / f"{_slug(label)}.md"
        if not path.exists():
            return None
        body = path.read_text(encoding="utf-8").split("---\n\n", 1)[-1].strip()
        self.scratch_set(body)
        return body

    def checkpoint_list(self) -> list[tuple[str, str]]:
        """[(label, saved-timestamp)] for all checkpoints, newest first."""
        cdir = self.dir / "checkpoints"
        if not cdir.exists():
            return []
        out = []
        for p in cdir.glob("*.md"):
            saved = ""
            for line in p.read_text(encoding="utf-8").splitlines()[:4]:
                if line.startswith("saved:"):
                    saved = line.split(":", 1)[1].strip()
                    break
            out.append((p.stem, saved))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    # ===================================================================
    # LONG-TERM MEMORY — durable facts
    # ===================================================================
    # -- write ----------------------------------------------------------
    def write(self, name: str, body: str, *, kind: str = "note",
              description: str | None = None) -> str:
        """Create or overwrite a memory note. Returns its slug."""
        slug = _slug(name)
        desc = (description or body.splitlines()[0] if body else name)[:160]
        path = self.dir / f"{slug}.md"
        path.write_text(
            f"---\nname: {slug}\nkind: {kind}\ndescription: {desc}\n---\n\n{body}\n",
            encoding="utf-8",
        )
        self._reindex(slug, desc)
        self._embed_note(slug)
        return slug

    def forget(self, name: str) -> bool:
        """Delete a note (and its vector + index entry). True if it existed."""
        slug = _slug(name)
        path = self.dir / f"{slug}.md"
        existed = path.exists()
        if existed:
            path.unlink()
        if self.vectors:
            self.vectors.drop(slug)
            self.vectors.flush()
        lines = self.index.read_text(encoding="utf-8").splitlines()
        kept = [ln for ln in lines if not ln.startswith(f"- [{slug}]")]
        if len(kept) != len(lines):
            self.index.write_text("\n".join(kept) + "\n", encoding="utf-8")
        return existed

    # -- recall ---------------------------------------------------------
    def recall(self, query: str, k: int = 3) -> str:
        """Return the bodies of the top-k notes most relevant to `query`.

        Hybrid ranking: cosine similarity over local embeddings (when an
        embedder is available) blended with keyword overlap. Falls back to
        pure keyword recall if embeddings are unavailable.
        """
        notes = self._notes()
        if not notes:
            return ""
        kw = self._keyword_scores(query, notes)
        sem = self._semantic_scores(query, notes)

        if sem is not None:
            kmax = max(kw.values()) if kw and max(kw.values()) > 0 else 1
            scored = [
                (0.75 * sem.get(slug, 0.0) + 0.25 * (kw.get(slug, 0) / kmax), slug)
                for slug in notes
            ]
            scored = [(s, slug) for s, slug in scored if s > 0.35]
        else:
            scored = [(float(score), slug) for slug, score in kw.items() if score]

        scored.sort(reverse=True)
        chunks = []
        for _, slug in scored[:k]:
            body = notes[slug].split("---\n\n", 1)[-1].strip()
            chunks.append(f"- [{slug}] {body}")
        return "\n".join(chunks)

    def list(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.md")
                      if p.name not in ("INDEX.md", "WORKING.md"))

    # -- recall internals -------------------------------------------------
    def _notes(self) -> dict[str, str]:
        """slug -> full note text (excluding INDEX/WORKING)."""
        out: dict[str, str] = {}
        for path in self.dir.glob("*.md"):
            if path.name in ("INDEX.md", "WORKING.md"):
                continue
            out[path.stem] = path.read_text(encoding="utf-8")
        return out

    @staticmethod
    def _keyword_scores(query: str, notes: dict[str, str]) -> dict[str, int]:
        qwords = set(_WORD.findall(query.lower()))
        if not qwords:
            return {}
        return {slug: sum(text.lower().count(w) for w in qwords)
                for slug, text in notes.items()}

    def _semantic_scores(self, query: str,
                         notes: dict[str, str]) -> dict[str, float] | None:
        """slug -> cosine(query, note), or None if embeddings unavailable."""
        if not self.embedder or not self.embedder.alive or not self.vectors:
            return None
        # backfill vectors for notes that are new/changed since last embed
        missing = []
        hashes = {slug: content_hash(text) for slug, text in notes.items()}
        for slug, h in hashes.items():
            if self.vectors.get(slug, h) is None:
                missing.append(slug)
        if missing:
            vecs = self.embedder.embed([notes[s] for s in missing])
            if vecs is None:
                return None                       # embedder just died
            for slug, vec in zip(missing, vecs):
                self.vectors.put(slug, hashes[slug], vec)
            self.vectors.flush()
        qvec = self.embedder.embed([query])
        if not qvec:
            return None
        return {slug: cosine(qvec[0], self.vectors.get(slug, hashes[slug]) or [])
                for slug in notes}

    def _embed_note(self, slug: str) -> None:
        """Best-effort: embed a freshly written note into the vector cache."""
        if not self.embedder or not self.embedder.alive or not self.vectors:
            return
        path = self.dir / f"{slug}.md"
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
        vecs = self.embedder.embed([text])
        if vecs:
            self.vectors.put(slug, content_hash(text), vecs[0])
            self.vectors.flush()

    # -- index ----------------------------------------------------------
    def _reindex(self, slug: str, desc: str) -> None:
        lines = self.index.read_text(encoding="utf-8").splitlines()
        head = [ln for ln in lines if not ln.startswith(f"- [{slug}]")]
        head.append(f"- [{slug}]({slug}.md) — {desc}")
        self.index.write_text("\n".join(head) + "\n", encoding="utf-8")
