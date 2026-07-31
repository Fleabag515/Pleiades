"""Encrypted per-profile secret store.

Each profile has its own SQLite vault (`vault.db`) whose values are Fernet-encrypted
at rest. The encryption key comes from PLEIADES_MASTER_KEY, or a generated keyfile at
~/.pleiades/master.key (mode 0600).

Guardrails (see CLAUDE.md §8):
  * Secrets are NEVER logged.
  * Secrets are NEVER auto-injected into prompts.
  * A secret reaches the model only when the model explicitly calls the vault tool.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import time
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from . import config

# Reserved keys: Pleiades-managed, not model-created site credentials.
RESERVED_KEYS = {"email.password", "email.address", "discord.token"}
SITE_PREFIX = "site:"


class VaultError(RuntimeError):
    pass


def generate_key() -> bytes:
    """A new urlsafe Fernet key (44 url-safe base64 bytes)."""
    return Fernet.generate_key()


def load_key() -> bytes:
    """Load the master key from env, else read/generate the keyfile (mode 0600)."""
    env = os.environ.get("PLEIADES_MASTER_KEY", "").strip()
    if env:
        key = env.encode()
        # Accept either a raw Fernet key or a passphrase we derive a key from.
        try:
            Fernet(key)
            return key
        except (ValueError, TypeError):
            return _key_from_passphrase(env)

    path = config.MASTER_KEY_PATH
    if path.is_file():
        return path.read_bytes().strip()

    config.ensure_home()
    key = generate_key()
    # O_CREAT|O_EXCL so two processes racing to bootstrap the first-ever key can't
    # clobber each other: the loser's open() fails with FileExistsError and it falls
    # back to reading the winner's key instead of overwriting it (see vault.py history
    # -- a plain write_bytes() here let the second writer silently stomp the first,
    # permanently orphaning whatever was already encrypted under the lost key).
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, stat.S_IRUSR | stat.S_IWUSR)
    except FileExistsError:
        return path.read_bytes().strip()
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(key)
    except OSError:
        pass
    return key


def _key_from_passphrase(passphrase: str) -> bytes:
    """Derive a stable Fernet key from an arbitrary passphrase (PBKDF2)."""
    import base64
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    # Static salt: a passphrase must derive the same key every run. The passphrase
    # itself is the secret; the salt only namespaces the derivation.
    salt = b"pleiades-vault-v1"
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480_000)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))


class Vault:
    """An encrypted key/value store backed by SQLite."""

    def __init__(self, path: Path | str, key: Optional[bytes] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(key or load_key())
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS secrets (
                key        TEXT PRIMARY KEY,
                value      BLOB NOT NULL,
                meta       TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Vault":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- operations --------------------------------------------------------- #
    def set(self, key: str, value: str, meta: Optional[dict] = None) -> None:
        token = self._fernet.encrypt(value.encode())
        now = time.time()
        meta_json = json.dumps(meta) if meta else None
        self._conn.execute(
            """
            INSERT INTO secrets (key, value, meta, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value, meta=excluded.meta, updated_at=excluded.updated_at
            """,
            (key, token, meta_json, now, now),
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM secrets WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        try:
            return self._fernet.decrypt(row[0]).decode()
        except InvalidToken as e:
            raise VaultError(
                f"Could not decrypt '{key}'. Wrong PLEIADES_MASTER_KEY / master.key?"
            ) from e

    def list(self) -> list[dict]:
        """Return metadata for all entries (keys + meta + timestamps) — never values."""
        rows = self._conn.execute(
            "SELECT key, meta, created_at, updated_at FROM secrets ORDER BY key"
        ).fetchall()
        out = []
        for key, meta, created, updated in rows:
            out.append(
                {
                    "key": key,
                    "meta": json.loads(meta) if meta else None,
                    "reserved": key in RESERVED_KEYS,
                    "site": key[len(SITE_PREFIX):] if key.startswith(SITE_PREFIX) else None,
                    "created_at": created,
                    "updated_at": updated,
                }
            )
        return out

    def delete(self, key: str) -> bool:
        cur = self._conn.execute("DELETE FROM secrets WHERE key=?", (key,))
        self._conn.commit()
        return cur.rowcount > 0

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def site_key(domain: str) -> str:
        """Canonicalize a domain into its vault key. Lowercased, and a leading
        'www.' is stripped, so 'instagram.com' and 'www.instagram.com' always
        resolve to the same secret regardless of which form was used to store it."""
        domain = domain.lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        return f"{SITE_PREFIX}{domain}"
