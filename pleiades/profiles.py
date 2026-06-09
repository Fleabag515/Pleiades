"""Profiles — the unifying layer.

A Pleiades profile *is* an Anamnesis character. Creating a profile creates: the
Anamnesis character, the profile dir, the encrypted vault, and the browser dir.
Non-secret config lives in profile.json; secrets (email password, discord token,
site credentials) live in the profile's vault. This is where "set up once per
character, never per tool" is enforced.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from . import config
from .anamnesis import Anamnesis
from .vault import Vault


@dataclass
class Profile:
    name: str
    email_address: str = ""
    imap_host: str = ""
    imap_port: int = 993
    smtp_host: str = ""
    smtp_port: int = 587
    discord_enabled: bool = False
    persona_source: str = "auto"  # auto | file | inline | disabled

    @property
    def browser_dir(self) -> str:
        return str(config.browser_dir(self.name))

    @property
    def has_email(self) -> bool:
        return bool(self.email_address and self.imap_host and self.smtp_host)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Profile":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})


class ProfileManager:
    """Create and manage profiles; keeps Anamnesis characters and Pleiades state in sync."""

    def __init__(self, settings: Optional[config.Settings] = None, anamnesis: Optional[Anamnesis] = None):
        self.settings = settings or config.Settings.load()
        self.anamnesis = anamnesis or Anamnesis(self.settings.anamnesis_control_url)

    # -- persistence -------------------------------------------------------- #
    def _save(self, profile: Profile) -> None:
        config.ensure_home()
        config.profile_dir(profile.name).mkdir(parents=True, exist_ok=True)
        config.profile_json_path(profile.name).write_text(
            json.dumps(profile.to_json(), indent=2), encoding="utf-8"
        )

    def _load(self, name: str) -> Profile:
        path = config.profile_json_path(name)
        if not path.is_file():
            raise FileNotFoundError(f"No Pleiades profile '{name}' (expected {path}).")
        return Profile.from_json(json.loads(path.read_text(encoding="utf-8")))

    # -- CRUD --------------------------------------------------------------- #
    def create(
        self,
        name: str,
        *,
        email_address: str = "",
        imap_host: str = "",
        imap_port: int = 993,
        smtp_host: str = "",
        smtp_port: int = 587,
        email_password: Optional[str] = None,
        discord_token: Optional[str] = None,
        persona_source: str = "auto",
        anamnesis_cfg: Optional[dict] = None,
    ) -> Profile:
        """Create the Anamnesis character + Pleiades profile + vault + dirs."""
        if config.profile_json_path(name).is_file():
            raise ValueError(f"Profile '{name}' already exists.")

        # 1. Anamnesis character. upstream.baseUrl points at OUR inference server.
        cfg = {
            "upstream": {"baseUrl": self.settings.upstream_base_url},
        }
        if self.settings.backend_api_key:
            cfg["upstream"]["apiKey"] = self.settings.backend_api_key
        if anamnesis_cfg:
            cfg.update(anamnesis_cfg)
        if not self.anamnesis.exists(name):
            self.anamnesis.create_character(name, **cfg)

        # 2. Profile record + dirs.
        profile = Profile(
            name=name,
            email_address=email_address,
            imap_host=imap_host,
            imap_port=imap_port,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            discord_enabled=bool(discord_token),
            persona_source=persona_source,
        )
        self._save(profile)
        config.browser_dir(name).mkdir(parents=True, exist_ok=True)

        # 3. Vault + reserved secrets.
        with self.open_vault(name) as vault:
            if email_address:
                vault.set("email.address", email_address, meta={"reserved": True})
            if email_password:
                vault.set("email.password", email_password, meta={"reserved": True})
            if discord_token:
                vault.set("discord.token", discord_token, meta={"reserved": True})

        return profile

    def get(self, name: str) -> Profile:
        return self._load(name)

    def list(self) -> list[Profile]:
        if not config.PROFILES_DIR.is_dir():
            return []
        out = []
        for child in sorted(config.PROFILES_DIR.iterdir()):
            if (child / "profile.json").is_file():
                try:
                    out.append(self._load(child.name))
                except (json.JSONDecodeError, OSError):
                    continue
        return out

    def delete(self, name: str, *, delete_anamnesis: bool = True) -> None:
        import shutil

        if delete_anamnesis and self.anamnesis.exists(name):
            self.anamnesis.delete(name)
        pdir = config.profile_dir(name)
        if pdir.is_dir():
            shutil.rmtree(pdir)

    def open_vault(self, name: str) -> Vault:
        return Vault(config.vault_path(name))
