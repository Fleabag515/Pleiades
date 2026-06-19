"""VaultTool._normalize: every natural way of naming a site must resolve to the
same stored secret, regardless of a 'www.' prefix, a pre-existing 'site:'
prefix, or case. Regression coverage for a bug where a credential stored as
'www.instagram.com' was unreachable via 'instagram.com'."""

from cryptography.fernet import Fernet

from pleiades.tools.vault_tool import VaultTool
from pleiades.vault import Vault


def test_normalize_reserved_key_untouched():
    assert VaultTool._normalize("email.password") == "email.password"
    assert VaultTool._normalize("discord.token") == "discord.token"


def test_normalize_bare_domain():
    assert VaultTool._normalize("instagram.com") == "site:instagram.com"


def test_normalize_strips_www():
    assert VaultTool._normalize("www.instagram.com") == "site:instagram.com"


def test_normalize_already_site_prefixed_still_canonicalized():
    assert VaultTool._normalize("site:www.Instagram.COM") == "site:instagram.com"
    assert VaultTool._normalize("site:instagram.com") == "site:instagram.com"


def test_normalize_no_dot_passthrough():
    assert VaultTool._normalize("Instagram") == "Instagram"


def test_store_then_get_any_form_finds_it(tmp_path):
    vault = Vault(tmp_path / "vault.db", key=Fernet.generate_key())
    ctx = type("Ctx", (), {"vault": vault})()
    tool = VaultTool()

    tool.run(ctx, action="store", name="www.instagram.com", secret="hunter2")

    for name in ("www.instagram.com", "instagram.com", "Instagram.com",
                 "site:instagram.com", "site:WWW.Instagram.com"):
        result = tool.run(ctx, action="get", name=name)
        assert "No secret stored" not in result, f"lookup failed for {name!r}"
