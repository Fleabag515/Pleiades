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


# ── "did you mean" hint on a miss ────────────────────────────────────────────
# Real-world reproduction: Mark's actual transcript guessed 'instagram:steph_berry'
# and bare 'instagram' for a secret stored as 'site:instagram.com'. Neither
# contains a '.', so neither reaches site_key() canonicalization even with the
# www/case fix above -- they need a fuzzy hint instead of a flat miss.

def test_did_you_mean_suggests_close_key_with_colon_qualifier(tmp_path):
    vault = Vault(tmp_path / "vault.db", key=Fernet.generate_key())
    ctx = type("Ctx", (), {"vault": vault})()
    tool = VaultTool()
    tool.run(ctx, action="store", name="instagram.com", secret="hunter2")

    result = tool.run(ctx, action="get", name="instagram:steph_berry")
    assert "No secret stored" in result
    assert "Did you mean" in result
    assert "site:instagram.com" in result


def test_did_you_mean_suggests_close_key_bare_word(tmp_path):
    vault = Vault(tmp_path / "vault.db", key=Fernet.generate_key())
    ctx = type("Ctx", (), {"vault": vault})()
    tool = VaultTool()
    tool.run(ctx, action="store", name="instagram.com", secret="hunter2")

    result = tool.run(ctx, action="get", name="instagram")
    assert "Did you mean: site:instagram.com?" in result


def test_did_you_mean_silent_when_nothing_close(tmp_path):
    vault = Vault(tmp_path / "vault.db", key=Fernet.generate_key())
    ctx = type("Ctx", (), {"vault": vault})()
    tool = VaultTool()
    tool.run(ctx, action="store", name="instagram.com", secret="hunter2")

    result = tool.run(ctx, action="get", name="totallyunrelated")
    assert "Did you mean" not in result


def test_did_you_mean_stoplist_avoids_false_positive():
    # Two unrelated site keys that only share the generic 'com'/'site' tokens
    # must NOT be suggested for each other.
    assert not (VaultTool._tokens("site:github.com") & VaultTool._tokens("site:gitlab.com"))
