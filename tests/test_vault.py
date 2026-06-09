from cryptography.fernet import Fernet

from pleiades.vault import Vault, _key_from_passphrase


def test_set_get_roundtrip(tmp_path):
    v = Vault(tmp_path / "vault.db", key=Fernet.generate_key())
    v.set("site:example.com", "hunter2", meta={"note": "me@example.com"})
    assert v.get("site:example.com") == "hunter2"


def test_list_never_returns_values(tmp_path):
    v = Vault(tmp_path / "vault.db", key=Fernet.generate_key())
    v.set("email.password", "secret")
    v.set("site:foo.com", "pw")
    entries = {e["key"]: e for e in v.list()}
    assert "email.password" in entries
    assert entries["email.password"]["reserved"] is True
    assert entries["site:foo.com"]["site"] == "foo.com"
    # Values must not appear anywhere in the listing.
    assert all("secret" not in str(e) and "pw" != e.get("value", "") for e in v.list())


def test_delete(tmp_path):
    v = Vault(tmp_path / "vault.db", key=Fernet.generate_key())
    v.set("k", "x")
    assert v.delete("k") is True
    assert v.get("k") is None
    assert v.delete("k") is False


def test_encrypted_at_rest(tmp_path):
    key = Fernet.generate_key()
    path = tmp_path / "vault.db"
    Vault(path, key=key).set("k", "plaintext-should-not-appear")
    assert b"plaintext-should-not-appear" not in path.read_bytes()


def test_passphrase_key_is_stable():
    assert _key_from_passphrase("hello") == _key_from_passphrase("hello")
    assert _key_from_passphrase("hello") != _key_from_passphrase("world")


def test_site_key():
    assert Vault.site_key("Example.COM") == "site:example.com"
