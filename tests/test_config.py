from pleiades import config


def test_inference_and_upstream_urls():
    s = config.Settings(inference_host="127.0.0.1", inference_port=8080)
    assert s.inference_base_url == "http://127.0.0.1:8080/v1"
    # Default upstream is our own inference server.
    assert s.upstream_base_url == s.inference_base_url


def test_backend_override_wins():
    s = config.Settings(backend_base_url="http://example/v1")
    assert s.upstream_base_url == "http://example/v1"


def test_email_presets_have_gmail_and_mailcom():
    assert "gmail" in config.EMAIL_PRESETS
    assert "mail.com" in config.EMAIL_PRESETS
    assert config.EMAIL_PRESETS["gmail"]["imap_host"] == "imap.gmail.com"


def test_path_helpers():
    assert config.vault_path("alice").name == "vault.db"
    assert config.profile_json_path("alice").name == "profile.json"
    assert config.browser_dir("alice").name == "browser"


def test_validate_name_accepts_normal_names():
    for ok in ("alice", "Maia 2", "zoe-prime", "café", "a.b"):
        assert config.validate_name(ok) == ok


def test_validate_name_rejects_traversal_and_separators():
    import pytest

    for bad in ("", ".", "..", "../x", "a/b", "a\\b", "..secret", " lead",
                "trail ", "x" * 65, "nul\x00", "con<>"):
        with pytest.raises(ValueError):
            config.validate_name(bad)


def test_profile_dir_rejects_traversal():
    import pytest

    with pytest.raises(ValueError):
        config.profile_dir("../../etc")
