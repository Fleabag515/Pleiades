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
