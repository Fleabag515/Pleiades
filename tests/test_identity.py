"""Identity bridge: harness tools bound to a character (no daemon/network needed)."""

import json

import pytest

import pleiades.harness.builtins  # noqa: F401  register builtin tools
import pleiades.harness.identity as identity
from pleiades.harness import registry
from pleiades import config as repo_config
from pleiades.profiles import Profile


def _make_profile(name: str, **kw) -> str:
    repo_config.ensure_home()
    repo_config.profile_dir(name).mkdir(parents=True, exist_ok=True)
    repo_config.profile_json_path(name).write_text(
        json.dumps(Profile(name=name, **kw).to_json()), encoding="utf-8"
    )
    return name


def test_identity_tools_registered():
    names = {t.name for t in registry.all()}
    assert "vault" in names and "email" in names


def test_unbound_identity_tool_errors():
    identity._ACTIVE.update(profile=None, vault=None, settings=None)
    with pytest.raises(RuntimeError):
        registry.get("vault").func(action="list")


def test_bind_and_vault_roundtrip():
    name = _make_profile("alice_t")
    identity.bind_character(name, cfg=None, route_inference=False)
    vault_fn = registry.get("vault").func
    assert "Stored secret" in vault_fn(action="store", name="github.com",
                                       secret="pw123", note="me@x.com")
    assert "pw123" in vault_fn(action="get", name="github.com")
    listing = vault_fn(action="list")
    assert "github.com" in listing and "pw123" not in listing  # never leaks values


def test_email_without_config_is_graceful():
    name = _make_profile("bob_t")
    identity.bind_character(name, cfg=None, route_inference=False)
    out = registry.get("email").func(action="list_unread")
    assert "no email configured" in out.lower()
