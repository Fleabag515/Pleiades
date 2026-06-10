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


def test_bind_scopes_memory_and_workspace_to_character():
    from pleiades.harness import Config
    from pleiades import config as repo_config
    name = _make_profile("carol_t")
    cfg = Config.load()
    identity.bind_character(name, cfg=cfg, route_inference=False)
    pdir = str(repo_config.profile_dir(name))
    assert cfg.workspace_root.startswith(pdir)
    assert cfg.memory_dir.startswith(pdir)


def test_bind_character_shares_browser_profile():
    import pleiades.harness.builtins.browser as br
    from pleiades import config as repo_config
    name = _make_profile("dora_t")
    identity.bind_character(name, cfg=None, route_inference=False)
    assert br._BOUND_PROFILE["dir"] == str(repo_config.browser_dir(name))


def test_searchtool_delegates_to_harness(monkeypatch):
    from pleiades.tools.search import SearchTool
    from pleiades.tools import ToolContext
    import pleiades.harness.builtins.web as web

    seen = {}
    monkeypatch.setattr(web, "web_search", lambda query, count=6: f"WS:{query}:{count}")
    monkeypatch.setattr(web, "bind_searxng", lambda url: seen.__setitem__("url", url))

    class _S:
        searxng_url = "http://searx.local:8888"

    ctx = ToolContext(profile=None, vault=None, settings=_S())  # type: ignore[arg-type]
    out = SearchTool().run(ctx, query="hello", num_results=3)
    assert out == "WS:hello:3"
    assert seen["url"] == "http://searx.local:8888"
