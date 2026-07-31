"""http_fetch/deep_research are safe=True (no approval) and take a
model-supplied URL reachable via prompt injection. _get_external must
refuse to fetch loopback/private/link-local/reserved addresses -- otherwise
it reaches Anamnesis's unauthenticated :9000 control API or the webui's
vault_reveal endpoint. See HANDOFF's 2026-07-25 correctness audit, finding #1."""

import pytest

import pleiades.harness.builtins.web as web
from pleiades.harness.builtins.web import _SSRFBlocked, _assert_public_ip, _resolve_pinned


@pytest.mark.parametrize("ip", [
    "127.0.0.1",        # loopback -- Anamnesis control API, webui
    "::1",               # loopback, IPv6
    "169.254.169.254",  # cloud metadata service
    "10.0.0.5",          # private
    "172.16.0.1",        # private
    "192.168.1.1",       # private
    "0.0.0.0",            # unspecified
])
def test_rejects_non_public_ips(ip):
    with pytest.raises(_SSRFBlocked):
        _assert_public_ip(ip, host="example.invalid")


@pytest.mark.parametrize("ip", ["93.184.216.34", "8.8.8.8", "1.1.1.1"])
def test_allows_public_ips(ip):
    _assert_public_ip(ip, host="example.invalid")  # must not raise


def test_resolve_pinned_rejects_localhost_hostname():
    with pytest.raises(_SSRFBlocked):
        _resolve_pinned("localhost")


# --------------------------------------------------------------------------- #
# _get_external -- the actual guarded fetch path http_fetch/deep_research call.
# Everything above only exercises the two low-level helpers; these tests cover
# the behavior the module's docstring says is the whole point: redirects are
# re-validated hop by hop, redirect count is capped, and the connection uses
# the pre-resolved pinned IP rather than the raw host. A fake connection class
# stands in for _PinnedHTTP(S)Connection so no real socket is opened.
# --------------------------------------------------------------------------- #
class _FakeHeaders:
    def __init__(self, location=None):
        self._location = location

    def get(self, name, default=None):
        return self._location if name == "Location" else default


class _FakeResp:
    def __init__(self, status, location=None, body=b""):
        self.status = status
        self.headers = _FakeHeaders(location)
        self._body = body

    def read(self, *a, **kw):
        return self._body


def test_get_external_revalidates_host_on_every_redirect_hop(monkeypatch):
    """A 302 from a validated public host to an internal host must be
    rejected at the redirect hop, not just at the original URL -- otherwise
    a URL that passes validation could bounce straight to an internal
    address (the DNS-rebind-adjacent gap this module exists to close)."""
    def fake_resolve(host):
        if host == "internal.example":
            raise _SSRFBlocked(f"refusing internal host {host}")
        return "93.184.216.34"
    monkeypatch.setattr(web, "_resolve_pinned", fake_resolve)

    class _FakeConn:
        def __init__(self, host, pinned_ip, port=None, timeout=None):
            self.host = host

        def request(self, method, path, headers=None):
            pass

        def getresponse(self):
            return _FakeResp(302, location="http://internal.example/secret")

        def close(self):
            pass

    monkeypatch.setattr(web, "_PinnedHTTPConnection", _FakeConn)
    monkeypatch.setattr(web, "_PinnedHTTPSConnection", _FakeConn)

    with pytest.raises(_SSRFBlocked):
        web._get_external("http://public.example/start")


def test_get_external_stops_after_max_redirects(monkeypatch):
    monkeypatch.setattr(web, "_resolve_pinned", lambda host: "93.184.216.34")

    class _FakeConn:
        def __init__(self, host, pinned_ip, port=None, timeout=None):
            pass

        def request(self, method, path, headers=None):
            pass

        def getresponse(self):
            return _FakeResp(302, location="http://public.example/next")

        def close(self):
            pass

    monkeypatch.setattr(web, "_PinnedHTTPConnection", _FakeConn)
    monkeypatch.setattr(web, "_PinnedHTTPSConnection", _FakeConn)

    with pytest.raises(_SSRFBlocked, match="too many redirects"):
        web._get_external("http://public.example/start")


def test_get_external_connects_to_pinned_ip_and_returns_body(monkeypatch):
    """The connection must be opened against the pre-resolved, pre-validated
    IP -- not left to re-resolve `host` itself, which would reopen the
    check-then-connect DNS-rebind gap."""
    seen = {}
    monkeypatch.setattr(web, "_resolve_pinned", lambda host: "93.184.216.34")

    class _FakeConn:
        def __init__(self, host, pinned_ip, port=None, timeout=None):
            seen["host"] = host
            seen["pinned_ip"] = pinned_ip

        def request(self, method, path, headers=None):
            seen["host_header"] = (headers or {}).get("Host")

        def getresponse(self):
            return _FakeResp(200, body=b"hello world")

        def close(self):
            pass

    monkeypatch.setattr(web, "_PinnedHTTPConnection", _FakeConn)
    monkeypatch.setattr(web, "_PinnedHTTPSConnection", _FakeConn)

    body = web._get_external("http://public.example/page")
    assert body == "hello world"
    assert seen["pinned_ip"] == "93.184.216.34"
    assert seen["host"] == "public.example"
    assert seen["host_header"] == "public.example"


if __name__ == "__main__":
    for ip in ["127.0.0.1", "::1", "169.254.169.254", "10.0.0.5", "0.0.0.0"]:
        try:
            _assert_public_ip(ip, host="x")
            raise AssertionError(f"{ip} should have been rejected")
        except _SSRFBlocked:
            pass
    _assert_public_ip("8.8.8.8", host="x")
    print("ok")
