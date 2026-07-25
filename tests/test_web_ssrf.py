"""http_fetch/deep_research are safe=True (no approval) and take a
model-supplied URL reachable via prompt injection. _get_external must
refuse to fetch loopback/private/link-local/reserved addresses -- otherwise
it reaches Anamnesis's unauthenticated :9000 control API or the webui's
vault_reveal endpoint. See HANDOFF's 2026-07-25 correctness audit, finding #1."""

import pytest

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


if __name__ == "__main__":
    for ip in ["127.0.0.1", "::1", "169.254.169.254", "10.0.0.5", "0.0.0.0"]:
        try:
            _assert_public_ip(ip, host="x")
            raise AssertionError(f"{ip} should have been rejected")
        except _SSRFBlocked:
            pass
    _assert_public_ip("8.8.8.8", host="x")
    print("ok")
