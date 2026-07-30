from pleiades.anamnesis import Anamnesis


def test_is_running_true_when_active_flag_set(monkeypatch):
    a = Anamnesis()
    monkeypatch.setattr(a, "get_character", lambda name: {"name": name, "port": 8086, "active": True, "running": True})
    assert a.is_running("Claude") is True


def test_is_running_false_when_active_flag_false(monkeypatch):
    # The real regression: a registered-but-not-started (or crashed) character's
    # /characters/:name response has active=False, running=False -- but DOES still
    # carry a `port` (assigned at registration, independent of whether the proxy
    # process is actually alive). Before this fix, is_running() only checked for a
    # status/state STRING field (a shape this daemon never sends), found neither,
    # fell through to "does this character have a port number" (true forever, once
    # created), and reported the character running when it genuinely was not --
    # so ensure_running() never called start() and every request just failed at the
    # client with a bare connection error, with nothing to self-heal it.
    a = Anamnesis()
    monkeypatch.setattr(a, "get_character", lambda name: {"name": name, "port": 8085, "active": False, "running": False})
    assert a.is_running("Mark") is False


def test_is_running_status_string_fallback(monkeypatch):
    # A hypothetical daemon shape using a status/state string instead of booleans --
    # kept working, not just replaced outright.
    a = Anamnesis()
    monkeypatch.setattr(a, "get_character", lambda name: {"name": name, "port": 8084, "status": "running"})
    assert a.is_running("legacy") is True
    monkeypatch.setattr(a, "get_character", lambda name: {"name": name, "port": 8084, "status": "stopped"})
    assert a.is_running("legacy") is False


def test_extract_port_flat():
    assert Anamnesis._extract_port({"port": 8084}) == 8084


def test_extract_port_nested_proxy():
    assert Anamnesis._extract_port({"proxy": {"port": 8085}}) == 8085


def test_extract_port_in_config():
    assert Anamnesis._extract_port({"config": {"proxy": {"port": 8086}}}) == 8086


def test_extract_port_missing():
    assert Anamnesis._extract_port({"name": "x"}) is None
