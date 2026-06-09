from pleiades.anamnesis import Anamnesis


def test_extract_port_flat():
    assert Anamnesis._extract_port({"port": 8084}) == 8084


def test_extract_port_nested_proxy():
    assert Anamnesis._extract_port({"proxy": {"port": 8085}}) == 8085


def test_extract_port_in_config():
    assert Anamnesis._extract_port({"config": {"proxy": {"port": 8086}}}) == 8086


def test_extract_port_missing():
    assert Anamnesis._extract_port({"name": "x"}) is None
