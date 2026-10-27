"""Visitor log: real IPs behind the Cloudflare Tunnel, and what gets stored."""

import json

import pytest

from cajetan import access_log


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    path = tmp_path / "access_log.sqlite"
    monkeypatch.setattr(access_log, "PATH", path)
    monkeypatch.setattr(access_log.config, "DATA_DIR", tmp_path)
    return path


def test_tunnel_ip_comes_from_cloudflare_header():
    assert access_log.client_ip("127.0.0.1", {"CF-Connecting-IP": "203.0.113.9"}) == ("203.0.113.9", True)
    assert access_log.client_ip("127.0.0.1", {"X-Forwarded-For": "198.51.100.4, 10.0.0.1"}) == ("198.51.100.4", True)
    assert access_log.client_ip("127.0.0.1", {}) == ("127.0.0.1", False)


def test_forged_header_from_non_local_address_is_ignored():
    assert access_log.client_ip("192.168.1.50", {"CF-Connecting-IP": "1.2.3.4"}) == ("192.168.1.50", False)


def test_records_page_loads_and_legs(log_path):
    access_log.record("203.0.113.9", True, "page", country="US", status=200)
    access_log.record("203.0.113.9", True, "rate", country="US", legs=["BUF -3.5", "James Cook o 79.5 rush yds"],
                      odds="+260", week="", status=200)
    rows = access_log.recent()
    assert [r["event"] for r in rows] == ["rate", "page"]
    assert json.loads(rows[0]["legs"]) == ["BUF -3.5", "James Cook o 79.5 rush yds"]
    assert rows[0]["odds"] == "+260" and rows[0]["week"] is None
    (v,) = access_log.visitors()
    assert (v["ip"], v["page_loads"], v["ratings"], v["country"]) == ("203.0.113.9", 1, 1, "US")
