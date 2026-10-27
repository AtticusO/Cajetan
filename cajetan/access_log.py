"""Visitor log for the web UI: who connected (IP), when, and the legs they rated.

Stored in data/access_log.sqlite (SQLite, separate from the DuckDB database so the web
server never holds a write lock that `cajetan grade` or `cajetan update` would need).
Behind a Cloudflare Tunnel every request arrives from cloudflared on this machine, so the
visitor's address is taken from Cloudflare's CF-Connecting-IP header, but only when the
request really comes from localhost (otherwise the header could be forged).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

from cajetan import config

PATH = config.DATA_DIR / "access_log.sqlite"
_lock = threading.Lock()
LOCAL = ("127.0.0.1", "::1")


def client_ip(remote_addr: str, headers) -> tuple[str, bool]:
    """(visitor IP, came through the tunnel)."""
    if remote_addr in LOCAL:
        forwarded = headers.get("CF-Connecting-IP") or (headers.get("X-Forwarded-For") or "").split(",")[0].strip()
        if forwarded:
            return forwarded, True
    return remote_addr, False


def _connect() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(PATH, timeout=5)
    con.execute(
        """CREATE TABLE IF NOT EXISTS connections (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               time_utc TEXT NOT NULL,
               ip TEXT NOT NULL,
               via_tunnel INTEGER NOT NULL,
               country TEXT,
               event TEXT NOT NULL,      -- page (opened the page) or rate (submitted legs)
               legs TEXT,                -- JSON list of the legs entered
               odds TEXT,
               week TEXT,
               status INTEGER            -- HTTP status of the response
           )"""
    )
    con.execute("CREATE INDEX IF NOT EXISTS connections_ip ON connections (ip)")
    return con


def record(ip: str, via_tunnel: bool, event: str, *, country: str | None = None, legs: list[str] | None = None,
           odds: str | None = None, week: str | None = None, status: int | None = None) -> None:
    row = (datetime.now(timezone.utc).isoformat(timespec="seconds"), ip, int(via_tunnel), country, event,
           json.dumps(legs) if legs is not None else None, odds or None, week or None, status)
    try:
        with _lock:
            con = _connect()
            try:
                con.execute("INSERT INTO connections (time_utc, ip, via_tunnel, country, event, legs, odds, week, status) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
                con.commit()
            finally:
                con.close()
    except sqlite3.Error:
        pass  # logging must never break the page


def recent(limit: int = 50, ip: str | None = None) -> list[sqlite3.Row]:
    if not PATH.exists():
        return []
    con = _connect()
    con.row_factory = sqlite3.Row
    try:
        sql, args = "SELECT * FROM connections", []
        if ip:
            sql += " WHERE ip = ?"
            args.append(ip)
        return con.execute(sql + " ORDER BY id DESC LIMIT ?", [*args, limit]).fetchall()
    finally:
        con.close()


def visitors() -> list[sqlite3.Row]:
    """One row per IP: first and last seen, page loads and ratings."""
    if not PATH.exists():
        return []
    con = _connect()
    con.row_factory = sqlite3.Row
    try:
        return con.execute(
            """SELECT ip, max(country) AS country, min(time_utc) AS first_seen, max(time_utc) AS last_seen,
                      sum(event = 'page') AS page_loads, sum(event = 'rate') AS ratings
               FROM connections GROUP BY ip ORDER BY last_seen DESC"""
        ).fetchall()
    finally:
        con.close()
