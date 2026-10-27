"""Local web UI: serves `static/index.html` and rates parlays posted to /api/rate."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cajetan import access_log, engine, report
from cajetan.db import DataMissingError, connect

PAGE = Path(__file__).parent / "static" / "index.html"
_lock = threading.Lock()  # one DuckDB connection, one rating at a time


def rate_payload(body: dict) -> dict:
    raws = [ln.strip() for ln in str(body.get("legs", "")).splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    if not raws:
        raise ValueError('Enter at least one leg, e.g. "Josh Allen o 34.5 rush yds".')
    week = int(body["week"]) if body.get("week") not in (None, "") else None
    odds = int(str(body["odds"]).replace("+", "")) if body.get("odds") not in (None, "") else None
    n_reasons = int(body.get("reasons") or 5)

    from cajetan.cli import _data_note

    with _lock:
        con = connect(read_only=True)
        try:
            legs, parlay = engine.rate(con, raws, week, odds, log_source="web")
            note = _data_note(con)
        finally:
            con.close()

    out = json.loads(report.to_json(legs, parlay))
    for d, leg in zip(out["legs"], legs):
        d["why"] = None if leg.error else report._why(leg)
        d["reasons"] = [] if leg.error else [{"text": t, "sign": s} for t, s in leg.reasons(n_reasons)]
    rated = sum(1 for l in legs if not l.error)
    out["parlay"]["verdict"] = report._parlay_verdict(parlay, rated) if rated else None
    out["parlay"]["rated_legs"] = rated
    out["data_note"] = note
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _visitor(self) -> dict:
        ip, tunnel = access_log.client_ip(self.client_address[0], self.headers)
        return {"ip": ip, "via_tunnel": tunnel, "country": self.headers.get("CF-IPCountry")}

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            v = self._visitor()
            access_log.record(v["ip"], v["via_tunnel"], "page", country=v["country"], status=200)
        else:
            self._send(404, b"Not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/api/rate":
            self._send(404, b"Not found", "text/plain")
            return
        body: dict = {}
        status = 200
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            result = rate_payload(body)
            self._send(200, json.dumps(result).encode(), "application/json")
        except (ValueError, DataMissingError) as e:
            status = 400
            self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
        except Exception as e:
            status = 500
            self._send(500, json.dumps({"error": str(e) or type(e).__name__}).encode(), "application/json")
        v = self._visitor()
        legs = [ln.strip() for ln in str(body.get("legs", "")).splitlines() if ln.strip()] if isinstance(body, dict) else []
        access_log.record(v["ip"], v["via_tunnel"], "rate", country=v["country"], legs=legs,
                          odds=str(body.get("odds") or "") if isinstance(body, dict) else None,
                          week=str(body.get("week") or "") if isinstance(body, dict) else None, status=status)

    def log_message(self, fmt: str, *args) -> None:
        pass


def serve(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)
