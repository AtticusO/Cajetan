"""Prediction log: every rating is appended to data/predictions.jsonl, and `cajetan grade`
settles the ones whose games are final and saves them in the `prediction_grades` table, so
accuracy can be measured (and tracked over time) at real sportsbook lines.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

from cajetan import config
from cajetan.analysis.explain import LegResult, ParlayResult
from cajetan.analysis.markets import MARKETS


def record(legs: list[LegResult], parlay: ParlayResult, source: str, book_odds: int | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pid = uuid.uuid4().hex[:12]
    lines = []
    for i, leg in enumerate(legs):
        if leg.error:
            continue
        lines.append(json.dumps({
            "leg_id": f"{pid}-{i}", "logged_at": now, "parlay_id": pid, "source": source,
            "input": getattr(leg.leg, "raw", ""),
            "leg": leg.leg.describe(), "market": leg.market, "side": leg.side, "line": leg.line,
            "odds": getattr(leg.leg, "odds", None), "player_id": leg.player_id, "team": leg.team,
            "opponent": leg.opponent, "position": leg.position, "game_id": leg.game_id, "season": leg.season,
            "week": leg.week, "probability": round(leg.probability, 5),
            "raw_probability": round(leg.raw_probability if leg.raw_probability is not None else leg.probability, 5),
            "projection": leg.projection, "reliability": leg.reliability,
            "parlay_probability": round(parlay.probability, 5), "parlay_odds": book_odds,
        }))
    if not lines:
        return
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.PREDICTION_LOG, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError:
        pass  # logging must never break a rating


def load() -> pd.DataFrame:
    if not config.PREDICTION_LOG.exists():
        return pd.DataFrame()
    rows = []
    for ln in config.PREDICTION_LOG.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(ln))
        except ValueError:
            continue
    return pd.DataFrame(rows)


def settle(market: str, side: str, line: float | None, actual: float | None) -> str | None:
    """won / lost / push for the bettor's side, None if unknown. For team markets `actual` is the subject
    team's margin (ml, spread) or points (team_total), or the combined points (game_total)."""
    if actual is None or actual != actual:
        return None
    if market == "ml":
        return "won" if actual > 0 else "lost" if actual < 0 else "push"
    if market == "spread":
        v = actual + line
        return "won" if v > 0 else "lost" if v < 0 else "push"
    if market == "anytime_td":
        return "won" if actual >= 1 else "lost"
    if actual == line:
        return "push"
    over = actual > line
    return "won" if over == (side == "over") else "lost"


def actual_value(con: duckdb.DuckDBPyConnection, market: str, game_id: str, player_id: str | None,
                 team: str | None) -> float | None:
    if market in MARKETS:
        row = con.execute(
            f"SELECT CAST({MARKETS[market].stat} AS DOUBLE) FROM (SELECT * REPLACE ("
            "COALESCE(passing_yards,0) AS passing_yards, COALESCE(rushing_yards,0) AS rushing_yards, "
            "COALESCE(receiving_yards,0) AS receiving_yards, COALESCE(rushing_tds,0) AS rushing_tds, "
            "COALESCE(receiving_tds,0) AS receiving_tds) FROM player_week) "
            "WHERE player_id = ? AND game_id = ?", [player_id, game_id]).fetchone()
        return None if row is None else row[0]  # no row: inactive, props are void
    row = con.execute("SELECT home_team, home_score, away_score FROM schedules WHERE game_id = ?", [game_id]).fetchone()
    if not row or row[1] is None:
        return None
    home, hs, as_ = row
    pts, opp = (hs, as_) if team == home else (as_, hs)
    return {"ml": pts - opp, "spread": pts - opp, "game_total": pts + opp, "team_total": pts}[market]


def grade(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Every logged leg with its actual result: won / lost / push, void (player inactive), or None (pending)."""
    df = load()
    if df.empty:
        return df
    if "leg_id" not in df.columns:
        df["leg_id"] = None
    # Entries logged before leg ids existed get a stable id from their parlay and leg text.
    df["leg_id"] = df["leg_id"].fillna(df["parlay_id"] + "-" + df["leg"])
    final = {gid for (gid,) in con.execute("SELECT game_id FROM schedules WHERE home_score IS NOT NULL").fetchall()}
    results, actuals = [], []
    for r in df.itertuples():
        a = actual_value(con, r.market, r.game_id, getattr(r, "player_id", None), r.team)
        actuals.append(a)
        res = settle(r.market, r.side, r.line, a)
        results.append("void" if res is None and r.game_id in final else res)
    return df.assign(actual=actuals, result=results)


GRADE_COLUMNS = {
    "leg_id": "VARCHAR PRIMARY KEY", "parlay_id": "VARCHAR", "logged_at": "VARCHAR", "graded_at": "VARCHAR",
    "source": "VARCHAR", "input": "VARCHAR", "leg": "VARCHAR", "market": "VARCHAR", "side": "VARCHAR",
    "line": "DOUBLE", "odds": "INTEGER", "player_id": "VARCHAR", "team": "VARCHAR", "opponent": "VARCHAR",
    "position": "VARCHAR", "game_id": "VARCHAR", "season": "INTEGER", "week": "INTEGER",
    "probability": "DOUBLE", "raw_probability": "DOUBLE", "projection": "DOUBLE", "reliability": "INTEGER",
    "parlay_probability": "DOUBLE", "parlay_odds": "INTEGER", "actual": "DOUBLE", "result": "VARCHAR",
}


def save_grades(con: duckdb.DuckDBPyConnection, graded: pd.DataFrame) -> tuple[int, int]:
    """Upsert settled legs into `prediction_grades`. Returns (newly graded, re-graded with a changed result)."""
    cols = ", ".join(f"{c} {t}" for c, t in GRADE_COLUMNS.items())
    con.execute(f"CREATE TABLE IF NOT EXISTS prediction_grades ({cols})")
    done = graded[graded["result"].notna()].copy()
    if done.empty:
        return 0, 0
    done["graded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for c in GRADE_COLUMNS:
        if c not in done.columns:
            done[c] = None
    done = done[list(GRADE_COLUMNS)].astype(object).where(done[list(GRADE_COLUMNS)].notna(), None)
    before = dict(con.execute("SELECT leg_id, result FROM prediction_grades").fetchall())
    new = sum(1 for lid in done["leg_id"] if lid not in before)
    changed = sum(1 for lid, res in zip(done["leg_id"], done["result"]) if lid in before and before[lid] != res)
    con.register("_grades", done)
    try:
        updates = ", ".join(f"{c} = excluded.{c}" for c in GRADE_COLUMNS if c not in ("leg_id", "graded_at"))
        con.execute(f"INSERT INTO prediction_grades SELECT * FROM _grades "
                    f"ON CONFLICT (leg_id) DO UPDATE SET {updates}")  # keeps the first graded_at
    finally:
        con.unregister("_grades")
    return new, changed


def load_grades(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    from cajetan.db import table_exists

    if not table_exists(con, "prediction_grades"):
        return pd.DataFrame()
    return con.execute("SELECT * FROM prediction_grades ORDER BY season, week, logged_at").df()


def summarize(df: pd.DataFrame, p_col: str = "probability") -> dict:
    """Hit rate, Brier score and log loss on decided legs, plus a reliability table."""
    d = df[df["result"].isin(["won", "lost"])]
    if d.empty:
        return {"n": 0}
    p = np.clip(d[p_col].to_numpy(float), 1e-4, 1 - 1e-4)
    y = (d["result"] == "won").to_numpy(float)
    bins = pd.cut(p, [0, .35, .45, .55, .65, 1], labels=["<35%", "35-45%", "45-55%", "55-65%", "65%+"])
    table = [{"bin": str(b), "n": int(m.sum()), "predicted": float(p[m].mean()), "hit_rate": float(y[m].mean())}
             for b, m in ((b, np.asarray(bins == b)) for b in bins.categories) if m.sum()]
    return {"n": int(len(d)), "hit_rate": float(y.mean()), "avg_prob": float(p.mean()),
            "brier": float(np.mean((p - y) ** 2)), "brier_coin": float(np.mean((0.5 - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "table": table}
