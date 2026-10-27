"""Game environment: how expected margin and total move team volume, and team scoring baselines."""

from __future__ import annotations

import math

import duckdb
import numpy as np

from cajetan import config
from cajetan.db import AsOf

_SCRIPT_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


def script_model(con: duckdb.DuckDBPyConnection, asof: AsOf) -> dict[str, np.ndarray]:
    """OLS of team pass attempts / carries on market-implied margin and total, fit on all prior games."""
    key = (id(con), asof)
    if key in _SCRIPT_CACHE:
        return _SCRIPT_CACHE[key]
    df = con.execute(
        f"""
        SELECT exp_margin, total_line, pass_att, carries FROM team_games
        WHERE {asof.clause(min_season=config.FIRST_SEASON)} AND game_type = 'REG'
          AND exp_margin IS NOT NULL AND total_line IS NOT NULL AND pass_att IS NOT NULL
        """
    ).df()
    out: dict[str, np.ndarray] = {}
    if len(df) > 50:
        X = np.column_stack([np.ones(len(df)), df["exp_margin"], df["total_line"]])
        for col in ("pass_att", "carries"):
            out[col], *_ = np.linalg.lstsq(X, df[col].to_numpy(), rcond=None)
    _SCRIPT_CACHE[key] = out
    return out


def predict_volume(beta: np.ndarray, margin: float, total: float) -> float:
    return float(beta[0] + beta[1] * margin + beta[2] * total)


def team_scoring(con: duckdb.DuckDBPyConnection, team: str, asof: AsOf) -> tuple[float, int]:
    """Recency-weighted points per game for `team` and current-season games played."""
    df = con.execute(
        f"""
        SELECT season, week, points FROM team_games
        WHERE team = ? AND points IS NOT NULL AND {asof.clause()}
        ORDER BY season, week
        """,
        [team],
    ).df()
    if df.empty:
        return math.nan, 0
    age = np.arange(len(df))[::-1]
    w = 0.5 ** (age / 6.0) * np.where(df["season"] == asof.season, 1.0, config.PRIOR_SEASON_WEIGHT)
    return float((df["points"] * w).sum() / w.sum()), int((df["season"] == asof.season).sum())


def games_played(con: duckdb.DuckDBPyConnection, team: str, asof: AsOf) -> int:
    row = con.execute(
        f"SELECT count(*) FROM team_games WHERE team = ? AND points IS NOT NULL AND season = ? AND week < ?",
        [team, asof.season, asof.week],
    ).fetchone()
    return int(row[0])
