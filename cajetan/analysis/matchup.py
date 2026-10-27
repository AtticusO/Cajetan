"""Play-level aggregation helpers: player, defense and league rates split by scheme buckets.

Every query is restricted to games before the rating's as-of point and pools the
current season with the previous one (down-weighted), so early-season ratings
still have a sample to stand on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from cajetan import config
from cajetan.analysis.markets import FAMILIES
from cajetan.db import AsOf

PCT_METRICS = {"complete_pass", "pass_touchdown", "interception", "rush_touchdown"}


def shrink(total: float, n: float, prior_rate: float, k: float) -> float:
    """Empirical-Bayes style shrinkage of total/n toward prior_rate with k pseudo-observations."""
    return (total + k * prior_rate) / (n + k) if (n + k) > 0 else prior_rate


def fmt_metric(metric: str, value: float) -> str:
    if metric in PCT_METRICS:
        return f"{value:.1%}"
    return f"{value:.2f}" if abs(value) < 10 else f"{value:.1f}"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


@dataclass
class Split:
    """Aggregated sample: weighted count, current-season count, weighted metric sum."""

    n: float = 0.0
    n_cur: float = 0.0
    val: float = 0.0

    @property
    def rate(self) -> float:
        return self.val / self.n if self.n else math.nan


class PlayStats:
    def __init__(self, con: duckdb.DuckDBPyConnection, asof: AsOf):
        self.con = con
        self.asof = asof
        self._cache: dict[tuple, pd.DataFrame] = {}

    def query(self, family: str, metric: str, *, group: str | None = None, dim: str | None = None,
              pos: str | None = None, player_id: str | None = None, extra: str | None = None) -> pd.DataFrame:
        """Columns: [group], [bucket], n, n_cur, val."""
        key = (family, metric, group, dim, pos, player_id, extra)
        if key in self._cache:
            return self._cache[key]
        fam = FAMILIES[family]
        where = [self.asof.clause(), fam["filter"]]
        params: list = []
        if pos and fam["pos_col"]:
            where.append(f"{fam['pos_col']} = ?")
            params.append(pos)
        if player_id:
            where.append(f"{fam['pid']} = ?")
            params.append(player_id)
        if dim:
            where.append(f"{dim} IS NOT NULL")
        if extra:
            where.append(extra)
        cols = [c for c in (group, f"{dim} AS bucket" if dim else None) if c]
        select = ", ".join(cols + [
            f"sum({self.asof.weight()}) AS n",
            f"sum(CASE WHEN season = {self.asof.season} THEN 1 ELSE 0 END) AS n_cur",
            f"sum({self.asof.weight()} * ({metric})) AS val",
        ])
        sql = f"SELECT {select} FROM plays WHERE {' AND '.join(where)}"
        if cols:
            sql += " GROUP BY ALL"
        df = self.con.execute(sql, params).df()
        self._cache[key] = df
        return df

    def total(self, family: str, metric: str, **kw) -> Split:
        df = self.query(family, metric, **kw)
        if df.empty or pd.isna(df["n"].iloc[0]):
            return Split()
        return Split(float(df["n"].sum()), float(df["n_cur"].sum()), float(df["val"].sum()))

    def buckets(self, family: str, metric: str, dim: str, **kw) -> dict[str, Split]:
        df = self.query(family, metric, dim=dim, **kw)
        return {r.bucket: Split(float(r.n), float(r.n_cur), float(r.val)) for r in df.itertuples()}

    def defense_table(self, family: str, metric: str, pos: str | None, k: float,
                      extra: str | None = None) -> pd.DataFrame:
        """Shrunk rate allowed by every defense with its rank (1 = stingiest)."""
        df = self.query(family, metric, group="defteam", pos=pos, extra=extra).copy()
        lg = df["val"].sum() / df["n"].sum() if df["n"].sum() else math.nan
        df["rate"] = df["val"] / df["n"]
        df["shrunk"] = [shrink(v, n, lg, k) for v, n in zip(df["val"], df["n"])]
        df["rank"] = df["shrunk"].rank(method="min").astype(int)
        df.attrs["league"] = lg
        return df.set_index("defteam")


def weighted_log(con: duckdb.DuckDBPyConnection, player_id: str, asof: AsOf) -> pd.DataFrame:
    """The player's game log (current + previous season) with recency weights."""
    df = con.execute(
        f"""
        SELECT w.season, w.week, w.game_id, w.team, w.opponent_team, w.position,
               COALESCE(w.attempts, 0) attempts, COALESCE(w.completions, 0) completions,
               COALESCE(w.passing_yards, 0) passing_yards, COALESCE(w.passing_tds, 0) passing_tds,
               COALESCE(w.passing_interceptions, 0) passing_interceptions,
               COALESCE(w.carries, 0) - COALESCE(k.kneels, 0) carries, COALESCE(w.rushing_yards, 0) rushing_yards,
               COALESCE(w.rushing_tds, 0) rushing_tds, COALESCE(w.targets, 0) targets,
               COALESCE(w.receptions, 0) receptions, COALESCE(w.receiving_yards, 0) receiving_yards,
               COALESCE(w.receiving_tds, 0) receiving_tds,
               tg.exp_margin, tg.total_line
        FROM player_week w
        LEFT JOIN team_games tg ON tg.game_id = w.game_id AND tg.team = w.team
        -- kneel-downs count as official carries but are excluded from play-level efficiency
        LEFT JOIN (SELECT game_id, rusher_player_id, count(*) AS kneels FROM pbp
                   WHERE qb_kneel = 1 GROUP BY ALL) k
               ON k.game_id = w.game_id AND k.rusher_player_id = w.player_id
        WHERE w.player_id = ? AND {asof.clause('w')} AND w.season_type IN ('REG', 'POST')
        ORDER BY w.season, w.week
        """,
        [player_id],
    ).df()
    if df.empty:
        df["w"] = []
        return df
    age = np.arange(len(df))[::-1]  # 0 = most recent game
    decay = 0.5 ** (age / config.RECENCY_HALF_LIFE)
    season_w = np.where(df["season"] == asof.season, 1.0, config.PRIOR_SEASON_WEIGHT)
    df["w"] = decay * season_w
    return df


def wmean(df: pd.DataFrame, expr: str | pd.Series) -> float:
    vals = df.eval(expr) if isinstance(expr, str) else expr
    return float((vals * df["w"]).sum() / df["w"].sum()) if len(df) and df["w"].sum() > 0 else math.nan
