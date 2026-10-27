"""Purely historical prediction model: gradient boosting on leak-free rolling features.

For every player-game the features only use information available before kickoff:

* season-to-date and last-3-game averages of 12 box-score stats and of the player's
  role (target share, air-yard share, carry share, snap share), plus last season's averages
* season-to-date Next Gen Stats (separation, cushion, rush yards over expected,
  completion % over expected, time to throw, aggressiveness, intended air yards)
* the week's injury report: the player's own designation and practice status, and the
  target/carry share vacated by teammates ruled Out or Doubtful
* what the opponent had allowed to the position so far that season (relative to league)
* age and years in the league, home/away, spread, total, implied points, wind and temperature

One HistGradientBoostingRegressor is trained per market on every season from
HISTORY_FIRST_SEASON. Accuracy is measured with rolling-origin validation: for each of
the last complete seasons, train on everything before it and score that season against
a naive "season average so far" guess. `cajetan train --tune` also picks the model
settings that score best across those folds.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from functools import lru_cache

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from cajetan import config

STATS = {
    "pass_yds": "passing_yards", "pass_att": "attempts", "pass_comp": "completions",
    "pass_tds": "passing_tds", "pass_int": "passing_interceptions", "rush_yds": "rushing_yards",
    "rush_att": "carries", "rec": "receptions", "rec_yds": "receiving_yards", "targets": "targets",
    "rush_rec_yds": "rushing_yards + receiving_yards", "tds": "rushing_tds + receiving_tds",
}
SHARES = ("tgt_share", "ay_share", "carry_share", "snap_pct")
NGS = ("separation", "cushion", "rec_iay", "yac_oe", "rush_eff", "stacked_pct", "ryoe", "ttt",
       "aggressiveness", "pass_iay", "cpoe")
TARGET = {m: m for m in STATS if m not in ("targets", "tds")} | {"anytime_td": "tds"}
MODEL_POS = {
    "pass_yds": ["QB"], "pass_att": ["QB"], "pass_comp": ["QB"], "pass_tds": ["QB"], "pass_int": ["QB"],
    "rush_yds": ["RB", "QB"], "rush_att": ["RB", "QB"], "rec": ["WR", "TE", "RB"],
    "rec_yds": ["WR", "TE", "RB"], "rush_rec_yds": ["RB", "WR", "TE"], "anytime_td": ["RB", "WR", "TE", "QB"],
}
COUNT_MARKETS = {"pass_att", "pass_comp", "pass_tds", "pass_int", "rush_att", "rec", "anytime_td"}
POS_CODE = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
MODEL_PATH = config.MODEL_DIR / "ml_models.joblib"
META_PATH = config.MODEL_DIR / "ml_meta.json"

DEFAULT_PARAMS = {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 1.0}
PARAM_GRID = [
    {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 100, "l2_regularization": 1.0},
    {"learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 3.0},
    {"learning_rate": 0.08, "max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 3.0},
    {"learning_rate": 0.03, "max_leaf_nodes": 63, "min_samples_leaf": 60, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_leaf_nodes": 7, "min_samples_leaf": 300, "l2_regularization": 0.0},
]
N_FOLDS = 3


def _base_sql(synthetic: bool, extra_where: str = "") -> str:
    cols = ", ".join(f"CAST({expr} AS DOUBLE) AS s_{k}" for k, expr in STATS.items())
    nulls = ", ".join(f"CAST(NULL AS DOUBLE) AS s_{k}" for k in STATS)
    share_nulls = ", ".join(f"CAST(NULL AS DOUBLE) AS s_{k}" for k in SHARES)
    sql = f"""
        SELECT w.player_id, w.position, w.season, w.week, w.game_id, w.team, w.opponent_team, {cols},
               CAST(ps.tgt_share AS DOUBLE) AS s_tgt_share, CAST(w.air_yards_share AS DOUBLE) AS s_ay_share,
               CAST(ps.carry_share AS DOUBLE) AS s_carry_share, CAST(ps.snap_pct AS DOUBLE) AS s_snap_pct
        FROM (SELECT * REPLACE (
                COALESCE(passing_yards, 0) AS passing_yards, COALESCE(attempts, 0) AS attempts,
                COALESCE(completions, 0) AS completions, COALESCE(passing_tds, 0) AS passing_tds,
                COALESCE(passing_interceptions, 0) AS passing_interceptions,
                COALESCE(rushing_yards, 0) AS rushing_yards, COALESCE(carries, 0) AS carries,
                COALESCE(receptions, 0) AS receptions, COALESCE(receiving_yards, 0) AS receiving_yards,
                COALESCE(targets, 0) AS targets, COALESCE(rushing_tds, 0) AS rushing_tds,
                COALESCE(receiving_tds, 0) AS receiving_tds)
              FROM player_week
              WHERE season_type IN ('REG', 'POST') AND position IN ('QB', 'RB', 'WR', 'TE') {extra_where}) w
        LEFT JOIN player_shares ps ON ps.player_id = w.player_id AND ps.game_id = w.game_id
    """
    if synthetic:
        sql += f"""
        UNION ALL
        SELECT $pid, $pos, $season, $week, $game_id, $team, $opp, {nulls}, {share_nulls}
        """
    return sql


def feature_sql(synthetic: bool = False, extra_where: str = "") -> str:
    keys = list(STATS) + list(SHARES)
    sd = ",\n".join(
        f"avg(s_{k}) OVER (PARTITION BY player_id, season ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS sd_{k},"
        f" avg(s_{k}) OVER (PARTITION BY player_id ORDER BY t ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS l3_{k}"
        for k in keys)
    pv = ", ".join(f"avg(s_{k}) AS pv_{k}" for k in keys)
    stat_keys = list(STATS)
    dg = ", ".join(f"sum(s_{k}) AS d_{k}" for k in stat_keys)
    dsd = ", ".join(f"avg(d.d_{k}) AS dsd_{k}" for k in stat_keys)
    lg_sum = ", ".join(f"sum(d_{k}) AS sum_{k}" for k in stat_keys)
    lg_cum = ", ".join(
        f"sum(sum_{k}) OVER (PARTITION BY season, position ORDER BY week ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
        f" / sum(cnt) OVER (PARTITION BY season, position ORDER BY week ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS lg_{k}"
        for k in stat_keys)
    rel = ", ".join(f"dr.dsd_{k} / NULLIF(lg.lg_{k}, 0) AS def_{k}" for k in stat_keys)
    ngs_cum = ", ".join(
        f"avg({k}) OVER (PARTITION BY player_id, season ORDER BY week ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS ngs_{k}"
        for k in NGS)
    ngs_sel = ", ".join(f"n.ngs_{k}" for k in NGS)
    return f"""
    WITH base AS ({_base_sql(synthetic, extra_where)}),
    ordered AS (SELECT *, season * 100 + week AS t FROM base),
    pf AS (
        SELECT *, {sd},
               count(s_pass_yds) OVER (PARTITION BY player_id, season ORDER BY t
                                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS sd_games
        FROM ordered
    ),
    prev AS (SELECT player_id, season + 1 AS season, count(*) AS pv_games, {pv}
             FROM base WHERE s_pass_yds IS NOT NULL GROUP BY player_id, season),
    defg AS (SELECT season, week, opponent_team AS defteam, position, {dg}
             FROM base WHERE s_pass_yds IS NOT NULL GROUP BY ALL),
    lgw AS (SELECT season, position, week, count(*) AS cnt, {lg_sum} FROM defg GROUP BY ALL),
    lg AS (SELECT season, position, week, {lg_cum} FROM lgw),
    dr AS (
        SELECT p.player_id, p.season, p.week, {dsd}, count(*) AS def_games
        FROM pf p JOIN defg d
          ON d.season = p.season AND d.defteam = p.opponent_team AND d.position = p.position AND d.week < p.week
        GROUP BY p.player_id, p.season, p.week
    ),
    ngs AS (SELECT player_id, season, week, {ngs_cum} FROM ngs_week),
    vac AS (
        SELECT p.player_id, p.season, p.week,
               sum(v.tgt_share) AS vac_tgt, sum(v.carry_share) AS vac_carry,
               sum(CASE WHEN v.position = p.position THEN v.tgt_share END) AS vac_tgt_pos,
               sum(CASE WHEN v.position = p.position THEN v.carry_share END) AS vac_carry_pos
        FROM pf p JOIN vacancy v
          ON v.team = p.team AND v.season = p.season AND v.week = p.week AND v.player_id <> p.player_id
        GROUP BY ALL
    )
    SELECT pf.*, prev.pv_games, {', '.join(f'prev.pv_{k}' for k in keys)},
           dr.def_games, {rel}, {ngs_sel},
           COALESCE(CASE i.status WHEN 'Questionable' THEN 1 WHEN 'Doubtful' THEN 2 WHEN 'Out' THEN 3 END, 0) AS inj_status,
           i.practice AS inj_practice,
           COALESCE(vac.vac_tgt, 0) AS vac_tgt, COALESCE(vac.vac_carry, 0) AS vac_carry,
           COALESCE(vac.vac_tgt_pos, 0) AS vac_tgt_pos, COALESCE(vac.vac_carry_pos, 0) AS vac_carry_pos,
           date_diff('day', bio.birth_date, make_date(pf.season, 9, 1)) / 365.25 AS age,
           pf.season - bio.entry_year AS years_exp,
           CASE WHEN tg.is_home THEN 1 ELSE 0 END AS is_home, tg.exp_margin, tg.total_line,
           tg.total_line / 2 + tg.exp_margin / 2 AS implied_pts,
           CASE WHEN s.roof IN ('dome', 'closed') THEN 1 ELSE 0 END AS indoors,
           CASE WHEN s.roof IN ('dome', 'closed') THEN 0 ELSE s.wind END AS wind,
           CASE WHEN s.roof IN ('dome', 'closed') THEN 70 ELSE s.temp END AS temp
    FROM pf
    LEFT JOIN prev USING (player_id, season)
    LEFT JOIN dr USING (player_id, season, week)
    LEFT JOIN vac USING (player_id, season, week)
    LEFT JOIN lg ON lg.season = pf.season AND lg.position = pf.position AND lg.week = (
        SELECT max(week) FROM lgw x WHERE x.season = pf.season AND x.position = pf.position AND x.week < pf.week)
    ASOF LEFT JOIN ngs n ON n.player_id = pf.player_id AND n.season = pf.season AND pf.week > n.week
    LEFT JOIN injury_status i ON i.player_id = pf.player_id AND i.season = pf.season AND i.week = pf.week
    LEFT JOIN player_bio bio ON bio.player_id = pf.player_id
    LEFT JOIN team_games tg ON tg.game_id = pf.game_id AND tg.team = pf.team
    LEFT JOIN schedules s ON s.game_id = pf.game_id
    """


def feature_columns() -> list[str]:
    cols = ["sd_games", "pv_games", "def_games", "is_home", "exp_margin", "total_line", "implied_pts", "pos_code",
            "inj_status", "inj_practice", "vac_tgt", "vac_carry", "vac_tgt_pos", "vac_carry_pos",
            "age", "years_exp", "indoors", "wind", "temp"]
    for k in STATS:
        cols += [f"sd_{k}", f"l3_{k}", f"pv_{k}", f"def_{k}"]
    for k in SHARES:
        cols += [f"sd_{k}", f"l3_{k}", f"pv_{k}"]
    cols += [f"ngs_{k}" for k in NGS]
    return cols


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pos_code"] = df["position"].map(POS_CODE)
    return df


_FRAMES: dict[int, pd.DataFrame] = {}


def build_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Features for every played player-game (cached per connection)."""
    key = id(con)
    if key not in _FRAMES:
        _FRAMES[key] = _prepare(con.execute(feature_sql()).df())
    return _FRAMES[key]


def _model(market: str, params: dict, feats: list[str]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson" if market in COUNT_MARKETS else "squared_error",
        max_iter=1000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        categorical_features=[feats.index("pos_code")] if "pos_code" in feats else None,
        random_state=7, **params,
    )


def _market_rows(df: pd.DataFrame, market: str) -> tuple[pd.DataFrame, pd.Series]:
    target_key = TARGET[market]
    rows = df[df["position"].isin(MODEL_POS[market]) & df[f"s_{target_key}"].notna()
              & (df["sd_games"].fillna(0).gt(0) | df["pv_games"].notna())]
    return rows, rows[f"s_{target_key}"].clip(lower=0)


def _usable(rows: pd.DataFrame) -> list[str]:
    """Feature columns with at least two distinct values for this market (e.g. QBs have no receiving NGS)."""
    return [c for c in feature_columns() if rows[c].nunique(dropna=True) >= 2]


def _naive(rows: pd.DataFrame, market: str, fallback: float) -> pd.Series:
    k = TARGET[market]
    return rows[f"sd_{k}"].fillna(rows[f"pv_{k}"]).fillna(fallback)


def _folds(df: pd.DataFrame, con: duckdb.DuckDBPyConnection, max_season: int | None) -> list[int]:
    complete = con.execute(
        "SELECT max(season) FROM schedules WHERE game_type = 'SB' AND home_score IS NOT NULL").fetchone()[0]
    if max_season is not None:
        complete = min(complete, max_season)
    seasons = sorted(int(s) for s in df["season"].unique() if s <= complete)
    return seasons[-N_FOLDS:] if len(seasons) > N_FOLDS else seasons[1:]


def _score(rows, y, market, folds, params) -> tuple[float, float]:
    maes, naives = [], []
    for s in folds:
        tr, te = rows["season"] < s, rows["season"] == s
        if tr.sum() < 500 or te.sum() < 100:
            continue
        feats = _usable(rows[tr])
        mdl = _model(market, params, feats).fit(rows.loc[tr, feats], y[tr])
        pred = mdl.predict(rows.loc[te, feats])
        maes.append(float(np.mean(np.abs(pred - y[te]))))
        naives.append(float(np.mean(np.abs(_naive(rows[te], market, float(y[tr].mean())) - y[te]))))
    return (float(np.mean(maes)), float(np.mean(naives))) if maes else (float("nan"), float("nan"))


def train(con: duckdb.DuckDBPyConnection, log=print, tune: bool = False, max_season: int | None = None,
          evaluate: bool = True) -> dict:
    """Fit one model per market. With `max_season`, only seasons up to it are used and the models are
    saved separately (for leak-free backtests of later seasons)."""
    log("Building leak-free features for every player-game")
    df = build_frame(con)
    if max_season is not None:
        df = df[df["season"] <= max_season]
    folds = _folds(df, con, max_season) if evaluate else []
    _, old_meta = _load_from(*_paths(None))
    models, meta = {}, {}
    for market in TARGET:
        rows, y = _market_rows(df, market)
        if len(rows) < 500:
            continue
        params = (old_meta or {}).get(market, {}).get("params", DEFAULT_PARAMS)
        mae = naive = float("nan")
        if tune and folds:
            best = None
            for cand in PARAM_GRID:
                m_, n_ = _score(rows, y, market, folds, cand)
                log(f"  {market:13s} " + ", ".join(f"{k}={v}" for k, v in cand.items()) + f" -> MAE {m_:.3f}")
                if best is None or m_ < best[0]:
                    best = (m_, n_, cand)
            mae, naive, params = best
        elif folds:
            mae, naive = _score(rows, y, market, folds, params)
        if folds:
            log(f"  {market:13s} rows={len(rows):6d}  validated on {', '.join(map(str, folds))}: "
                f"MAE {mae:.3f} (naive {naive:.3f})")
        feats = _usable(rows)
        models[market] = _model(market, params, feats).fit(rows[feats], y)
        seasons = f"{int(rows['season'].min())}-{int(rows['season'].max())}"
        meta[market] = {"rows": int(len(rows)), "seasons": seasons, "holdout_season": folds[-1] if folds else None,
                        "folds": folds, "mae": mae, "naive_mae": naive, "params": params, "features": feats}
    model_path, meta_path = _paths(max_season)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, model_path)
    meta_path.write_text(json.dumps(meta, indent=1))
    _load_from.cache_clear()
    return meta


def _paths(max_season: int | None):
    if max_season is None:
        return MODEL_PATH, META_PATH
    return (config.MODEL_DIR / f"ml_models_through_{max_season}.joblib",
            config.MODEL_DIR / f"ml_meta_through_{max_season}.json")


@lru_cache(maxsize=4)
def _load_from(model_path, meta_path):
    if not model_path.exists() or not meta_path.exists():
        return None, None
    return joblib.load(model_path), json.loads(meta_path.read_text())


_ACTIVE: list = [None]  # season cutoff of the models in use (None = the production models)
_PRELOADED: dict = {}


def _load():
    return _load_from(*_paths(_ACTIVE[0]))


@contextmanager
def models_through(max_season: int | None, frame: pd.DataFrame | None = None):
    """Use models trained only on seasons <= max_season, optionally with precomputed features."""
    prev, prev_frame = _ACTIVE[0], _PRELOADED.get("frame")
    _ACTIVE[0] = max_season
    if frame is not None:
        _PRELOADED["frame"] = frame.set_index(["player_id", "game_id"])
    try:
        yield
    finally:
        _ACTIVE[0] = prev
        if prev_frame is None:
            _PRELOADED.pop("frame", None)
        else:
            _PRELOADED["frame"] = prev_frame


def predict(con: duckdb.DuckDBPyConnection, market: str, player, game, conditions=None) -> tuple[float, dict] | None:
    models, meta = _load()
    pos = "RB" if player.position == "FB" else player.position
    if not models or market not in models or pos not in MODEL_POS[market]:
        return None
    feats = meta[market].get("features") or feature_columns()
    pre = _PRELOADED.get("frame")
    if pre is not None and (player.player_id, game.game_id) in pre.index:
        x = pre.loc[[(player.player_id, game.game_id)], feats]
    else:
        params = {"pid": player.player_id, "pos": pos, "season": game.season, "week": game.week,
                  "game_id": game.game_id, "team": game.team, "opp": game.opponent}
        # Rate the game as if unplayed: only rows from before it feed the features.
        sql = feature_sql(synthetic=True, extra_where="AND season >= $season - 1 AND "
                                                      "(season < $season OR (season = $season AND week < $week))")
        df = con.execute(f"SELECT * FROM ({sql}) WHERE player_id = $pid AND game_id = $game_id", params).df()
        if df.empty:
            return None
        df = _prepare(df)
        if conditions is not None and not conditions.indoors:
            df["wind"] = df["wind"].fillna(conditions.wind if conditions.wind is not None else np.nan)
            df["temp"] = df["temp"].fillna(conditions.temp if conditions.temp is not None else np.nan)
        x = df[feats]
    return float(models[market].predict(x)[0]), meta[market]
