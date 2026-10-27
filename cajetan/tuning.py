"""`cajetan fit`: estimate the hand-set constants from data and fit the final probability calibration.

* ML_BLEND_BY_MARKET  per-market share of the ML model, chosen by log loss on the backtest
* MARKET_BLEND_WEIGHT share of the betting line in team projections, by log loss on the backtest
* MARGIN_SD, TOTAL_SD, TEAM_POINTS_SD  spread of real outcomes around the blended team projections
* RECENCY_HALF_LIFE, PRIOR_SEASON_WEIGHT  the weighting of a player's game log that best predicts his
  next game's volume (carries, targets, pass attempts) in recent seasons
* probability calibration  see probcal.py; fitted on backtest legs re-scored with the weights above,
  plus graded prediction-log legs (real sportsbook lines)
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm

from cajetan import config, probcal
from cajetan.analysis.player_eval import outcome
from cajetan.calibration import Calibration

ML_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
MIN_MARKET_LEGS = 200
HALF_LIVES = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 1000.0)
PRIOR_WEIGHTS = (0.0, 0.1, 0.25, 0.4, 0.6, 1.0)


def _ll(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def player_probs(rows: pd.DataFrame, weights: dict[str, float], calib: Calibration) -> np.ndarray:
    """Re-score player legs with the given ML weights (rows without an ML projection keep the engine's)."""
    out = np.empty(len(rows))
    for i, r in enumerate(rows.itertuples(index=False)):
        mean = r.c_structural
        if r.c_ml is not None and r.c_ml == r.c_ml:
            w = weights.get(r.market, r.c_ml_weight)
            mean = (1 - w) * r.c_structural + w * r.c_ml
        out[i] = outcome(calib, r.market, r.position, max(mean, 1e-3), r.line, r.side)[0]
    return out


def fit_ml_weights(rows: pd.DataFrame, calib: Calibration, log) -> tuple[dict, dict]:
    weights, evidence = {}, {}
    for market, sub in rows.groupby("market"):
        sub = sub[sub["c_ml"].notna()]
        if len(sub) < MIN_MARKET_LEGS:
            log(f"  {market:13s} {len(sub):5d} legs: too few, keeping the default weight")
            continue
        y = (sub["result"] == "won").to_numpy(float)
        scores = {float(w): _ll(player_probs(sub, {market: w}, calib), y) for w in ML_GRID}
        best = min(scores, key=scores.get)
        weights[market] = best
        evidence[market] = {"n": int(len(sub)), "log_loss_engine_only": round(scores[0.0], 5),
                            "log_loss_ml_only": round(scores[1.0], 5), "log_loss_best": round(scores[best], 5)}
        log(f"  {market:13s} {len(sub):5d} legs: ML weight {best:.0%}  (log loss engine-only {scores[0.0]:.4f}, "
            f"ML-only {scores[1.0]:.4f}, best {scores[best]:.4f})")
    return weights, evidence


def _team_projection(sub: pd.DataFrame, w: float) -> tuple[np.ndarray, np.ndarray]:
    """Blended projection and the realised value for each team leg."""
    model = np.select([sub["market"].isin(["ml", "spread"]), sub["market"] == "game_total"],
                      [sub["c_model_margin"], sub["c_model_total"]], sub["c_model_team_pts"])
    market = np.select([sub["market"].isin(["ml", "spread"]), sub["market"] == "game_total"],
                       [sub["c_market_margin"], sub["c_market_total"]], sub["c_market_team_pts"])
    market = np.where(np.isnan(market.astype(float)), model, market)
    return (1 - w) * model + w * market, sub["actual"].to_numpy(float)


def _team_probs(sub: pd.DataFrame, proj: np.ndarray, sds: dict) -> np.ndarray:
    sd = sub["market"].map({"ml": sds["MARGIN_SD"], "spread": sds["MARGIN_SD"], "game_total": sds["TOTAL_SD"],
                            "team_total": sds["TEAM_POINTS_SD"]}).to_numpy(float)
    thresh = np.where(sub["market"] == "ml", 0.0,
                      np.where(sub["market"] == "spread", -sub["line"].fillna(0), sub["line"].fillna(0)))
    return 1 - norm.cdf((thresh - proj) / sd)  # P(value > threshold): win / cover / over


def fit_team(rows: pd.DataFrame, log) -> tuple[dict, dict]:
    sub = rows[rows["result"].isin(["won", "lost"])]
    if len(sub) < 200:
        log(f"  only {len(sub)} team legs: keeping the market blend and standard deviations")
        return {}, {}
    y = (sub["result"] == "won").to_numpy(float)
    best = None
    for w in np.round(np.arange(0, 1.0001, 0.05), 2):
        sds = {}
        for key, mk in (("MARGIN_SD", ["spread"]), ("TOTAL_SD", ["game_total"]), ("TEAM_POINTS_SD", ["team_total"])):
            m = sub["market"].isin(mk)
            proj, actual = _team_projection(sub[m], w)
            sds[key] = float(np.sqrt(np.mean((actual - proj) ** 2)))
        ll = _ll(_team_probs(sub, _team_projection(sub, w)[0], sds), y)
        if best is None or ll < best[0]:
            best = (ll, float(w), sds)
    ll, w, sds = best
    current = {"MARGIN_SD": config.MARGIN_SD, "TOTAL_SD": config.TOTAL_SD, "TEAM_POINTS_SD": config.TEAM_POINTS_SD}
    ll_cur = _ll(_team_probs(sub, _team_projection(sub, config.MARKET_BLEND_WEIGHT)[0], current), y)
    log(f"  market blend {w:.0%}, margin SD {sds['MARGIN_SD']:.1f}, total SD {sds['TOTAL_SD']:.1f}, team points SD "
        f"{sds['TEAM_POINTS_SD']:.1f}  (log loss {ll_cur:.4f} with current settings -> {ll:.4f}, {len(sub)} legs)")
    values = {"MARKET_BLEND_WEIGHT": w, **{k: round(v, 2) for k, v in sds.items()}}
    return values, {"n": int(len(sub)), "log_loss_current": round(ll_cur, 5), "log_loss_fitted": round(ll, 5)}


def fit_recency(con: duckdb.DuckDBPyConnection, log) -> tuple[dict, dict]:
    """Grid-search the game-log weighting on next-game volume error (2023 season on)."""
    df = con.execute(
        """SELECT player_id, position, season, week,
                  COALESCE(carries, 0) AS carries, COALESCE(targets, 0) AS targets, COALESCE(attempts, 0) AS attempts
           FROM player_week WHERE season_type IN ('REG', 'POST') AND position IN ('QB', 'RB', 'WR', 'TE')
             AND season >= 2022 ORDER BY player_id, season, week"""
    ).df()
    vol_col = {"QB": "attempts", "RB": "carries", "WR": "targets", "TE": "targets"}
    errs = np.zeros((len(HALF_LIVES), len(PRIOR_WEIGHTS)))
    n = 0
    hl = np.array(HALF_LIVES)[:, None, None]
    pw = np.array(PRIOR_WEIGHTS)[None, :, None]
    for (_, pos), g in df.groupby(["player_id", "position"], sort=False):
        v = g[vol_col[pos]].to_numpy(float)
        seasons = g["season"].to_numpy()
        for i in range(len(g)):
            s = seasons[i]
            if s < 2023:
                continue
            j = np.nonzero((seasons[:i] >= s - 1))[0]
            cur = int((seasons[j] == s).sum())
            if cur < 2 or v[j][seasons[j] == s].mean() < 3:  # established role this season
                continue
            age = (i - 1 - j)[None, None, :]
            w = 0.5 ** (age / hl) * np.where(seasons[j] == s, 1.0, pw)
            pred = (w * v[j]).sum(-1) / w.sum(-1)
            errs += np.abs(pred - v[i])
            n += 1
    errs /= max(n, 1)
    a, b = np.unravel_index(errs.argmin(), errs.shape)
    cur = errs[HALF_LIVES.index(4.0), PRIOR_WEIGHTS.index(0.25)]
    log(f"  recency half-life {HALF_LIVES[a]:g} games, prior-season weight {PRIOR_WEIGHTS[b]:g}  "
        f"(next-game volume MAE {cur:.3f} at the defaults 4 / 0.25 -> {errs[a, b]:.3f}, {n:,} player-games)")
    return ({"RECENCY_HALF_LIFE": float(HALF_LIVES[a]), "PRIOR_SEASON_WEIGHT": float(PRIOR_WEIGHTS[b])},
            {"n": n, "mae_default": round(float(cur), 4), "mae_fitted": round(float(errs[a, b]), 4)})


def calibration_rows(bt: pd.DataFrame, ml_weights: dict, team_values: dict, calib: Calibration, log) -> pd.DataFrame:
    rows = []
    pl = bt[(bt["kind"] == "player") & bt["result"].isin(["won", "lost"])]
    if len(pl):
        rows.append(pd.DataFrame({"market": pl["market"], "side": pl["side"],
                                  "p": player_probs(pl, ml_weights, calib),
                                  "won": (pl["result"] == "won").astype(int),
                                  "fold": pl["season"].astype(str) + "-" + pl["week"].astype(str)}))
    tm = bt[(bt["kind"] == "team") & bt["result"].isin(["won", "lost"])]
    if len(tm):
        w = team_values.get("MARKET_BLEND_WEIGHT", config.MARKET_BLEND_WEIGHT)
        sds = {k: team_values.get(k, getattr(config, k)) for k in ("MARGIN_SD", "TOTAL_SD", "TEAM_POINTS_SD")}
        rows.append(pd.DataFrame({"market": tm["market"], "side": tm["side"],
                                  "p": _team_probs(tm, _team_projection(tm, w)[0], sds),
                                  "won": (tm["result"] == "won").astype(int),
                                  "fold": tm["season"].astype(str) + "-" + tm["week"].astype(str)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit(con: duckdb.DuckDBPyConnection, bt: pd.DataFrame, graded: pd.DataFrame | None = None,
        log: Callable[[str], None] = print) -> dict:
    calib = Calibration.load()
    values: dict = {}
    evidence: dict = {"backtest_legs": int(len(bt))}

    log("Recency weighting of game logs (next-game volume)")
    v, e = fit_recency(con, log)
    values.update(v)
    evidence["recency"] = e

    log("ML blend weight per market (backtest log loss)")
    players = bt[(bt["kind"] == "player") & bt["result"].isin(["won", "lost"])]
    ml_weights, e = fit_ml_weights(players, calib, log)
    values["ML_BLEND_BY_MARKET"] = ml_weights
    evidence["ml_blend"] = e

    log("Team markets: market blend and outcome spread (backtest log loss)")
    team_values, e = fit_team(bt[bt["kind"] == "team"], log)
    values.update(team_values)
    evidence["team"] = e

    log("Probability calibration (two-fold cross-validated)")
    rows = calibration_rows(bt, ml_weights, team_values, calib, log)
    if graded is not None and len(graded):
        g = graded[graded["result"].isin(["won", "lost"])]
        if len(g):
            log(f"  including {len(g)} graded prediction-log legs at real sportsbook lines")
            rows = pd.concat([rows, pd.DataFrame({
                "market": g["market"], "side": g["side"], "p": g["raw_probability"],
                "won": (g["result"] == "won").astype(int), "fold": g["parlay_id"]})], ignore_index=True)
    pc = probcal.fit(rows, log)
    probcal.save(pc)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.TUNING_PATH.write_text(json.dumps({
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "values": values, "evidence": evidence}, indent=1))
    return {"values": values, "evidence": evidence, "probcal": pc}


def score(bt: pd.DataFrame) -> pd.DataFrame:
    """Per-market accuracy of the backtest as rated (raw probabilities)."""
    d = bt[bt["result"].isin(["won", "lost"])].copy()
    d["y"] = (d["result"] == "won").astype(float)
    out = []
    for market, g in d.groupby("market"):
        p = np.clip(g["p"].to_numpy(float), 1e-4, 1 - 1e-4)
        y = g["y"].to_numpy()
        base = np.full(len(y), y.mean())  # knowing only the overall hit rate
        out.append({"market": market, "legs": len(g), "hit_rate": y.mean(), "avg_prob": p.mean(),
                    "brier": np.mean((p - y) ** 2), "brier_base": np.mean((base - y) ** 2),
                    "log_loss": _ll(p, y), "log_loss_base": _ll(base, y)})
    return pd.DataFrame(out)


def describe_value(name: str, value) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k} {v:.0%}" for k, v in sorted(value.items()))
    return f"{value:g}" if isinstance(value, float) and not math.isnan(value) else str(value)
