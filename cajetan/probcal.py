"""Final probability calibration: map raw ratings to observed hit rates.

`cajetan fit` fits one monotone (isotonic) curve per group of markets, from backtest
legs and graded prediction-log legs, on the probability of the "up" side (over / yes /
win / cover). The curve is shrunk toward "no change" when the sample is small, and a
group is only switched on if it improves log loss in two-fold cross-validation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from cajetan import config

GROUPS = {
    "yardage": ("pass_yds", "rush_yds", "rec_yds", "rush_rec_yds"),
    "counts": ("pass_att", "pass_comp", "pass_tds", "pass_int", "rush_att", "rec"),
    "anytime_td": ("anytime_td",),
    "team": ("ml", "spread", "game_total", "team_total"),
}
UP_SIDES = ("over", "yes", "win", "cover")
GRID = np.round(np.linspace(0.01, 0.99, 99), 2)
SHRINK_N = 1000  # legs at which the fitted curve and "no change" get equal weight
MIN_LEGS = 300
MIN_GAIN = 0.0005  # log-loss improvement required to switch a group on


def group_of(market: str | None) -> str | None:
    return next((g for g, ms in GROUPS.items() if market in ms), None)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class ProbCalibration:
    def __init__(self, data: dict | None = None):
        self.data = data or {}

    @classmethod
    def load(cls) -> "ProbCalibration":
        try:
            return cls(json.loads(config.PROB_CALIBRATION_PATH.read_text()))
        except (OSError, ValueError):
            return cls()

    def apply(self, market: str | None, side: str, p: float) -> float:
        curve = self.data.get("groups", {}).get(group_of(market) or "", {})
        if not curve.get("active"):
            return p
        up = side in UP_SIDES
        p_up = p if up else 1 - p
        q = float(np.interp(p_up, GRID, curve["y"]))
        return q if up else 1 - q


def _curve(p_up: np.ndarray, y_up: np.ndarray) -> np.ndarray:
    iso = IsotonicRegression(y_min=0.005, y_max=0.995, out_of_bounds="clip").fit(p_up, y_up)
    w = len(p_up) / (len(p_up) + SHRINK_N)
    return w * iso.predict(GRID) + (1 - w) * GRID


def fit(rows: pd.DataFrame, log=print) -> dict:
    """rows: market, side, p (raw probability for the side), won (1/0), fold (any hashable split key)."""
    out = {"fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "groups": {}}
    rows = rows.assign(group=rows["market"].map(group_of)).dropna(subset=["group"])
    for g, sub in rows.groupby("group"):
        up = sub["side"].isin(UP_SIDES).to_numpy()
        p_up = np.where(up, sub["p"], 1 - sub["p"])
        y_up = np.where(up, sub["won"], 1 - sub["won"]).astype(float)
        entry = {"n": int(len(sub)), "active": False}
        if len(sub) >= MIN_LEGS:
            fold = (pd.util.hash_array(sub["fold"].astype(str).to_numpy()) % 2).astype(bool)
            pred = np.empty(len(sub))
            for f in (True, False):
                train, test = fold != f, fold == f
                pred[test] = np.interp(p_up[test], GRID, _curve(p_up[train], y_up[train]))
            raw_ll, cal_ll = log_loss(p_up, y_up), log_loss(pred, y_up)
            entry.update(raw_log_loss=round(raw_ll, 5), cv_log_loss=round(cal_ll, 5),
                         active=bool(raw_ll - cal_ll > MIN_GAIN), y=[round(float(v), 4) for v in _curve(p_up, y_up)])
            bins = pd.cut(p_up, [0, .3, .4, .45, .5, .55, .6, .7, 1])
            entry["reliability"] = [
                {"bin": str(b), "n": int(m.sum()), "predicted": round(float(p_up[m].mean()), 3),
                 "hit_rate": round(float(y_up[m].mean()), 3)}
                for b, m in ((b, np.asarray(bins == b)) for b in bins.categories) if m.sum()]
            log(f"  {g:10s} {len(sub):6,d} legs  log loss raw {raw_ll:.4f} -> calibrated (cross-validated) "
                f"{cal_ll:.4f}: {'ON' if entry['active'] else 'off (no reliable gain)'}")
        else:
            log(f"  {g:10s} {len(sub):6,d} legs  too few to calibrate (need {MIN_LEGS})")
        out["groups"][g] = entry
    return out


def save(data: dict) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.PROB_CALIBRATION_PATH.write_text(json.dumps(data, indent=1))
