"""Combine legs into a parlay probability, accounting for same-game correlation.

Each leg is represented by a latent standard-normal variable for the stat it is
about (a player's yards, a team's margin, the game total, ...). Legs from the
same game get correlations estimated from historical player-games (see
calibration.py); the joint hit probability is then estimated by Monte Carlo
from a Gaussian copula. Legs from different games are treated as independent.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from cajetan import config
from cajetan.analysis.explain import LegResult, ParlayResult
from cajetan.calibration import Calibration
from cajetan.parsing import american_to_decimal

TEAM_VAR = {"ml": "margin", "spread": "margin", "game_total": "total", "team_total": "team_pts"}


def _sign(leg: LegResult) -> int:
    return -1 if leg.side == "under" else 1


def latent_corr(a: LegResult, b: LegResult, calib: Calibration) -> float:
    if not a.game_id or a.game_id != b.game_id:
        return 0.0
    a_team, b_team = a.market in TEAM_VAR, b.market in TEAM_VAR
    if a_team and b_team:
        va, vb = TEAM_VAR[a.market], TEAM_VAR[b.market]
        same = a.team == b.team
        if va == vb == "margin":
            return 0.95 if same else -0.95
        if va == vb == "team_pts":
            return 0.95 if same else 0.05
        if {va, vb} == {"margin", "total"}:
            return 0.0
        if {va, vb} == {"total", "team_pts"}:
            return 0.7
        return 0.65 if same else -0.65  # margin vs team points
    if a_team or b_team:
        team_leg, pl = (a, b) if a_team else (b, a)
        var = TEAM_VAR[team_leg.market]
        same = team_leg.team == pl.team
        if var == "margin":
            r = calib.env(pl.position, pl.market, "margin")
            return r if same else -r
        if var == "team_pts":
            return calib.env(pl.position, pl.market, "team_pts" if same else "opp_pts")
        return calib.env(pl.position, pl.market, "total")
    rel = "same" if a.player_id == b.player_id else "teammate" if a.team == b.team else "opponent"
    return calib.pair(rel, a.position, a.market, b.position, b.market)


def _nearest_psd(m: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(m)
    vals = np.clip(vals, 1e-6, None)
    fixed = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    return fixed / np.outer(d, d)


def combine(legs: list[LegResult], calib: Calibration, parlay_odds: int | None = None) -> ParlayResult:
    ok = [l for l in legs if not l.error]
    notes: list[str] = []
    if not ok:
        return ParlayResult(legs, 0.0, 0.0, 0.0, notes=["No legs could be rated."])
    p = np.array([l.probability for l in ok])
    indep = float(np.prod(p))
    n = len(ok)
    R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho = latent_corr(ok[i], ok[j], calib) * _sign(ok[i]) * _sign(ok[j])
            R[i, j] = R[j, i] = rho
            if abs(rho) >= 0.15:
                kind = "helps" if rho > 0 else "hurts"
                notes.append(f"Same-game correlation {rho:+.2f} between '{ok[i].leg.describe()}' and "
                             f"'{ok[j].leg.describe()}' ({kind} the parlay)")
    if np.allclose(R, np.eye(n)):
        joint = indep
    else:
        rng = np.random.default_rng(20261022)
        z = rng.multivariate_normal(np.zeros(n), _nearest_psd(R), size=config.PARLAY_SIMULATIONS, method="cholesky")
        thresholds = stats.norm.ppf(1 - np.clip(p, 1e-6, 1 - 1e-6))
        joint = float(np.mean(np.all(z > thresholds, axis=1)))
    strength = float(math.exp(np.mean(np.log(np.clip(p, 1e-9, 1)))))

    book_odds, book_p, ev = parlay_odds, None, None
    if book_odds is None and all(l.leg.odds is not None for l in ok) and n > 1:
        dec = math.prod(american_to_decimal(l.leg.odds) for l in ok)
        book_odds = round((dec - 1) * 100) if dec >= 2 else round(-100 / (dec - 1))
    if book_odds is not None:
        dec = american_to_decimal(book_odds)
        book_p = 1 / dec
        ev = joint * dec - 1
    weakest = min(ok, key=lambda l: l.probability)
    if n > 1:
        notes.insert(0, f"Weakest leg: '{weakest.leg.describe()}' at {weakest.confidence}%")
    if len(ok) < len(legs):
        notes.append(f"{len(legs) - len(ok)} leg(s) could not be rated and are excluded from the parlay math.")
    return ParlayResult(legs, joint, indep, strength, book_odds, book_p, ev, notes)
