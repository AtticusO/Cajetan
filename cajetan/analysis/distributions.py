"""Probability that a stat goes over/under a line, given a projected mean and dispersion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Outcome:
    p_over: float
    p_under: float
    p_push: float

    def p_side(self, side: str) -> float:
        """Win probability for a side with pushes removed (a push refunds the leg)."""
        win = self.p_over if side in ("over", "yes") else self.p_under
        lose = self.p_under if side in ("over", "yes") else self.p_over
        return win / (win + lose) if win + lose > 0 else 0.5


def _thresholds(line: float) -> tuple[float, float]:
    """Integer thresholds: under wins at <= lo, over wins at >= hi."""
    if float(line).is_integer():
        return line - 1, line + 1
    return math.floor(line), math.ceil(line)


def gamma_outcome(mean: float, cv: float, line: float) -> Outcome:
    """Yardage props: gamma with the given coefficient of variation (continuity-corrected)."""
    mean = max(mean, 0.1)
    shape = 1.0 / max(cv, 0.05) ** 2
    dist = stats.gamma(a=shape, scale=mean / shape)
    lo, hi = _thresholds(line)
    p_under = float(dist.cdf(lo + 0.5))
    p_over = float(dist.sf(hi - 0.5))
    return Outcome(p_over, p_under, max(0.0, 1 - p_over - p_under))


def _empirical_cdf(mean: float, quantiles: list[float]):
    q = np.asarray(quantiles, dtype=float) * max(mean, 0.1)
    q = q + np.arange(len(q)) * 1e-9  # strictly increasing for interpolation
    ps = np.linspace(0, 1, len(q))
    return lambda x: float(np.interp(x, q, ps, left=0.0, right=1.0))


def empirical_outcome(mean: float, quantiles: list[float], line: float) -> Outcome:
    """Yardage props from the historical shape of outcome / expectation, scaled to `mean`."""
    cdf = _empirical_cdf(mean, quantiles)
    lo, hi = _thresholds(line)
    p_under = cdf(lo + 0.5)
    p_over = 1 - cdf(hi - 0.5)
    return Outcome(p_over, p_under, max(0.0, 1 - p_over - p_under))


def empirical_median(mean: float, quantiles: list[float]) -> float:
    q = np.asarray(quantiles) * max(mean, 0.1)
    return float(np.interp(0.5, np.linspace(0, 1, len(q)), q))


def gamma_median(mean: float, cv: float) -> float:
    shape = 1.0 / max(cv, 0.05) ** 2
    return float(stats.gamma(a=shape, scale=max(mean, 0.1) / shape).median())


def count_median(mean: float, r: float | None) -> float:
    mean = max(mean, 1e-6)
    dist = stats.poisson(mean) if r is None or r > 1e4 else stats.nbinom(n=r, p=r / (r + mean))
    return float(dist.median())


def count_outcome(mean: float, line: float, r: float | None) -> Outcome:
    """Count props: negative binomial with size `r` (Poisson when r is None / huge)."""
    mean = max(mean, 1e-6)
    if r is None or r > 1e4:
        dist = stats.poisson(mean)
    else:
        dist = stats.nbinom(n=r, p=r / (r + mean))
    lo, hi = _thresholds(line)
    p_under = float(dist.cdf(lo))
    p_over = float(dist.sf(hi - 1))
    return Outcome(p_over, p_under, max(0.0, 1 - p_over - p_under))


def normal_outcome(mean: float, sd: float, line: float) -> Outcome:
    """Team markets (margins, totals): P(value > line), continuity-corrected for whole-number lines."""
    dist = stats.norm(mean, sd)
    lo, hi = _thresholds(line)
    p_under = float(dist.cdf(lo + 0.5))
    p_over = float(dist.sf(hi - 0.5))
    return Outcome(p_over, p_under, max(0.0, 1 - p_over - p_under))
