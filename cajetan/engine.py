"""Rate a list of leg strings: parse -> evaluate each leg -> calibrate -> combine into a parlay."""

from __future__ import annotations

import duckdb

from cajetan.analysis import parlay, player_eval, team_eval
from cajetan.analysis.explain import Factor, LegResult, ParlayResult
from cajetan.calibration import Calibration
from cajetan.parsing import Leg, LegParseError, parse_leg
from cajetan.probcal import ProbCalibration


def rate_leg(con: duckdb.DuckDBPyConnection, raw: str | Leg, calib: Calibration, week: int | None = None,
             season: int | None = None, probcal: ProbCalibration | None = None) -> LegResult:
    """Rate one leg. `probcal=None` returns the raw (uncalibrated) probability, as the backtest needs."""
    if isinstance(raw, Leg):
        leg = raw
    else:
        try:
            leg = parse_leg(raw)
        except LegParseError as e:
            return LegResult(leg=Leg(raw, "?", raw, "ml", "?", None), description=raw, market_label="?",
                             side="?", line=None, projection=None, probability=0.0, error=str(e))
    try:
        if leg.kind == "team":
            res = team_eval.evaluate(con, leg, week, season, calib)
        else:
            res = player_eval.evaluate(con, leg, calib, week, season)
    except Exception as e:  # report the problem for this leg and keep rating the rest
        return LegResult(leg=leg, description=leg.describe(), market_label=leg.market_label, side=leg.side,
                         line=leg.line, projection=None, probability=0.0, error=str(e) or type(e).__name__)
    res.raw_probability = res.probability
    if probcal is not None:
        p = probcal.apply(res.market, res.side, res.probability)
        if abs(p - res.probability) >= 0.005:
            res.factors.append(Factor(
                "Calibration", 1.0,
                f"raw rating {res.probability:.1%} adjusted to {p:.1%}: in backtests, legs rated like this hit "
                f"{'more' if p > res.probability else 'less'} often than the raw number says", "info"))
            res.summary += f" Calibrated to {p:.0%} from past results."
        res.probability = p
    return res


def rate(con: duckdb.DuckDBPyConnection, raws: list[str], week: int | None = None,
         parlay_odds: int | None = None, log_source: str | None = None) -> tuple[list[LegResult], ParlayResult]:
    calib = Calibration.load()
    probcal = ProbCalibration.load()
    legs = [rate_leg(con, r, calib, week, probcal=probcal) for r in raws]
    result = parlay.combine(legs, calib, parlay_odds)
    if log_source:
        from cajetan import predlog

        predlog.record(legs, result, log_source, parlay_odds)
    return legs, result
