"""Team markets: moneyline, spread, game total and team total.

Projected points come from opponent-adjusted ridge "power ratings" (offense and
defense per team, plus home field) fit on scores, blended with an EPA-per-play
margin estimate, then blended with the market consensus line from the schedule.
Red-zone and late-game splits are reported as context.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np

from cajetan import config, weather
from cajetan.analysis.distributions import normal_outcome
from cajetan.analysis.environment import games_played
from cajetan.analysis.explain import Factor, LegResult
from cajetan.analysis.matchup import ordinal, shrink
from cajetan.calibration import Calibration
from cajetan.db import AsOf
from cajetan.parsing import Leg
from cajetan.resolve import GameContext, ResolveError, current_season, find_game

RIDGE_LAMBDA = 4.0
PLAYS_PER_GAME = 62.0


@dataclass
class Ratings:
    mu: float
    hfa: float
    off: dict[str, float]
    deff: dict[str, float]  # larger = better defense (points prevented)

    def points(self, team: str, opp: str, home: bool) -> float:
        return self.mu + self.off.get(team, 0) - self.deff.get(opp, 0) + (self.hfa / 2 if home else -self.hfa / 2)

    def rank(self, table: str, team: str) -> int:
        vals = getattr(self, table)
        return 1 + sum(v > vals.get(team, 0) for v in vals.values())


def power_ratings(con: duckdb.DuckDBPyConnection, asof: AsOf) -> Ratings:
    df = con.execute(
        f"""
        SELECT season, team, opponent, is_home, points, game_type FROM team_games
        WHERE points IS NOT NULL AND {asof.clause()} AND game_type <> 'PRE'
        """
    ).df()
    teams = sorted(set(df["team"]) | set(df["opponent"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    X = np.zeros((len(df), 2 + 2 * n))
    X[:, 0] = 1
    X[:, 1] = np.where(df["is_home"], 0.5, -0.5)  # neutral-site games are rare enough to ignore
    for r, (t, o) in enumerate(zip(df["team"], df["opponent"])):
        X[r, 2 + idx[t]] = 1
        X[r, 2 + n + idx[o]] = -1
    w = np.where(df["season"] == asof.season, 1.0, config.PRIOR_SEASON_WEIGHT)
    penalty = np.r_[0, 0, np.full(2 * n, RIDGE_LAMBDA)]
    A = X.T @ (X * w[:, None]) + np.diag(penalty)
    beta = np.linalg.solve(A, X.T @ (w * df["points"].to_numpy()))
    return Ratings(beta[0], beta[1], {t: beta[2 + i] for t, i in idx.items()},
                   {t: beta[2 + n + i] for t, i in idx.items()})


def epa_table(con: duckdb.DuckDBPyConnection, asof: AsOf) -> tuple[dict[str, float], dict[str, float]]:
    """Shrunk offensive EPA/play and defensive EPA/play allowed."""
    out = []
    for side in ("posteam", "defteam"):
        df = con.execute(
            f"""SELECT {side} AS team, sum({asof.weight()}) AS n, sum({asof.weight()} * epa) AS val
                FROM plays WHERE {asof.clause()} AND epa IS NOT NULL GROUP BY 1"""
        ).df()
        lg = df["val"].sum() / df["n"].sum()
        out.append({r.team: shrink(r.val, r.n, lg, 250) - lg for r in df.itertuples()})
    return out[0], out[1]


def situational(con: duckdb.DuckDBPyConnection, asof: AsOf) -> dict[str, dict]:
    """Red-zone TD rate per trip (offense and defense) and 4th-quarter EPA/play allowed."""
    rz = con.execute(
        f"""
        WITH drives AS (
            SELECT season, posteam, defteam, game_id, fixed_drive,
                   max(CASE WHEN redzone THEN 1 ELSE 0 END) AS rz,
                   max(pass_touchdown + rush_touchdown) AS td
            FROM plays WHERE {asof.clause()} GROUP BY ALL
        )
        SELECT 'off' AS side, posteam AS team, sum(rz) AS trips, sum(CASE WHEN rz = 1 THEN td ELSE 0 END) AS tds
        FROM drives WHERE season = {asof.season} GROUP BY 2
        UNION ALL
        SELECT 'def', defteam, sum(rz), sum(CASE WHEN rz = 1 THEN td ELSE 0 END)
        FROM drives WHERE season = {asof.season} GROUP BY 2
        """
    ).df()
    late = con.execute(
        f"""SELECT defteam AS team, avg(epa) AS epa, count(*) AS n FROM plays
            WHERE late AND season = {asof.season} AND week < {asof.week} AND epa IS NOT NULL GROUP BY 1"""
    ).df()
    out: dict[str, dict] = {}
    for side in ("off", "def"):
        sub = rz[rz["side"] == side]
        lg = sub["tds"].sum() / max(sub["trips"].sum(), 1)
        sub = sub.assign(rate=[shrink(t, n, lg, 12) for t, n in zip(sub["tds"], sub["trips"])])
        sub = sub.assign(rank=sub["rate"].rank(ascending=(side == "def"), method="min"))
        for r in sub.itertuples():
            out.setdefault(r.team, {})[f"rz_{side}"] = (r.tds / max(r.trips, 1), int(r.rank), lg, int(r.trips))
    if len(late):
        late = late.assign(rank=late["epa"].rank(method="min"))
        for r in late.itertuples():
            out.setdefault(r.team, {})["late_def"] = (r.epa, int(r.rank), len(late))
    return out


def evaluate(con: duckdb.DuckDBPyConnection, leg: Leg, week: int | None = None, season: int | None = None,
             calib: Calibration | None = None) -> LegResult:
    season = season or current_season(con)
    game = find_game(con, leg.subject, season, week)
    if leg.market == "game_total" and leg.opponent and leg.opponent != game.opponent:
        raise ResolveError(f"{leg.subject}'s next game is against {game.opponent}, not {leg.opponent}")
    asof = game.asof
    r = power_ratings(con, asof)
    off_epa, def_epa = epa_table(con, asof)
    t, o = game.team, game.opponent
    pts_t = r.points(t, o, game.is_home)
    pts_o = r.points(o, t, not game.is_home)
    srs_margin = pts_t - pts_o
    epa_margin = ((off_epa.get(t, 0) + def_epa.get(o, 0)) - (off_epa.get(o, 0) + def_epa.get(t, 0))) * PLAYS_PER_GAME
    epa_margin += r.hfa if game.is_home else -r.hfa
    model_margin = 0.5 * srs_margin + 0.5 * epa_margin
    factors: list[Factor] = []
    # Weather moves scoring in the model's projection; the market line already prices the forecast.
    cond = weather.game_conditions(con, game.game_id)
    wx = (calib or Calibration.load()).weather("points", cond)
    if wx and abs(wx[0] - 1) >= 0.005:
        before = pts_t + pts_o
        pts_t, pts_o = pts_t * wx[0], pts_o * wx[0]
        if leg.market in ("game_total", "team_total"):
            shift = (pts_t + pts_o - before) if leg.market == "game_total" else pts_t * (1 - 1 / wx[0])
            factors.append(Factor("Weather", 1.0, f"{cond.describe()}: {wx[1]} (applied to the model's points)",
                                  "projection", points=shift))
    model_total = pts_t + pts_o
    sit = situational(con, asof)
    mk = leg.market
    components = {"model_margin": model_margin, "market_margin": game.exp_margin, "model_total": model_total,
                  "market_total": game.total_line, "model_team_pts": pts_t, "market_team_pts": game.implied_points}

    def market_blend(model_value: float, market_value: float | None, what: str) -> float:
        if market_value is None:
            return model_value
        final = (1 - config.MARKET_BLEND_WEIGHT) * model_value + config.MARKET_BLEND_WEIGHT * market_value
        factors.append(Factor("Market consensus", 1.0, f"consensus {what} {market_value:+.1f} vs model "
                              f"{model_value:+.1f}; blended {config.MARKET_BLEND_WEIGHT:.0%}", "projection",
                              points=final - model_value))
        return final

    if mk in ("ml", "spread"):
        hfa = r.hfa if game.is_home else -r.hfa
        factors += [
            Factor("Power ratings", 1.0,
                   f"opponent-adjusted scoring: {t} offense {ordinal(r.rank('off', t))}, defense "
                   f"{ordinal(r.rank('deff', t))}; {o} offense {ordinal(r.rank('off', o))}, defense "
                   f"{ordinal(r.rank('deff', o))} -> {srs_margin:+.1f} margin", "projection",
                   points=0.5 * (srs_margin - hfa)),
            Factor("EPA per play", 1.0,
                   f"net EPA/play {t} {off_epa.get(t, 0) - def_epa.get(t, 0):+.3f} vs {o} "
                   f"{off_epa.get(o, 0) - def_epa.get(o, 0):+.3f}", "projection",
                   points=0.5 * (epa_margin - hfa)),
            Factor("Home field", 1.0, f"{t} {'home' if game.is_home else 'away'} (league HFA {r.hfa:.1f})",
                   "projection", points=hfa),
        ]
        mean = market_blend(model_margin, game.exp_margin, "margin")
        if mk == "ml":
            out = normal_outcome(mean, config.MARGIN_SD, 0)
            p, push = out.p_over / (out.p_over + out.p_under), 0.0
            summary = f"Projected margin {t} {mean:+.1f} -> {p:.0%} to win outright."
        else:
            out = normal_outcome(mean, config.MARGIN_SD, -leg.line)
            p, push = out.p_side("over"), out.p_push
            summary = f"Projected margin {t} {mean:+.1f} vs spread {leg.line:+g} -> {p:.0%} to cover."
        proj = mean
    elif mk == "game_total":
        factors.append(Factor("Power ratings", 1.0, f"{t} {pts_t:.1f} + {o} {pts_o:.1f} projected points",
                              "projection", points=model_total - 2 * r.mu))
        mean = market_blend(model_total, game.total_line, "total")
        out = normal_outcome(mean, config.TOTAL_SD, leg.line)
        p, push, proj = out.p_side(leg.side), out.p_push, mean
        summary = f"Projected total {mean:.1f} vs {leg.line:g} -> {p:.0%} to go {leg.side}."
    else:  # team_total
        factors.append(Factor("Power ratings", 1.0,
                              f"{t} offense {ordinal(r.rank('off', t))} vs {o} defense {ordinal(r.rank('deff', o))}",
                              "projection", points=pts_t - r.mu))
        mean = market_blend(pts_t, game.implied_points, "implied team total")
        out = normal_outcome(mean, config.TEAM_POINTS_SD, leg.line)
        p, push, proj = out.p_side(leg.side), out.p_push, mean
        summary = f"Projected {t} points {mean:.1f} vs {leg.line:g} -> {p:.0%} to go {leg.side}."

    for team, opp in ((t, o), (o, t)):
        s_off, s_def = sit.get(team, {}).get("rz_off"), sit.get(opp, {}).get("rz_def")
        if s_off and s_def:
            factors.append(Factor(
                f"Red zone ({team} offense)", 1.0,
                f"{team} scores TDs on {s_off[0]:.0%} of red-zone trips ({ordinal(s_off[1])}); {opp} allows "
                f"{s_def[0]:.0%} ({ordinal(s_def[1])}-stingiest); league {s_off[2]:.0%}", "info"))
    for team in (t, o):
        ld = sit.get(team, {}).get("late_def")
        if ld:
            factors.append(Factor(f"Late-game defense ({team})", 1.0,
                                  f"{ld[0]:+.3f} EPA/play allowed in the 4th quarter/OT ({ordinal(ld[1])}-best of {ld[2]})",
                                  "info"))

    g_t, g_o = games_played(con, t, asof), games_played(con, o, asof)
    reliability = int(min(100, 30 * min(g_t, 6) / 6 + 30 * min(g_o, 6) / 6
                          + (25 if game.exp_margin is not None else 0) + 15))
    desc = f"{leg.describe()} ({game.label})"
    return LegResult(
        leg=leg, description=desc, market_label=leg.market_label, side=leg.side, line=leg.line,
        projection=proj, probability=p, p_push=push, reliability=reliability, factors=factors,
        summary=summary, market=mk, game_id=game.game_id, team=t, opponent=o, season=game.season, week=game.week,
        components=components,
    )
