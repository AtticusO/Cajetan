"""Player prop evaluation: baseline -> matchup factors -> ML blend -> probability.

A projection is built as  volume x efficiency  (e.g. carries x yards per carry):

* volume   recency-weighted per-game usage, adjusted for game script (market spread
           and total) and for how much volume the opponent concedes to the position.
* efficiency  the player's per-play rate shrunk toward the positional average, then
           adjusted by:
             - the opponent's rate allowed to that position,
             - scheme/personnel matchups (the player's usage across run direction,
               formation, personnel, play action, motion, coverage, blitz, box count
               against what the opponent allows in each of those buckets),
             - situational defense (late game, running QBs, red zone for TD markets).
"""

from __future__ import annotations

import math

import duckdb

from cajetan import config, weather
from cajetan.analysis import ml
from cajetan.analysis.distributions import (count_median, count_outcome, empirical_median, empirical_outcome,
                                            gamma_median, gamma_outcome)
from cajetan.analysis.environment import games_played, predict_volume, script_model, team_scoring
from cajetan.analysis.explain import Factor, LegResult
from cajetan.analysis.markets import DIM_LABELS, FAMILIES, MARKETS, Market, bucket_label, plural
from cajetan.analysis.matchup import PlayStats, Split, fmt_metric, ordinal, shrink, weighted_log, wmean
from cajetan.calibration import Calibration
from cajetan.parsing import Leg, decimal_to_american
from cajetan.resolve import GameContext, PlayerRef, current_season, find_game, find_player, player_team_for_week

DEF_K = {"rush": 120.0, "target": 100.0, "dropback": 250.0}
SCHEME_WEIGHT = 0.6  # partial pooling of the (overlapping) scheme dimensions
SCHEME_CAP = math.log(1.18)
EFF_CAP = (0.75, 1.30)
VOL_CAP = (0.80, 1.25)
VACANCY_CAP = 1.8
UNIT_PLURAL = {"carries": "carries", "targets": "targets", "attempts": "attempts"}
# Team stat whose weather effect applies to each market (see calibration._weather).
WEATHER_STAT = {"pass_yds": "pass_yds", "rec_yds": "pass_yds", "pass_comp": "pass_comp", "rec": "pass_comp",
                "rush_yds": "rush_yds", "pass_tds": "points", "anytime_td": "points"}
# Market whose historical injury-designation effect applies to each market.
INJURY_MARKET = {"pass_yds": "pass_yds", "pass_att": "pass_att", "pass_comp": "pass_comp", "pass_tds": "pass_yds",
                 "rush_yds": "rush_yds", "rush_att": "rush_att", "rec": "rec", "rec_yds": "rec_yds",
                 "rush_rec_yds": "rush_rec_yds", "anytime_td": "rush_rec_yds"}


def cap_factors(factors: list[Factor], lo: float, hi: float) -> None:
    """Scale factors (in log space) so their product stays inside [lo, hi]."""
    total = sum(math.log(f.multiplier) for f in factors)
    bound = math.log(hi) if total > 0 else math.log(lo)
    if total and abs(total) > abs(bound):
        scale = bound / total
        for f in factors:
            f.multiplier = math.exp(math.log(f.multiplier) * scale)


def product(factors: list[Factor]) -> float:
    out = 1.0
    for f in factors:
        out *= f.multiplier
    return out


class PlayerEvaluator:
    def __init__(self, con: duckdb.DuckDBPyConnection, player: PlayerRef, game: GameContext, calib: Calibration):
        self.con, self.player, self.game, self.calib = con, player, game, calib
        self.asof = game.asof
        self.opp = game.opponent
        self.pos = "RB" if player.position == "FB" else player.position
        self.ps = PlayStats(con, self.asof)
        self.log = weighted_log(con, player.player_id, self.asof)
        self.dims_tried = 0
        self.dims_current = 0
        self.notes: list[str] = []
        name_parts = [w for w in player.name.split() if w.rstrip(".").upper() not in ("JR", "SR", "II", "III", "IV", "V")]
        self.short = name_parts[-1] if name_parts else player.name

    # ------------------------------------------------------------------ volume
    def volume(self, col: str) -> tuple[float, list[Factor]]:
        base = wmean(self.log, col)
        factors: list[Factor] = []
        g = self.game
        model = script_model(self.con, self.asof)
        sample = self.log.dropna(subset=["exp_margin", "total_line"])
        if model and g.exp_margin is not None and g.total_line is not None and len(sample):
            team_cols = ["pass_att"] if col in ("attempts", "targets") else (
                ["carries", "pass_att"] if self.pos == "QB" else ["carries"])
            ratios = []
            for tc in team_cols:
                usual = wmean(sample, sample.apply(
                    lambda r: predict_volume(model[tc], r.exp_margin, r.total_line), axis=1))
                now = predict_volume(model[tc], g.exp_margin, g.total_line)
                ratios.append((now / usual, now, usual))
            mult = math.prod(r[0] for r in ratios) ** (1 / len(ratios))
            _, now, usual = ratios[0]
            fav = (f"favored by {g.exp_margin:g}" if g.exp_margin > 0 else
                   f"a {-g.exp_margin:g}-pt underdog" if g.exp_margin < 0 else "a pick'em")
            what = "team pass attempts" if team_cols[0] == "pass_att" else "team carries"
            factors.append(Factor(
                "Game script", mult,
                f"{g.team} is {fav} (total {g.total_line:g}); spread/total model expects {now:.1f} {what} "
                f"vs {usual:.1f} in his typical game environment", "volume"))
        dv = self.defense_volume(col)
        if dv:
            factors.append(dv)
        cap_factors(factors, *VOL_CAP)
        vac = self.vacated(col)
        if vac:
            factors.append(vac)  # outside the cap: an injured starter can legitimately double a backup's role
        return base, factors

    def vacated(self, col: str) -> Factor | None:
        """Targets or carries freed up by teammates ruled Out/Doubtful who played the previous game."""
        if col not in ("targets", "carries"):
            return None
        share = "tgt_share" if col == "targets" else "carry_share"
        g = self.game
        rows = self.con.execute(
            f"""SELECT name, status, position, {share} AS s FROM vacancy
                WHERE team = ? AND season = ? AND week = ? AND player_id <> ? AND {share} >= 0.03
                ORDER BY s DESC""",
            [g.team, g.season, g.week, self.player.player_id],
        ).fetchall()
        vacated = sum(r[3] for r in rows)
        if vacated < 0.03:
            return None
        kind = "tgt" if col == "targets" else "carry"
        k = self.calib.vacancy(kind)
        even_split = 1 / (1 - min(vacated, 0.8))
        mult = min(VACANCY_CAP, 1 + k * (even_split - 1))
        who = ", ".join(f"{n} ({p}, {st}, {s:.0%})" for n, st, p, s in rows[:4])
        return Factor(
            "Teammates out", mult,
            f"{who} vacate {vacated:.0%} of {g.team}'s {col}; historically players absorb {k:.0%} of an even "
            f"redistribution, so {self.short}'s {col} rise {(mult - 1) * 100:+.0f}%", "volume")

    def injury(self, market: str) -> Factor | None:
        """The player's own designation on this week's injury report."""
        g = self.game
        row = self.con.execute(
            "SELECT status, practice, injury FROM injury_status WHERE player_id = ? AND season = ? AND week = ?",
            [self.player.player_id, g.season, g.week],
        ).fetchone()
        reported = self.con.execute(
            "SELECT count(*) FROM injury_status WHERE team = ? AND season = ? AND week = ?",
            [g.team, g.season, g.week],
        ).fetchone()[0]
        if not reported:
            unplayed = self.con.execute("SELECT home_score IS NULL FROM schedules WHERE game_id = ?",
                                        [g.game_id]).fetchone()
            if unplayed and unplayed[0]:
                self.notes.append(f"{g.team}'s week {g.week} injury report isn't loaded yet; run `cajetan update` "
                                  "after it's published (Wednesday-Friday) to account for injuries.")
            return None
        if not row or not row[0]:
            return None
        status, practice, what = row
        what = f" ({what})" if what else ""
        if status == "Out":
            self.notes.append(f"{self.player.name} is listed OUT{what}. Sportsbooks usually void props on inactive players.")
            return None
        if status == "Doubtful":
            self.notes.append(f"{self.player.name} is listed DOUBTFUL{what} and may not play.")
        eff = self.calib.injury(INJURY_MARKET.get(market, market), status)
        practiced = {2: "did not practice", 1: "was limited in practice", 0: "practiced fully"}.get(practice, "")
        if not eff:
            return Factor("Injury report", 1.0, f"listed {status}{what}{'; ' + practiced if practiced else ''}", "info")
        mult, n = eff
        return Factor(
            "Injury report", mult,
            f"listed {status}{what}{'; ' + practiced if practiced else ''}; in {n:,} games where a player carried that "
            f"designation and played, he produced {(mult - 1) * 100:+.0f}% vs his season average", "projection")

    def weather_factor(self, market: str, cond, mix: dict[str, float] | None = None) -> Factor | None:
        """Kickoff conditions vs the average game; `mix` blends stats for combined markets."""
        if cond is None:
            return None
        stats = mix or ({WEATHER_STAT[market]: 1.0} if market in WEATHER_STAT else {})
        mult, details = 0.0, []
        for stat, share in stats.items():
            res = self.calib.weather(stat, cond)
            m, d = res if res else (1.0, "")
            mult += share * m
            if d and d not in details:
                details.append(d)
        if not stats or abs(mult - 1) < 0.005:
            return None
        return Factor("Weather", mult, f"{cond.describe()}: {'; '.join(details)}", "projection")

    def defense_volume(self, col: str) -> Factor | None:
        df = self.con.execute(
            f"""
            WITH g AS (
                SELECT season, week, opponent_team AS defteam, sum(COALESCE({col}, 0)) AS v
                FROM player_week
                WHERE {self.asof.clause()} AND season_type IN ('REG', 'POST') AND position = ?
                GROUP BY ALL
            )
            SELECT defteam, sum({self.asof.weight()}) AS n, sum({self.asof.weight()} * v) AS val
            FROM g GROUP BY 1
            """,
            [self.pos],
        ).df()
        if self.opp not in set(df["defteam"]):
            return None
        lg = df["val"].sum() / df["n"].sum()
        df["shrunk"] = [shrink(v, n, lg, 3.0) for v, n in zip(df["val"], df["n"])]
        df["rank"] = df["shrunk"].rank(method="min").astype(int)
        row = df.set_index("defteam").loc[self.opp]
        idx = row["shrunk"] / lg
        return Factor(
            "Volume allowed", 1 + 0.5 * (idx - 1),
            f"{self.opp} concedes {row['val'] / row['n']:.1f} {self.pos} {UNIT_PLURAL[col]}/game "
            f"(league {lg:.1f}; {ordinal(int(row['rank']))}-fewest of {len(df)})", "volume")

    # -------------------------------------------------------------- efficiency
    def efficiency(self, m: Market) -> tuple[float, list[Factor], Factor]:
        fam, metric = m.family, m.metric
        pos = None if fam == "dropback" else self.pos
        lg = self.ps.total(fam, metric, pos=pos)
        me = self.ps.total(fam, metric, player_id=self.player.player_id)
        k = 40.0 if (fam == "rush" and self.pos == "QB") else m.k
        base = shrink(me.val, me.n, lg.rate, k)
        unit = FAMILIES[fam]["unit"]
        baseline_note = (f"{fmt_metric(metric, me.rate) if me.n else 'n/a'} {m.metric_label} on "
                         f"{me.n_cur:.0f} {plural(unit)} this season (prior season down-weighted); shrunk to "
                         f"{fmt_metric(metric, base)} toward the {self.pos if pos else 'QB'} average "
                         f"{fmt_metric(metric, lg.rate)}")
        if me.n < 25:
            self.notes.append(f"Small sample: only {me.n:.0f} weighted {plural(unit)} for {self.player.name}.")
        factors: list[Factor] = []

        # 1. Opponent vs the position
        table = self.ps.defense_table(fam, metric, pos, DEF_K[fam])
        def_idx = 1.0
        if self.opp in table.index:
            row = table.loc[self.opp]
            def_idx = row["shrunk"] / table.attrs["league"]
            who = f"{self.pos}s" if pos else "QBs"
            factors.append(Factor(
                "Defense vs position", def_idx,
                f"{self.opp} allows {fmt_metric(metric, row['rate'])} {m.metric_label} to {who} "
                f"(league {fmt_metric(metric, table.attrs['league'])}; {ordinal(int(row['rank']))}-stingiest "
                f"of {len(table)}, {row['n_cur']:.0f} plays this season)"))

        # 2. Scheme & personnel matchups
        dims = list(m.offense_dims)
        if fam == "rush" and self.pos == "QB":
            dims = ["qb_run_type"] + [d for d in dims if d != "personnel"]
        scheme: list[tuple[str, float, str]] = []
        for dim in dims:
            res = self.offense_dim(m, dim, pos, def_idx, me)
            if res:
                scheme.append(res)
        for dim in m.defense_dims:
            res = self.defense_dim(m, dim, pos, me)
            if res:
                scheme.append(res)
        scheme_factors = [Factor(f"Scheme: {DIM_LABELS[d]}", math.exp(SCHEME_WEIGHT * math.log(mult)), detail)
                          for d, mult, detail in scheme]
        cap_factors(scheme_factors, math.exp(-SCHEME_CAP), math.exp(SCHEME_CAP))
        factors += scheme_factors

        # 3. Situational: late game
        late = self.late_game(m, pos, def_idx, me)
        if late:
            factors.append(late)

        # 4. Situational: running quarterbacks
        if fam == "rush" and self.pos == "QB":
            rq = self.running_qb("rushing_yards")
            if rq:
                factors.append(rq)

        cap_factors(factors, *EFF_CAP)
        return base, factors, Factor("Baseline efficiency", 1.0, baseline_note, "info")

    def _splits(self, m: Market, dim: str, pos: str | None) -> tuple[dict[str, Split], dict[str, Split], dict[str, Split]]:
        fam, metric = m.family, m.metric
        P = self.ps.buckets(fam, metric, dim, player_id=self.player.player_id)
        D = self.ps.buckets(fam, metric, dim, pos=pos, extra=f"defteam = '{self.opp}'")
        L = self.ps.buckets(fam, metric, dim, pos=pos)
        return P, D, L

    def _stale_note(self, P: dict[str, Split], D: dict[str, Split]) -> bool:
        return sum(s.n_cur for s in P.values()) == 0 or sum(s.n_cur for s in D.values()) == 0

    def offense_dim(self, m: Market, dim: str, pos: str | None, def_idx: float, me: Split):
        """Player's usage across an offense-chosen bucket vs what the opponent allows in each bucket."""
        P, D, L = self._splits(m, dim, pos)
        nP, nD = sum(s.n for s in P.values()), sum(s.n for s in D.values())
        self.dims_tried += 1
        if nP < 8 or nD < 20:
            return None
        lg_all = sum(s.val for s in L.values()) / sum(s.n for s in L.values())
        kdef = DEF_K[m.family]
        def_all = shrink(sum(s.val for s in D.values()), nD, lg_all, kdef) / lg_all
        r_p = sum(s.val for s in P.values()) / nP
        rows = []
        for b, sp in P.items():
            lb = L.get(b)
            if lb is None or lb.n < 30 or lb.rate <= 0:
                continue
            eff_b = shrink(sp.val, sp.n, r_p, m.k / 2)
            d = D.get(b, Split())
            idx_b = shrink(d.val, d.n, lb.rate, kdef / 2) / lb.rate
            rows.append((b, sp.n / nP, eff_b, idx_b / def_all, d, lb))
        prod_total = sum(share * eff for _, share, eff, *_ in rows)
        if not rows or prod_total <= 0:
            return None
        mult = sum(share * eff / prod_total * rel for _, share, eff, rel, *_ in rows)
        b, share, eff, rel, d, lb = max(rows, key=lambda r: abs(r[1] * r[2] / prod_total * math.log(r[3])))
        unit = FAMILIES[m.family]["unit"]
        allows = fmt_metric(m.metric, d.rate) if d.n else "n/a"
        detail = (f"{share:.0%} of {self.short}'s {plural(unit)} come from {bucket_label(dim, b)}; {self.opp} allows "
                  f"{allows} {m.metric_label} vs those ({d.n_cur:.0f} this season), league {fmt_metric(m.metric, lb.rate)}, "
                  f"{(rel - 1) * 100:+.0f}% relative to {self.opp}'s overall level")
        if self._stale_note(P, D):
            mult = 1 + (mult - 1) * 0.5
            detail += f" [uses {self.asof.season - 1} data: {self.asof.season} {DIM_LABELS[dim]} charting not published yet]"
        else:
            self.dims_current += 1
        return dim, mult, detail

    def defense_dim(self, m: Market, dim: str, pos: str | None, me: Split):
        """Opponent's tendency in a defense-chosen bucket (coverage, blitz, box) vs how the player fares there."""
        P, D, L = self._splits(m, dim, pos)
        nP, nD, nL = (sum(s.n for s in x.values()) for x in (P, D, L))
        self.dims_tried += 1
        if nP < 15 or nD < 20 or nL == 0:
            return None
        lg_all = sum(s.val for s in L.values()) / nL
        r_p = sum(s.val for s in P.values()) / nP
        rows = []
        for b, lb in L.items():
            if lb.n < 30:
                continue
            sp = P.get(b, Split())
            prior = r_p * (lb.rate / lg_all) if lg_all else r_p
            eff_b = shrink(sp.val, sp.n, prior, m.k / 2)
            fd = (D.get(b, Split()).n + 40 * lb.n / nL) / (nD + 40)
            fp = sp.n / nP
            rows.append((b, fd, fp, eff_b, sp))
        usual = sum(fp * eff for _, _, fp, eff, _ in rows)
        expected = sum(fd * eff for _, fd, _, eff, _ in rows)
        if usual <= 0:
            return None
        mult = expected / usual
        b, fd, fp, eff, sp = max(rows, key=lambda r: abs((r[1] - r[2]) * (r[3] - r_p)))
        detail = (f"{self.opp} shows {bucket_label(dim, b)} on {fd:.0%} of snaps vs the {fp:.0%} {self.short} usually "
                  f"sees; he averages {fmt_metric(m.metric, eff)} {m.metric_label} against it "
                  f"({fmt_metric(m.metric, r_p)} overall)")
        if self._stale_note(P, D):
            mult = 1 + (mult - 1) * 0.5
            detail += f" [uses {self.asof.season - 1} data: {self.asof.season} {DIM_LABELS[dim]} charting not published yet]"
        else:
            self.dims_current += 1
        return dim, mult, detail

    def late_game(self, m: Market, pos: str | None, def_idx: float, me: Split) -> Factor | None:
        fam, metric = m.family, m.metric
        pl_late = self.ps.total(fam, metric, player_id=self.player.player_id, extra="late")
        d_late = self.ps.total(fam, metric, pos=pos, extra=f"late AND defteam = '{self.opp}'")
        l_late = self.ps.total(fam, metric, pos=pos, extra="late")
        if me.n < 10 or d_late.n < 10 or not l_late.rate:
            return None
        share = pl_late.n / me.n
        margin = self.game.exp_margin or 0
        script = ""
        if (fam == "rush" and margin >= 3) or (fam != "rush" and margin <= -3):
            share = min(1.0, share * 1.25)
            script = f"; game script points to more late {plural(FAMILIES[fam]['unit'])}"
        idx_late = shrink(d_late.val, d_late.n, l_late.rate, DEF_K[fam] / 2) / l_late.rate
        rel = idx_late / def_idx
        return Factor(
            "Late-game defense", (1 - share) + share * rel,
            f"{self.opp} allows {fmt_metric(metric, d_late.rate)} {m.metric_label} in the 4th quarter/OT "
            f"(league {fmt_metric(metric, l_late.rate)}); {share:.0%} of {self.short}'s volume comes late{script}")

    def running_qb(self, col: str) -> Factor | None:
        """How the opponent has handled QBs who run, relative to those QBs' own averages.

        `col` is rushing_yards or carries. A running QB averages 25+ rush yards per game.
        """
        unit = "rush yds" if col == "rushing_yards" else "carries"
        df = self.con.execute(
            f"""
            WITH qb AS (
                SELECT player_id, season, avg({col}) AS avg_v, avg(rushing_yards) AS avg_ry, count(*) AS g
                FROM player_week
                WHERE position = 'QB' AND {self.asof.clause()} AND attempts >= 10 AND season_type IN ('REG', 'POST')
                GROUP BY 1, 2
            )
            SELECT w.opponent_team AS defteam, count(*) AS games,
                   sum(CASE WHEN w.season = {self.asof.season} THEN 1 ELSE 0 END) AS games_cur,
                   sum({self.asof.weight('w')} * w.{col}) AS allowed,
                   sum({self.asof.weight('w')} * q.avg_v) AS expected,
                   sum({self.asof.weight('w')}) AS n
            FROM player_week w JOIN qb q USING (player_id, season)
            WHERE w.position = 'QB' AND {self.asof.clause('w')} AND w.attempts >= 10
              AND q.avg_ry >= 25 AND q.g >= 2
            GROUP BY 1
            """
        ).df()
        own = wmean(self.log, "rushing_yards") if len(self.log) else 0.0
        if self.opp not in set(df["defteam"]):
            return Factor("Vs running QBs", 1.0,
                          f"{self.opp} hasn't faced a running QB (25+ rush yds/g) in the sample", "info")
        K = 3 * (df["expected"].sum() / df["n"].sum())  # ~3 games of a typical running QB
        df["ratio"] = (df["allowed"] + K) / (df["expected"] + K)
        df["rank"] = df["ratio"].rank(method="min").astype(int)
        r = df.set_index("defteam").loc[self.opp]
        runner = "is" if own >= 25 else "is not"
        return Factor(
            "Vs running QBs", float(r["ratio"]),
            f"in {int(r['games'])} games vs running QBs ({int(r['games_cur'])} this season), {self.opp} allowed "
            f"{(r['allowed'] / r['n']):.1f} {unit}/g, {(r['ratio'] - 1) * 100:+.0f}% vs those QBs' norms "
            f"({ordinal(int(r['rank']))}-stingiest of {len(df)}); {self.short} {runner} a running QB "
            f"({own:.1f} rush yds/g)", "projection")

    # -------------------------------------------------------------- markets
    def project(self, m: Market) -> tuple[float, list[Factor]]:
        vol, vf = self.volume(m.volume)
        info = Factor("Baseline volume", 1.0,
                      f"{vol:.1f} {UNIT_PLURAL[m.volume]}/game (recency-weighted over {len(self.log)} games, "
                      f"{self.games_current} this season)", "info")
        if m.family is None:
            factors = vf
            if m.key == "rush_att" and self.pos == "QB":
                rq = self.running_qb("carries")
                if rq and rq.applies_to != "info":
                    rq.multiplier = 1 + 0.5 * (rq.multiplier - 1)
                factors = factors + ([rq] if rq else [])
            return vol * product(factors), factors + [info]
        eff, ef, eff_info = self.efficiency(m)
        factors = vf + ef
        mean = vol * eff * product(factors)
        if m.key == "pass_tds":
            extra = self.scoring_factors("dropback", "pass_touchdown", None)
            factors += extra
            mean *= product(extra)
        return mean, factors + [info, eff_info]

    def scoring_factors(self, fam: str, td_col: str, pos: str | None) -> list[Factor]:
        out = []
        table = self.ps.defense_table(fam, td_col, pos, 120.0, extra="redzone")
        if self.opp in table.index:
            row = table.loc[self.opp]
            lg = table.attrs["league"]
            who = f"{pos}s" if pos else "passes"
            out.append(Factor(
                "Red-zone defense", min(1.15, max(0.85, row["shrunk"] / lg)),
                f"{self.opp} allows a TD on {row['rate']:.1%} of red-zone {plural(FAMILIES[fam]['unit'])} to {who} "
                f"(league {lg:.1%}; {ordinal(int(row['rank']))}-stingiest)", "projection"))
        avg, _ = team_scoring(self.con, self.game.team, self.asof)
        if self.game.implied_points and avg == avg and avg > 0:
            m = min(1.25, max(0.8, self.game.implied_points / avg))
            out.append(Factor(
                "Scoring environment", m,
                f"market implies {self.game.implied_points:.1f} pts for {self.game.team} vs its "
                f"{avg:.1f} pts/game average", "projection"))
        return out

    def td_projection(self) -> tuple[float, list[Factor]]:
        """Expected touchdowns = sum over rush/receiving of volume x (red-zone share x RZ TD rate + rest)."""
        comps = [("rush", "carries", "rush_touchdown")]
        if self.pos != "QB":
            comps.append(("target", "targets", "pass_touchdown"))
        parts: list[tuple[float, list[Factor], str]] = []
        for fam, volcol, td in comps:
            vol, vf = self.volume(volcol)
            pid = self.player.player_id
            me = self.ps.total(fam, "1", player_id=pid)
            me_rz = self.ps.total(fam, td, player_id=pid, extra="redzone")
            me_nrz = self.ps.total(fam, td, player_id=pid, extra="NOT redzone")
            lg = self.ps.total(fam, td, pos=self.pos)
            lg_rz = self.ps.total(fam, td, pos=self.pos, extra="redzone")
            lg_nrz = self.ps.total(fam, td, pos=self.pos, extra="NOT redzone")
            if not lg.n:
                continue
            rz_share = shrink(me_rz.n, me.n, lg_rz.n / lg.n, 60)
            td_rz = shrink(me_rz.val, me_rz.n, lg_rz.rate, 40)
            td_nrz = shrink(me_nrz.val, me_nrz.n, lg_nrz.rate, 80)
            lam = vol * (rz_share * td_rz + (1 - rz_share) * td_nrz)
            unit = FAMILIES[fam]["unit"]
            info = Factor(f"{unit.title()} TD baseline", 1.0,
                          f"{vol:.1f} {plural(unit)}/g, {rz_share:.0%} in the red zone, TD rate {td_rz:.1%} inside the 20 "
                          f"and {td_nrz:.1%} outside -> {lam:.2f} TDs/g before adjustments", "info")
            rz = self.scoring_factors(fam, td, self.pos)[:1]
            parts.append((lam, vf + rz + [info], unit))
        total = sum(p[0] * product(p[1]) for p in parts)
        factors: list[Factor] = []
        for lam, fs, unit in parts:
            share = lam * product(fs) / total if total else 0
            for f in fs:
                if f.applies_to == "info":
                    factors.append(f)
                else:
                    factors.append(Factor(f"{f.name} ({plural(unit)})", 1 + share * (f.multiplier - 1), f.detail, f.applies_to))
        env = [f for f in self.scoring_factors("rush", "rush_touchdown", self.pos) if f.name == "Scoring environment"]
        factors += env
        return total * product(env), factors

    @property
    def games_current(self) -> int:
        return int((self.log["season"] == self.asof.season).sum()) if len(self.log) else 0

    def reliability(self, has_ml: bool) -> int:
        g_cur = self.games_current
        g_prior = len(self.log) - g_cur
        opp_games = games_played(self.con, self.opp, self.asof)
        score = 35 * min(g_cur, 6) / 6 + 10 * min(g_prior, 8) / 8 + 20 * min(opp_games, 6) / 6
        if self.dims_tried:
            score += 15 * self.dims_current / self.dims_tried
        else:
            score += 10
        score += 10 if has_ml else 0
        score += 10 if self.calib.loaded else 0
        return int(round(min(100, score)))


def ml_weight(market: str, meta: dict) -> float:
    """Share of the projection given to the ML model: fitted per market by `cajetan fit` when available."""
    if market in config.ML_BLEND_BY_MARKET:
        return float(config.ML_BLEND_BY_MARKET[market])
    beats_naive = meta["mae"] <= meta["naive_mae"]
    return config.ML_BLEND_WEIGHT if beats_naive else config.ML_BLEND_WEIGHT_WEAK


def outcome(calib: Calibration, market: str, pos: str, mean: float, line: float, side: str
            ) -> tuple[float, float, float | None]:
    """(probability the side wins with pushes removed, push probability, median outcome)."""
    m = MARKETS[market]
    if market == "anytime_td":
        return 1 - math.exp(-mean), 0.0, None
    if m.dist == "gamma" and (q := calib.ratio_quantiles(market, pos, mean)):
        out = empirical_outcome(mean, q, line)
        median = round(empirical_median(mean, q), 1)
    elif m.dist == "gamma":
        cv = calib.cv(market, pos, mean)
        out = gamma_outcome(mean, cv, line)
        median = round(gamma_median(mean, cv), 1)
    else:
        r = calib.nb_size(market, pos)
        out = count_outcome(mean, line, r)
        median = count_median(mean, r)
    return out.p_side(side), out.p_push, median


def evaluate(con: duckdb.DuckDBPyConnection, leg: Leg, calib: Calibration, week: int | None = None,
             season: int | None = None) -> LegResult:
    m = MARKETS[leg.market]
    season = season or current_season(con)
    player = find_player(con, leg.subject, season, m.positions)
    team = player_team_for_week(con, player, season, week)
    game = find_game(con, team, season, week)
    ev = PlayerEvaluator(con, player, game, calib)
    if ev.log.empty:
        raise LookupError(f"{player.name} has no games in {season - 1}-{season} before week {game.week}")

    mix = None
    if m.key == "anytime_td":
        mean, factors = ev.td_projection()
    elif m.key == "rush_rec_yds":
        parts = [ev.project(MARKETS["rush_yds"]), ev.project(MARKETS["rec_yds"])]
        mean = sum(p[0] for p in parts)
        factors = []
        for (part_mean, fs), label in zip(parts, ("Rushing", "Receiving")):
            share = part_mean / mean if mean else 0
            for f in fs:
                if f.applies_to == "info":
                    factors.append(Factor(f"{label}: {f.name.lower()}", 1.0, f.detail, "info"))
                elif share >= 0.1:
                    factors.append(Factor(f"{label}: {f.name}", 1 + share * (f.multiplier - 1), f.detail, f.applies_to))
        if mean:
            mix = {"rush_yds": parts[0][0] / mean, "pass_yds": parts[1][0] / mean}
    else:
        mean, factors = ev.project(m)

    cond = weather.game_conditions(con, game.game_id)
    for extra in (ev.weather_factor(m.key, cond, mix), ev.injury(m.key)):
        if extra:
            factors.append(extra)
            mean *= extra.multiplier

    structural = mean
    baseline = structural / product([f for f in factors if f.applies_to != "info"])
    pred = ml.predict(con, m.key, player, game, cond)
    components = {"structural": structural, "ml": None, "ml_weight": 0.0}
    if pred is not None:
        mu_ml, meta = pred
        w = ml_weight(m.key, meta)
        components.update(ml=mu_ml, ml_weight=w)
        mean = (1 - w) * structural + w * mu_ml
        how = ("weight fitted on the backtest" if m.key in config.ML_BLEND_BY_MARKET else
               f"validated MAE {meta['mae']:.2f} vs {meta['naive_mae']:.2f} for a season-average guess")
        factors.append(Factor(
            "Historical ML model", mean / structural if structural else 1.0,
            f"gradient-boosted model on {meta['rows']:,} player-games ({meta['seasons']}) projects {mu_ml:.2f}; "
            f"blended at {w:.0%} ({how})", "projection"))
    else:
        ev.notes.append("No trained ML model for this market (run `cajetan train`); matchup engine only.")

    p, p_push, median = outcome(calib, m.key, ev.pos, mean, leg.line, leg.side)
    if m.key == "anytime_td":
        summary = (f"Expected {mean:.2f} TDs -> {p:.0%} to score "
                   f"(fair odds {decimal_to_american(1 / max(p, 1e-6)):+d}).")
    else:
        summary = (f"Projects {mean:.1f} {m.label} on average (median {median:g}) vs a line of {leg.line:g} "
                   f"-> {p:.0%} to go {leg.side}.")

    return LegResult(
        leg=leg, description=f"{player.name} ({ev.pos}, {game.label})", market_label=m.label,
        side=leg.side, line=leg.line, projection=mean, probability=p, p_push=p_push, baseline=baseline,
        median=median, reliability=ev.reliability(pred is not None), factors=factors, notes=ev.notes,
        summary=summary, market=m.key, game_id=game.game_id, team=game.team, opponent=game.opponent,
        position=ev.pos, player_id=player.player_id, season=game.season, week=game.week, components=components,
    )
