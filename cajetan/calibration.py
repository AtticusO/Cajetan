"""Empirical calibration: stat dispersion (for over/under probabilities) and leg-to-leg correlations.

`compute()` derives everything from the local database during `cajetan train`;
`Calibration` loads the saved JSON and falls back to sensible NFL defaults.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import duckdb
import numpy as np

from cajetan import config

# Coefficient of variation for gamma-distributed yardage props.
DEFAULT_CV = {
    "pass_yds": 0.28, "rush_yds|RB": 0.50, "rush_yds|QB": 0.85, "rush_yds": 0.90,
    "rec_yds|WR": 0.62, "rec_yds|TE": 0.70, "rec_yds|RB": 0.85, "rec_yds": 0.70,
    "rush_rec_yds|RB": 0.45, "rush_rec_yds": 0.60,
}
# Negative-binomial size for count props (None = Poisson).
DEFAULT_NB = {"pass_att": 80.0, "pass_comp": 60.0, "pass_tds": None, "pass_int": None,
              "rush_att": 12.0, "rec": 15.0}

DEFAULT_PAIR = {
    "same|QB|pass_comp|QB|pass_yds": 0.80, "same|QB|pass_att|QB|pass_yds": 0.65,
    "same|QB|pass_tds|QB|pass_yds": 0.50, "same|RB|rush_att|RB|rush_yds": 0.75,
    "same|RB|rec|RB|rec_yds": 0.75, "same|WR|rec|WR|rec_yds": 0.75, "same|TE|rec|TE|rec_yds": 0.75,
    "same|RB|anytime_td|RB|rush_yds": 0.35, "same|WR|anytime_td|WR|rec_yds": 0.35,
    "teammate|QB|pass_yds|WR|rec_yds": 0.40, "teammate|QB|pass_yds|TE|rec_yds": 0.35,
    "teammate|QB|pass_yds|RB|rush_yds": -0.10, "teammate|QB|pass_tds|WR|anytime_td": 0.30,
    "teammate|RB|rush_yds|RB|rush_yds": -0.20, "teammate|WR|rec_yds|WR|rec_yds": 0.0,
    "opponent|QB|pass_yds|QB|pass_yds": 0.15,
}
DEFAULT_ENV = {
    "QB|pass_yds|margin": -0.10, "QB|pass_tds|margin": 0.20, "RB|rush_yds|margin": 0.25,
    "RB|anytime_td|margin": 0.15, "QB|pass_yds|total": 0.35, "QB|pass_tds|total": 0.45,
    "WR|rec_yds|total": 0.25, "RB|anytime_td|total": 0.30, "WR|anytime_td|total": 0.30,
    "QB|pass_yds|team_pts": 0.35, "QB|pass_tds|team_pts": 0.55, "RB|anytime_td|team_pts": 0.40,
}

CORR_MARKETS = {
    "pass_yds": "passing_yards", "pass_tds": "passing_tds", "pass_comp": "completions",
    "pass_att": "attempts", "rush_yds": "rushing_yards", "rush_att": "carries",
    "rec": "receptions", "rec_yds": "receiving_yards",
    "rush_rec_yds": "rushing_yards + receiving_yards",
    "anytime_td": "CASE WHEN rushing_tds + receiving_tds > 0 THEN 1 ELSE 0 END",
}
POS_MARKETS = {
    "QB": ["pass_yds", "pass_tds", "pass_comp", "pass_att", "rush_yds", "rush_att", "anytime_td"],
    "RB": ["rush_yds", "rush_att", "rec", "rec_yds", "rush_rec_yds", "anytime_td"],
    "WR": ["rec", "rec_yds", "rush_rec_yds", "anytime_td"],
    "TE": ["rec", "rec_yds", "anytime_td"],
}
# Empirical outcome shapes for yardage props: (stat, prior-volume column, minimum prior volume).
RATIO_MARKETS = {
    ("pass_yds", "QB"): ("passing_yards", "attempts", 15),
    ("rush_yds", "RB"): ("rushing_yards", "carries", 5),
    ("rush_yds", "QB"): ("rushing_yards", "attempts", 15),
    ("rec_yds", "WR"): ("receiving_yards", "targets", 3),
    ("rec_yds", "TE"): ("receiving_yards", "targets", 2),
    ("rec_yds", "RB"): ("receiving_yards", "targets", 2),
    ("rush_rec_yds", "RB"): ("rushing_yards + receiving_yards", "carries", 5),
    ("rush_rec_yds", "WR"): ("rushing_yards + receiving_yards", "targets", 3),
}
N_QUANTILES = 201

# Player-season qualifiers (per-game averages) so correlations reflect real roles.
QUALIFY = {"QB": "avg(attempts) >= 20", "RB": "avg(carries) >= 7", "WR": "avg(targets) >= 4", "TE": "avg(targets) >= 3"}


def pair_key(rel: str, pa: str, ma: str, pb: str, mb: str) -> str:
    a, b = sorted([(pa, ma), (pb, mb)])
    return f"{rel}|{a[0]}|{a[1]}|{b[0]}|{b[1]}"


class Calibration:
    def __init__(self, data: dict | None = None):
        self.data = data or {}
        self.loaded = bool(data)

    @classmethod
    def load(cls) -> "Calibration":
        if config.CALIBRATION_PATH.exists():
            return cls(json.loads(config.CALIBRATION_PATH.read_text()))
        return cls()

    def _lookup(self, table: str, defaults: dict, market: str, pos: str):
        learned = self.data.get(table, {})
        for key in (f"{market}|{pos}", market):
            if key in learned:
                return learned[key]
        for key in (f"{market}|{pos}", market):
            if key in defaults:
                return defaults[key]
        return None

    def cv(self, market: str, pos: str, mean: float | None = None) -> float:
        """Coefficient of variation; scales with the projected mean when a fit is available."""
        fit = self.data.get("cv_fit", {}).get(f"{market}|{pos}")
        if fit and mean and mean > 0:
            a, b = fit
            return float(min(1.5, max(0.18, math.exp(a + b * math.log(mean)))))
        return self._lookup("cv", DEFAULT_CV, market, pos) or 0.7

    def ratio_quantiles(self, market: str, pos: str, mean: float) -> list[float] | None:
        """Quantiles of outcome / pre-game expectation for players at this expected level."""
        entry = self.data.get("ratio_q", {}).get(f"{market}|{pos}")
        if not entry:
            return None
        tier = sum(mean > e for e in entry["edges"])
        return entry["q"][tier]

    def nb_size(self, market: str, pos: str) -> float | None:
        return self._lookup("nb_size", DEFAULT_NB, market, pos)

    def pair(self, rel: str, pa: str, ma: str, pb: str, mb: str) -> float:
        key = pair_key(rel, pa, ma, pb, mb)
        if key in self.data.get("corr_pair", {}):
            return self.data["corr_pair"][key]
        if key in DEFAULT_PAIR:
            return DEFAULT_PAIR[key]
        if rel == "same":
            return 0.4 if ma != mb else 0.95
        return 0.0

    def env(self, pos: str, market: str, var: str) -> float:
        key = f"{pos}|{market}|{var}"
        return self.data.get("corr_env", {}).get(key, DEFAULT_ENV.get(key, 0.0))

    def weather(self, stat: str, cond) -> tuple[float, str] | None:
        """Multiplier for a team stat (pass_yds, pass_comp, rush_yds, points) in these conditions."""
        from cajetan.weather import TEMP_LABELS, WIND_LABELS, temp_bucket, wind_bucket

        table = self.data.get("weather", {}).get(stat)
        if not table or cond is None:
            return None
        if cond.indoors:
            mult, n = table["indoors"]
            return mult, f"indoor games run {(mult - 1) * 100:+.1f}% vs average ({n:,} team-games)"
        mult, parts = 1.0, []
        wb, tb = wind_bucket(cond.wind), temp_bucket(cond.temp)
        if wb and wb in table["wind"]:
            m, n = table["wind"][wb]
            mult *= m
            parts.append(f"wind {WIND_LABELS[wb]}: {(m - 1) * 100:+.1f}% ({n:,} team-games)")
        if tb and tb in table["temp"]:
            m, n = table["temp"][tb]
            mult *= m
            parts.append(f"{TEMP_LABELS[tb]}: {(m - 1) * 100:+.1f}%")
        return (mult, "; ".join(parts)) if parts else None

    def injury(self, market: str, status: str) -> tuple[float, int] | None:
        """How players carrying this designation (who then played) produced vs their season average."""
        entry = self.data.get("injury", {}).get(market, {}).get(status)
        return tuple(entry) if entry else None

    def vacancy(self, kind: str) -> float:
        """Share of proportionally redistributed vacated volume a player actually absorbs."""
        return float(self.data.get("vacancy", {}).get(kind, 0.5))


def _long_table(con: duckdb.DuckDBPyConnection) -> None:
    unions = []
    for pos, markets in POS_MARKETS.items():
        for mk in markets:
            unions.append(
                f"SELECT game_id, season, team, player_id, '{pos}' AS pos, '{mk}' AS market, "
                f"CAST({CORR_MARKETS[mk]} AS DOUBLE) AS value FROM qual WHERE position = '{pos}'"
            )
    qualified = " UNION ALL ".join(
        f"SELECT player_id, season FROM player_week WHERE position = '{p}' AND season_type = 'REG' "
        f"GROUP BY player_id, season HAVING {cond} AND count(*) >= 4"
        for p, cond in QUALIFY.items()
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE long_stats AS
        WITH q AS ({qualified}),
        qual AS (
            SELECT w.* REPLACE (COALESCE(attempts, 0) AS attempts, COALESCE(completions, 0) AS completions,
                   COALESCE(passing_yards, 0) AS passing_yards, COALESCE(passing_tds, 0) AS passing_tds,
                   COALESCE(carries, 0) AS carries, COALESCE(rushing_yards, 0) AS rushing_yards,
                   COALESCE(rushing_tds, 0) AS rushing_tds, COALESCE(receptions, 0) AS receptions,
                   COALESCE(receiving_yards, 0) AS receiving_yards, COALESCE(receiving_tds, 0) AS receiving_tds)
            FROM player_week w JOIN q USING (player_id, season)
            WHERE w.season_type IN ('REG', 'POST')
        )
        {' UNION ALL '.join(unions)}
        """
    )


def compute(con: duckdb.DuckDBPyConnection, log=print) -> dict:
    _long_table(con)
    out: dict = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "cv": {}, "cv_fit": {}, "nb_size": {}}

    log("Dispersion of yardage and count stats")
    disp = con.execute(
        """
        SELECT market, pos, player_id, season, avg(value) AS m, var_samp(value) AS v, count(*) AS g
        FROM long_stats GROUP BY ALL HAVING count(*) >= 6 AND avg(value) > 0
        """
    ).df()
    for (market, pos), grp in disp.groupby(["market", "pos"]):
        if market in ("pass_yds", "rush_yds", "rec_yds", "rush_rec_yds"):
            cvs = grp["v"] ** 0.5 / grp["m"]
            out["cv"][f"{market}|{pos}"] = round(float(cvs.median()) * 1.05, 4)
            # Busier players are relatively more consistent: log(cv) = a + b*log(mean).
            ok = (grp["m"] > 1) & (cvs > 0)
            if ok.sum() >= 30:
                b, a = np.polyfit(np.log(grp.loc[ok, "m"]), np.log(cvs[ok] * 1.05), 1)
                out["cv_fit"][f"{market}|{pos}"] = [round(float(a), 4), round(float(b), 4)]
        elif market in ("pass_att", "pass_comp", "pass_tds", "rush_att", "rec"):
            excess = (grp["v"] - grp["m"]).sum()
            out["nb_size"][f"{market}|{pos}"] = (
                round(float((grp["m"] ** 2).sum() / excess), 3) if excess > 0 else None)

    log("Empirical outcome distributions (actual / pre-game average)")
    out["ratio_q"] = {}
    for (market, pos), (stat, vol, min_vol) in RATIO_MARKETS.items():
        df = con.execute(
            f"""
            WITH g AS (
                SELECT CAST({stat} AS DOUBLE) AS v,
                       avg({stat}) OVER w AS prior, avg(COALESCE({vol}, 0)) OVER w AS prior_vol,
                       count(*) OVER w AS n_prior
                FROM (SELECT * REPLACE (COALESCE(passing_yards, 0) AS passing_yards,
                             COALESCE(rushing_yards, 0) AS rushing_yards,
                             COALESCE(receiving_yards, 0) AS receiving_yards) FROM player_week)
                WHERE season_type = 'REG' AND position = '{pos}'
                WINDOW w AS (PARTITION BY player_id, season ORDER BY week
                             ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
            )
            SELECT v, prior FROM g WHERE n_prior >= 3 AND prior > 0 AND prior_vol >= {min_vol}
            """
        ).df()
        if len(df) < 600:
            continue
        edges = [float(x) for x in np.quantile(df["prior"], [1 / 3, 2 / 3])]
        tiers = np.digitize(df["prior"], edges)
        qs = []
        for t in range(3):
            ratio = (df["v"] / df["prior"])[tiers == t].to_numpy()
            ratio = ratio / ratio.mean()  # keep the shape; the projection supplies the level
            qs.append([round(float(x), 4) for x in np.quantile(ratio, np.linspace(0, 1, N_QUANTILES))])
        out["ratio_q"][f"{market}|{pos}"] = {"edges": [round(e, 2) for e in edges], "q": qs, "n": int(len(df))}

    log("Correlations between player stats in the same game")
    pairs = con.execute(
        """
        SELECT CASE WHEN a.player_id = b.player_id THEN 'same'
                    WHEN a.team = b.team THEN 'teammate' ELSE 'opponent' END AS rel,
               a.pos AS pa, a.market AS ma, b.pos AS pb, b.market AS mb,
               corr(a.value, b.value) AS r, count(*) AS n
        FROM long_stats a JOIN long_stats b
          ON a.game_id = b.game_id
         AND (a.player_id, a.market) < (b.player_id, b.market)
        GROUP BY ALL HAVING count(*) >= 200
        """
    ).df()
    corr_pair: dict[str, list[float]] = {}
    for r in pairs.itertuples():
        if r.r == r.r:
            corr_pair.setdefault(pair_key(r.rel, r.pa, r.ma, r.pb, r.mb), []).append((r.r, r.n))
    out["corr_pair"] = {k: round(sum(c * n for c, n in v) / sum(n for _, n in v), 4) for k, v in corr_pair.items()}

    log("Correlations between player stats and game results")
    env = con.execute(
        """
        SELECT s.pos, s.market,
               corr(s.value, tg.points - tg.opp_points) AS margin,
               corr(s.value, tg.points + tg.opp_points) AS total,
               corr(s.value, tg.points) AS team_pts,
               corr(s.value, tg.opp_points) AS opp_pts,
               count(*) AS n
        FROM long_stats s JOIN team_games tg ON tg.game_id = s.game_id AND tg.team = s.team
        WHERE tg.points IS NOT NULL GROUP BY ALL
        """
    ).df()
    out["corr_env"] = {
        f"{r.pos}|{r.market}|{var}": round(float(getattr(r, var)), 4)
        for r in env.itertuples() for var in ("margin", "total", "team_pts", "opp_pts")
        if getattr(r, var) == getattr(r, var)
    }

    log("Weather effects on team passing, rushing and scoring")
    out["weather"] = _weather(con)
    log("Injury designations and vacated teammate volume")
    out["injury"], out["vacancy"] = _injury_effects(con)
    return out


WEATHER_K = 60.0  # pseudo-games shrinking each weather bucket toward no effect


def _weather(con: duckdb.DuckDBPyConnection) -> dict:
    """Team stat in a game / the team's season-to-date average before it, by wind and temperature bucket."""
    from cajetan.weather import temp_bucket, wind_bucket

    df = con.execute(
        """
        WITH t AS (
            SELECT w.game_id, w.team, w.season, w.week,
                   sum(COALESCE(passing_yards, 0)) AS pass_yds, sum(COALESCE(completions, 0)) AS pass_comp,
                   sum(COALESCE(rushing_yards, 0)) AS rush_yds
            FROM player_week w WHERE season_type = 'REG' GROUP BY ALL
        ),
        j AS (
            SELECT t.*, tg.points, s.roof, s.temp, s.wind FROM t
            JOIN team_games tg ON tg.game_id = t.game_id AND tg.team = t.team
            JOIN schedules s ON s.game_id = t.game_id
            WHERE tg.points IS NOT NULL
        )
        SELECT *,
               avg(pass_yds) OVER w AS b_pass_yds, avg(pass_comp) OVER w AS b_pass_comp,
               avg(rush_yds) OVER w AS b_rush_yds, avg(points) OVER w AS b_points, count(*) OVER w AS n_prior
        FROM j
        WINDOW w AS (PARTITION BY team, season ORDER BY week ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        """
    ).df()
    df = df[df["n_prior"] >= 3]
    indoors = df["roof"].isin(["dome", "closed"])
    out = {}
    for stat in ("pass_yds", "pass_comp", "rush_yds", "points"):
        ok = df[f"b_{stat}"] > 0
        ratio = (df[stat] / df[f"b_{stat}"]).where(ok)
        overall = float(ratio.mean())

        def cell(mask) -> list:
            r = ratio[mask & ok]
            mult = (float(r.sum()) + WEATHER_K * overall) / (len(r) + WEATHER_K) / overall
            return [round(mult, 4), int(len(r))]

        wind = df["wind"].where(~indoors).map(lambda w: wind_bucket(None if w != w else w))
        temp = df["temp"].where(~indoors).map(lambda t: temp_bucket(None if t != t else t))
        out[stat] = {
            "indoors": cell(indoors),
            "wind": {b: cell(wind == b) for b in ("calm", "breezy", "windy", "very_windy")},
            "temp": {b: cell(temp == b) for b in ("frigid", "cold", "mild")},
        }
    return out


INJURY_MARKETS = ("pass_yds", "pass_att", "pass_comp", "rush_yds", "rush_att", "rec", "rec_yds", "rush_rec_yds")


def _injury_effects(con: duckdb.DuckDBPyConnection) -> tuple[dict, dict]:
    from cajetan.analysis import ml

    f = ml.build_frame(con)
    f = f[f["sd_games"].fillna(0) >= 2]
    injury: dict = {}
    for market in INJURY_MARKETS:
        k = ml.TARGET[market]
        rows = f[f["position"].isin(ml.MODEL_POS[market]) & (f[f"sd_{k}"] > 0)]
        base = rows[rows["inj_status"] == 0]
        r0 = base[f"s_{k}"].sum() / base[f"sd_{k}"].sum()
        injury[market] = {}
        for code, status in ((1, "Questionable"), (2, "Doubtful")):
            sub = rows[rows["inj_status"] == code]
            if len(sub) >= 100:
                r = sub[f"s_{k}"].sum() / sub[f"sd_{k}"].sum()
                injury[market][status] = [round(float(r / r0), 4), int(len(sub))]

    vacancy: dict = {}
    for kind, share, vac in (("tgt", "tgt_share", "vac_tgt"), ("carry", "carry_share", "vac_carry")):
        rows = f[(f[f"sd_{share}"] >= 0.05) & f[f"s_{share}"].notna()]
        ratio = rows[f"s_{share}"] / rows[f"sd_{share}"]
        r0 = float(ratio[rows[vac] == 0].mean())
        hit = rows[vac].between(0.03, 0.8)
        pred = 1 / (1 - rows.loc[hit, vac]) - 1  # proportional redistribution
        resid = ratio[hit] - r0
        k = float((resid * pred).sum() / (pred ** 2).sum()) if hit.sum() >= 100 else 0.5
        vacancy[kind] = round(min(1.0, max(0.0, k)), 4)
        vacancy[f"n_{kind}"] = int(hit.sum())
    return injury, vacancy


def save(data: dict) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.CALIBRATION_PATH.write_text(json.dumps(data, indent=1, sort_keys=True))
