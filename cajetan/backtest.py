"""Leak-free backtest: rate past games with only the data available before each one, then score them.

Player props are rated at a proxy line: the player's season-to-date average rounded to
the half point below it (no free source has historical sportsbook prop lines; real lines
come from the prediction log). Team markets are rated at the real closing consensus
spread and total from the nflverse schedule. For a season S the ML models are retrained
on seasons before S, so the projections never see the games they are scored on.

Each row keeps the pieces of the projection (matchup engine, ML model, market line) so
`cajetan fit` can re-weight them without re-rating anything.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import duckdb
import pandas as pd

from cajetan import config
from cajetan.parsing import Leg
from cajetan.predlog import settle

# Player markets rated per position, with the season-to-date volume needed to qualify.
POS_MARKETS = {
    "QB": ["pass_yds", "pass_comp", "pass_att", "pass_tds", "pass_int", "rush_yds"],
    "RB": ["rush_yds", "rush_att", "rec", "rec_yds", "rush_rec_yds", "anytime_td"],
    "WR": ["rec", "rec_yds", "anytime_td"],
    "TE": ["rec", "rec_yds", "anytime_td"],
}
QUALIFY = {"QB": ("sd_pass_att", 20), "RB": ("sd_rush_att", 8), "WR": ("sd_targets", 4), "TE": ("sd_targets", 3)}
MARKET_MIN = {"rush_yds|QB": ("sd_rush_yds", 15), "rec|RB": ("sd_targets", 2.5), "rec_yds|RB": ("sd_targets", 2.5)}
MIN_WEEK = 3


def proxy_line(avg: float) -> float:
    return math.floor(avg) + 0.5


def player_cases(frame: pd.DataFrame, season: int, weeks: list[int]) -> list[dict]:
    from cajetan.analysis import ml

    f = frame[(frame["season"] == season) & frame["week"].isin(weeks) & (frame["sd_games"].fillna(0) >= 2)]
    cases = []
    for r in f.itertuples(index=False):
        pos = r.position
        col, need = QUALIFY.get(pos, (None, None))
        if col is None or not (getattr(r, col) or 0) >= need:
            continue
        for market in POS_MARKETS[pos]:
            gate = MARKET_MIN.get(f"{market}|{pos}")
            if gate and not (getattr(r, gate[0]) or 0) >= gate[1]:
                continue
            key = ml.TARGET[market]
            avg = getattr(r, f"sd_{key}")
            actual = getattr(r, f"s_{key}")
            if avg is None or avg != avg or actual is None or actual != actual:
                continue
            line = 0.5 if market == "anytime_td" else proxy_line(avg)
            side = "yes" if market == "anytime_td" else "over"
            cases.append({"kind": "player", "season": season, "week": int(r.week), "game_id": r.game_id,
                          "player_id": r.player_id, "market": market, "side": side, "line": line,
                          "position": pos, "actual": float(actual)})
    return cases


def team_cases(con: duckdb.DuckDBPyConnection, season: int, weeks: list[int]) -> list[dict]:
    games = con.execute(
        f"""SELECT game_id, week, home_team, away_team, home_score, away_score, spread_line, total_line
            FROM schedules WHERE season = ? AND week IN ({','.join(map(str, weeks))})
              AND home_score IS NOT NULL AND spread_line IS NOT NULL AND total_line IS NOT NULL""",
        [season],
    ).fetchall()
    cases = []
    for gid, wk, home, away, hs, as_, spread, total in games:
        base = {"kind": "team", "season": season, "week": wk, "game_id": gid, "team": home}
        implied = total / 2 + spread / 2
        cases += [
            {**base, "market": "ml", "side": "win", "line": None, "actual": hs - as_},
            {**base, "market": "spread", "side": "cover", "line": -spread, "actual": hs - as_},
            {**base, "market": "game_total", "side": "over", "line": total, "actual": hs + as_, "opponent": away},
            {**base, "market": "team_total", "side": "over", "line": round(implied * 2) / 2, "actual": hs},
        ]
    return cases


_WORKER: dict = {}


def _rate_chunk(cases: list[dict]) -> list[dict]:
    """Runs in a worker process: rate a chunk of cases from one season."""
    from cajetan.analysis import ml
    from cajetan.calibration import Calibration
    from cajetan.db import connect
    from cajetan.engine import rate_leg

    if "con" not in _WORKER:
        _WORKER["con"] = connect(read_only=True)
        _WORKER["calib"] = Calibration.load()
        _WORKER["names"] = dict(_WORKER["con"].execute("SELECT player_id, player_name FROM players").fetchall())
    con, calib, names = _WORKER["con"], _WORKER["calib"], _WORKER["names"]
    season = cases[0]["season"]
    out = []
    with ml.models_through(season - 1, ml.build_frame(con)):
        for c in cases:
            if c["kind"] == "player":
                name = names.get(c["player_id"])
                if not name:
                    continue
                leg = Leg(f"{name} {c['side']} {c['line']} {c['market']}", "player", name, c["market"], c["side"],
                          c["line"])
            else:
                leg = Leg(f"{c['team']} {c['market']}", "team", c["team"], c["market"], c["side"], c["line"],
                          opponent=c.get("opponent"))
            res = rate_leg(con, leg, calib, c["week"], c["season"])
            if res.error or (c["kind"] == "player" and res.player_id != c["player_id"]):
                continue  # unresolvable or a different player with the same name
            result = settle(c["market"], c["side"], c["line"], c["actual"])
            out.append({**c, "name": name if c["kind"] == "player" else c["team"], "p": res.probability,
                        "p_push": res.p_push, "projection": res.projection, "reliability": res.reliability,
                        "result": result, **{f"c_{k}": v for k, v in res.components.items()}})
    return out


def run(con: duckdb.DuckDBPyConnection, seasons: list[int], weeks: list[int] | None = None,
        workers: int | None = None, max_player_legs: int | None = None, log: Callable[[str], None] = print
        ) -> pd.DataFrame:
    from cajetan.analysis import ml

    frame = ml.build_frame(con)
    cases: list[dict] = []
    for season in seasons:
        through = season - 1
        if not ml._paths(through)[0].exists():
            log(f"Training ML models on seasons through {through} (so {season} stays unseen)")
            ml.train(con, log=lambda m: None, max_season=through, evaluate=False)
        played = con.execute("SELECT DISTINCT week FROM schedules WHERE season = ? AND home_score IS NOT NULL "
                             "AND game_type = 'REG' ORDER BY 1", [season]).fetchall()
        wks = [w for (w,) in played if w >= MIN_WEEK and (weeks is None or w in weeks)]
        pc = player_cases(frame, season, wks)
        if max_player_legs and len(pc) > max_player_legs:
            pc = pd.DataFrame(pc).sample(max_player_legs, random_state=7).to_dict("records")
        tc = team_cases(con, season, wks)
        log(f"{season}: weeks {wks[0] if wks else '-'}-{wks[-1] if wks else '-'}, "
            f"{len(pc):,} player legs, {len(tc):,} team legs")
        cases += pc + tc
    if not cases:
        return pd.DataFrame()

    # Chunks of one season-week each keep every worker's caches warm.
    chunks: dict[tuple, list] = {}
    for c in cases:
        chunks.setdefault((c["season"], c["week"], c["kind"]), []).append(c)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_rate_chunk, chunk) for chunk in chunks.values()]
        for fut in as_completed(futures):
            rows += fut.result()
            done += 1
            if done % 10 == 0 or done == len(futures):
                log(f"  rated {done}/{len(futures)} season-weeks ({len(rows):,} legs)")
    df = pd.DataFrame(rows).sort_values(["season", "week", "kind", "game_id"]).reset_index(drop=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.BACKTEST_PATH, index=False)
    return df


def load() -> pd.DataFrame:
    if not config.BACKTEST_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(config.BACKTEST_PATH)
