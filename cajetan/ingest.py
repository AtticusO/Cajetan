"""Download nflverse data with nflreadpy and store it in the local DuckDB file.

Raw tables (one row per play / player-game / game) are upserted by season so the
current season can be refreshed cheaply. After every update the derived tables
(`plays`, `players`, `team_games`, snaps, injuries, Next Gen Stats, vacated volume)
are rebuilt; those are what the analysis code queries.

Seasons from HISTORY_FIRST_SEASON up to FIRST_SEASON - 1 are loaded "light": weekly
stats, schedules, rosters, snaps, injuries and Next Gen Stats only (no play-by-play),
which is all the ML model needs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timezone

import duckdb
import nflreadpy as nfl
import polars as pl

from cajetan import config
from cajetan.db import table_exists

S, F = pl.Utf8, pl.Float64

PBP_SCHEMA: dict[str, pl.DataType] = {
    "game_id": S, "play_id": pl.Int64, "season": pl.Int32, "week": pl.Int32, "season_type": S,
    "home_team": S, "away_team": S, "posteam": S, "defteam": S,
    "qtr": F, "down": F, "ydstogo": F, "yardline_100": F, "game_seconds_remaining": F,
    "score_differential": F, "play_type": S, "pass": F, "rush": F, "qb_dropback": F,
    "qb_scramble": F, "qb_kneel": F, "qb_spike": F, "sack": F, "qb_hit": F,
    "yards_gained": F, "epa": F, "success": F, "shotgun": F, "no_huddle": F,
    "pass_attempt": F, "rush_attempt": F, "complete_pass": F, "interception": F,
    "touchdown": F, "pass_touchdown": F, "rush_touchdown": F, "two_point_attempt": F,
    "passing_yards": F, "receiving_yards": F, "rushing_yards": F, "air_yards": F,
    "yards_after_catch": F, "passer_player_id": S, "receiver_player_id": S,
    "rusher_player_id": S, "td_player_id": S, "run_location": S, "run_gap": S,
    "pass_location": S, "pass_length": S, "fixed_drive": F,
}

FTN_SCHEMA: dict[str, pl.DataType] = {
    "nflverse_game_id": S, "nflverse_play_id": pl.Int64, "season": pl.Int32, "week": pl.Int32,
    "qb_location": S, "n_offense_backfield": F, "n_defense_box": F, "is_motion": pl.Boolean,
    "is_play_action": pl.Boolean, "is_screen_pass": pl.Boolean, "is_rpo": pl.Boolean,
    "is_qb_out_of_pocket": pl.Boolean, "is_qb_sneak": pl.Boolean, "n_blitzers": F,
    "n_pass_rushers": F,
}

PARTICIPATION_SCHEMA: dict[str, pl.DataType] = {
    "nflverse_game_id": S, "play_id": pl.Int64, "season": pl.Int32, "possession_team": S,
    "offense_formation": S, "offense_personnel": S, "defenders_in_box": F,
    "defense_personnel": S, "number_of_pass_rushers": F, "was_pressure": pl.Boolean,
    "defense_man_zone_type": S, "defense_coverage_type": S,
}

PLAYER_WEEK_SCHEMA: dict[str, pl.DataType] = {
    "player_id": S, "player_name": S, "position": S, "season": pl.Int32, "week": pl.Int32,
    "season_type": S, "game_id": S, "team": S, "opponent_team": S,
    "completions": F, "attempts": F, "passing_yards": F, "passing_tds": F,
    "passing_interceptions": F, "sacks_suffered": F, "carries": F, "rushing_yards": F,
    "rushing_tds": F, "receptions": F, "targets": F, "receiving_yards": F,
    "receiving_tds": F, "target_share": F, "air_yards_share": F,
}

SCHEDULE_SCHEMA: dict[str, pl.DataType] = {
    "game_id": S, "season": pl.Int32, "game_type": S, "week": pl.Int32, "gameday": S,
    "away_team": S, "home_team": S, "away_score": F, "home_score": F, "result": F,
    "total": F, "spread_line": F, "total_line": F, "away_moneyline": F,
    "home_moneyline": F, "roof": S, "surface": S, "temp": F, "wind": F, "div_game": F,
    "away_qb_id": S, "home_qb_id": S, "location": S, "stadium_id": S, "gametime": S,
}

ROSTER_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int32, "week": pl.Int32, "team": S, "position": S,
    "depth_chart_position": S, "full_name": S, "gsis_id": S, "status": S,
    "birth_date": S, "years_exp": F, "entry_year": F, "pfr_id": S,
}

INJURY_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int32, "game_type": S, "team": S, "week": pl.Int32, "gsis_id": S, "position": S,
    "full_name": S, "report_primary_injury": S, "report_status": S, "practice_status": S,
}

SNAP_SCHEMA: dict[str, pl.DataType] = {
    "game_id": S, "season": pl.Int32, "game_type": S, "week": pl.Int32, "player": S,
    "pfr_player_id": S, "position": S, "team": S, "opponent": S, "offense_snaps": F, "offense_pct": F,
}

NGS_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "receiving": {"season": pl.Int32, "season_type": S, "week": pl.Int32, "player_gsis_id": S,
                  "avg_separation": F, "avg_cushion": F, "avg_intended_air_yards": F,
                  "avg_yac_above_expectation": F, "targets": F},
    "rushing": {"season": pl.Int32, "season_type": S, "week": pl.Int32, "player_gsis_id": S,
                "efficiency": F, "percent_attempts_gte_eight_defenders": F,
                "rush_yards_over_expected_per_att": F, "rush_attempts": F},
    "passing": {"season": pl.Int32, "season_type": S, "week": pl.Int32, "player_gsis_id": S,
                "avg_time_to_throw": F, "aggressiveness": F, "avg_intended_air_yards": F,
                "completion_percentage_above_expectation": F, "attempts": F},
}

# Tables that hold per-season raw data, and whether "light" history seasons load them.
SEASON_TABLES = ("pbp", "player_week", "schedules", "ftn", "participation", "rosters",
                 "injuries", "snap_counts", "ngs_receiving", "ngs_rushing", "ngs_passing")


def conform(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Select and cast the columns we keep; columns missing from a season become nulls."""
    cols = []
    for name, dtype in schema.items():
        if name in df.columns:
            cols.append(pl.col(name).cast(dtype, strict=False).alias(name))
        else:
            cols.append(pl.lit(None, dtype=dtype).alias(name))
    return df.select(cols)


def upsert(con: duckdb.DuckDBPyConnection, table: str, df: pl.DataFrame, seasons: Iterable[int]) -> int:
    seasons = list(seasons)
    con.register("_incoming", df.to_arrow())
    try:
        if not table_exists(con, table):
            con.execute(f"CREATE TABLE {table} AS SELECT * FROM _incoming")
        else:
            have = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}
            for name, dtype, *_ in con.execute("DESCRIBE _incoming").fetchall():
                if name not in have:  # schema grew since the table was created
                    con.execute(f'ALTER TABLE {table} ADD COLUMN "{name}" {dtype}')
            con.execute(f"DELETE FROM {table} WHERE season IN ({','.join(map(str, seasons))})")
            con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _incoming")
    finally:
        con.unregister("_incoming")
    return df.height


def _participation(season: int) -> pl.DataFrame | None:
    try:
        df = nfl.load_participation([season])
    except Exception:
        return None  # nflverse has not published this season (common for the current one)
    if df.height == 0:
        return None
    df = df.with_columns(pl.col("nflverse_game_id").str.slice(0, 4).cast(pl.Int32).alias("season"))
    return conform(df, PARTICIPATION_SCHEMA)


def _ftn(season: int) -> pl.DataFrame | None:
    try:
        df = nfl.load_ftn_charting([season])
    except Exception:
        return None
    return conform(df, FTN_SCHEMA) if df.height else None


def _optional(loader: Callable[[], pl.DataFrame], schema: dict[str, pl.DataType]) -> pl.DataFrame | None:
    """Datasets nflverse may not have published yet for a season."""
    try:
        df = loader()
    except Exception:
        return None
    return conform(df, schema) if df.height else None


def _upsert_optional(con, table: str, df: pl.DataFrame | None, season: int, record, log) -> None:
    if df is not None:
        record(table, season, upsert(con, table, df, [season]))
    else:
        log(f"[{season}]   {table} not published by nflverse for this season")
        if table_exists(con, table):
            con.execute(f"DELETE FROM {table} WHERE season = {season}")


def update(
    con: duckdb.DuckDBPyConnection,
    seasons: list[int],
    log: Callable[[str], None] = print,
) -> dict[str, dict[int, int]]:
    """Fetch every dataset for `seasons`, upsert it, then rebuild the derived tables."""
    counts: dict[str, dict[int, int]] = {}

    def record(table: str, season: int, n: int) -> None:
        counts.setdefault(table, {})[season] = n

    for season in seasons:
        light = season < config.FIRST_SEASON
        if not light:
            log(f"[{season}] play-by-play")
            pbp = conform(nfl.load_pbp([season]), PBP_SCHEMA)
            record("pbp", season, upsert(con, "pbp", pbp, [season]))

        log(f"[{season}] weekly player stats")
        pw = nfl.load_player_stats([season])
        if "player_name" in pw.columns and "player_display_name" in pw.columns:
            pw = pw.drop("player_name").rename({"player_display_name": "player_name"})
        record("player_week", season, upsert(con, "player_week", conform(pw, PLAYER_WEEK_SCHEMA), [season]))

        log(f"[{season}] schedule & betting lines")
        sched = conform(nfl.load_schedules([season]), SCHEDULE_SCHEMA)
        record("schedules", season, upsert(con, "schedules", sched, [season]))

        if not light:
            log(f"[{season}] FTN charting (play action, motion, blitz, box)")
            ftn = _ftn(season)
            if ftn is not None:
                record("ftn", season, upsert(con, "ftn", ftn, [season]))
            elif table_exists(con, "ftn"):
                con.execute(f"DELETE FROM ftn WHERE season = {season}")

            log(f"[{season}] participation (personnel, formation, coverage)")
            part = _participation(season)
            if part is not None:
                record("participation", season, upsert(con, "participation", part, [season]))
            else:
                log(f"[{season}]   participation not published by nflverse; personnel will fall back to earlier seasons")

        log(f"[{season}] rosters")
        rost = conform(nfl.load_rosters([season]), ROSTER_SCHEMA)
        record("rosters", season, upsert(con, "rosters", rost, [season]))

        log(f"[{season}] injury reports, snap counts, Next Gen Stats")
        _upsert_optional(con, "injuries", _optional(lambda: nfl.load_injuries([season]), INJURY_SCHEMA),
                         season, record, log)
        _upsert_optional(con, "snap_counts", _optional(lambda: nfl.load_snap_counts([season]), SNAP_SCHEMA),
                         season, record, log)
        for kind, schema in NGS_SCHEMAS.items():
            df = _optional(lambda: nfl.load_nextgen_stats(stat_type=kind, seasons=[season]), schema)
            _upsert_optional(con, f"ngs_{kind}", df, season, record, log)

    log("Building derived tables")
    build_derived(con)
    con.execute("CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
    con.execute(
        "INSERT OR REPLACE INTO meta VALUES ('last_update', ?)",
        [datetime.now(timezone.utc).isoformat(timespec="seconds")],
    )
    return counts


def _ensure_optional_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create empty optional tables so the derived-table joins work without them."""
    optional = [("ftn", FTN_SCHEMA), ("participation", PARTICIPATION_SCHEMA), ("injuries", INJURY_SCHEMA),
                ("snap_counts", SNAP_SCHEMA)] + [(f"ngs_{k}", s) for k, s in NGS_SCHEMAS.items()]
    for table, schema in optional:
        if not table_exists(con, table):
            empty = pl.DataFrame(schema=schema)
            con.register("_empty", empty.to_arrow())
            con.execute(f"CREATE TABLE {table} AS SELECT * FROM _empty")
            con.unregister("_empty")


def build_derived(con: duckdb.DuckDBPyConnection) -> None:
    _ensure_optional_tables(con)

    # Latest known name / position / team for every player.
    con.execute(
        """
        CREATE OR REPLACE TABLE players AS
        WITH ranked AS (
            SELECT player_id, player_name, position, team, season, week,
                   row_number() OVER (PARTITION BY player_id ORDER BY season DESC, week DESC) AS rn
            FROM player_week
            WHERE player_id IS NOT NULL
        ),
        roster_latest AS (
            SELECT gsis_id, team, position, full_name,
                   row_number() OVER (PARTITION BY gsis_id ORDER BY season DESC, week DESC NULLS LAST) AS rn
            FROM rosters WHERE gsis_id IS NOT NULL
              AND season = (SELECT max(season) FROM rosters)
        )
        SELECT r.player_id,
               COALESCE(r.player_name, rl.full_name) AS player_name,
               COALESCE(r.position, rl.position) AS position,
               COALESCE(rl.team, r.team) AS team,
               r.season AS last_season, r.week AS last_week
        FROM ranked r
        LEFT JOIN roster_latest rl ON rl.gsis_id = r.player_id AND rl.rn = 1
        WHERE r.rn = 1
        """
    )

    con.execute(
        r"""
        CREATE OR REPLACE TABLE plays AS
        WITH base AS (
            SELECT p.*,
                   f.qb_location, f.n_offense_backfield, NULLIF(f.n_defense_box, 0) AS ftn_box,
                   f.is_motion, f.is_play_action, f.is_rpo, f.is_screen_pass, f.is_qb_sneak,
                   f.n_blitzers,
                   pa.offense_formation, pa.offense_personnel, pa.defenders_in_box,
                   pa.defense_man_zone_type, pa.defense_coverage_type, pa.was_pressure,
                   rp.position AS rusher_pos, cp.position AS receiver_pos
            FROM pbp p
            LEFT JOIN ftn f ON f.nflverse_game_id = p.game_id AND f.nflverse_play_id = p.play_id
            LEFT JOIN participation pa ON pa.nflverse_game_id = p.game_id AND pa.play_id = p.play_id
            LEFT JOIN players rp ON rp.player_id = p.rusher_player_id
            LEFT JOIN players cp ON cp.player_id = p.receiver_player_id
            WHERE p.season_type IN ('REG', 'POST')
              AND p.play_type IN ('pass', 'run')
              AND COALESCE(p.qb_kneel, 0) = 0
              AND COALESCE(p.qb_spike, 0) = 0
              AND COALESCE(p.two_point_attempt, 0) = 0
        )
        SELECT
            season, week, season_type, game_id, play_id, posteam, defteam, home_team, away_team,
            qtr, down, ydstogo, yardline_100, game_seconds_remaining, score_differential, fixed_drive,
            CAST(COALESCE(pass_attempt, 0) AS INT) AS pass_attempt,
            CAST(COALESCE(rush_attempt, 0) AS INT) AS rush_attempt,
            CAST(COALESCE(qb_dropback, 0) AS INT) AS dropback,
            CAST(COALESCE(qb_scramble, 0) AS INT) AS scramble,
            CAST(COALESCE(sack, 0) AS INT) AS sack,
            CAST(COALESCE(qb_hit, 0) AS INT) AS qb_hit,
            CAST(COALESCE(complete_pass, 0) AS INT) AS complete_pass,
            CAST(COALESCE(interception, 0) AS INT) AS interception,
            CAST(COALESCE(pass_touchdown, 0) AS INT) AS pass_touchdown,
            CAST(COALESCE(rush_touchdown, 0) AS INT) AS rush_touchdown,
            yards_gained, epa, success,
            COALESCE(passing_yards, 0) AS passing_yards,
            COALESCE(receiving_yards, 0) AS receiving_yards,
            COALESCE(rushing_yards, 0) AS rushing_yards,
            air_yards, passer_player_id, receiver_player_id, rusher_player_id, td_player_id,
            rusher_pos, receiver_pos,
            yardline_100 <= 20 AS redzone,
            qtr >= 4 AS late,
            CASE WHEN run_gap = 'end' THEN 'outside'
                 WHEN run_gap = 'tackle' THEN 'off_tackle'
                 WHEN run_gap = 'guard' OR run_location = 'middle' THEN 'inside' END AS run_dir,
            CASE WHEN COALESCE(rush_attempt, 0) = 0 OR rusher_pos IS DISTINCT FROM 'QB' THEN NULL
                 WHEN COALESCE(qb_scramble, 0) = 1 THEN 'scramble'
                 WHEN COALESCE(is_qb_sneak, false) THEN 'sneak'
                 ELSE 'designed' END AS qb_run_type,
            COALESCE(
                CASE qb_location WHEN 'S' THEN 'shotgun' WHEN 'U' THEN 'under_center' WHEN 'P' THEN 'pistol' END,
                CASE offense_formation WHEN 'SHOTGUN' THEN 'shotgun' WHEN 'UNDER CENTER' THEN 'under_center'
                                       WHEN 'PISTOL' THEN 'pistol' END,
                CASE WHEN shotgun = 1 THEN 'shotgun' ELSE 'under_center' END) AS formation,
            CASE WHEN n_offense_backfield IS NULL OR qb_location = '0' THEN NULL
                 WHEN n_offense_backfield = 0 THEN 'empty'
                 WHEN n_offense_backfield = 1 THEN 'one_back'
                 ELSE 'two_back' END AS backfield,
            CASE WHEN COALESCE(ftn_box, defenders_in_box) IS NULL THEN NULL
                 WHEN COALESCE(ftn_box, defenders_in_box) <= 6 THEN 'light_box'
                 WHEN COALESCE(ftn_box, defenders_in_box) = 7 THEN 'neutral_box'
                 ELSE 'heavy_box' END AS box,
            CASE WHEN offense_personnel IS NULL OR offense_personnel = '' THEN NULL ELSE
                CAST(COALESCE(TRY_CAST(regexp_extract(offense_personnel, '(\d+) RB', 1) AS INT), 0)
                   + COALESCE(TRY_CAST(regexp_extract(offense_personnel, '(\d+) FB', 1) AS INT), 0) AS VARCHAR)
                || CAST(COALESCE(TRY_CAST(regexp_extract(offense_personnel, '(\d+) TE', 1) AS INT), 0) AS VARCHAR)
            END AS personnel,
            CASE WHEN is_play_action IS NULL THEN NULL WHEN is_play_action THEN 'play_action' ELSE 'no_play_action' END AS play_action,
            CASE WHEN is_motion IS NULL THEN NULL WHEN is_motion THEN 'motion' ELSE 'no_motion' END AS motion,
            CASE WHEN is_rpo IS NULL THEN NULL WHEN is_rpo THEN 'rpo' ELSE 'no_rpo' END AS rpo,
            CASE WHEN is_screen_pass IS NULL THEN NULL WHEN is_screen_pass THEN 'screen' ELSE 'no_screen' END AS screen,
            CASE WHEN n_blitzers IS NULL OR COALESCE(qb_dropback, 0) = 0 OR qb_location = '0' THEN NULL
                 WHEN n_blitzers > 0 THEN 'blitz' ELSE 'no_blitz' END AS blitz,
            CASE defense_man_zone_type WHEN 'MAN_COVERAGE' THEN 'man' WHEN 'ZONE_COVERAGE' THEN 'zone' END AS coverage,
            lower(NULLIF(defense_coverage_type, '')) AS coverage_shell,
            CASE WHEN was_pressure IS NULL THEN NULL WHEN was_pressure THEN 'pressured' ELSE 'clean' END AS pressure
        FROM base
        """
    )

    # Per-team, per-game results used by the team model and the game-script regression.
    con.execute(
        """
        CREATE OR REPLACE TABLE team_games AS
        WITH g AS (
            SELECT game_id, season, week, game_type, home_team AS team, away_team AS opponent, TRUE AS is_home,
                   home_score AS points, away_score AS opp_points, spread_line AS exp_margin, total_line
            FROM schedules
            UNION ALL
            SELECT game_id, season, week, game_type, away_team, home_team, FALSE,
                   away_score, home_score, -spread_line, total_line
            FROM schedules
        ),
        vol AS (
            SELECT game_id, team, sum(attempts) AS pass_att, sum(carries) AS carries
            FROM player_week GROUP BY ALL
        )
        SELECT g.*, v.pass_att, v.carries
        FROM g LEFT JOIN vol v USING (game_id, team)
        """
    )
    build_player_context(con)


def build_player_context(con: duckdb.DuckDBPyConnection) -> None:
    """Snaps, injury status, Next Gen Stats, age and vacated volume, keyed on nflverse player ids."""
    # Snap counts are keyed on Pro Football Reference ids; rosters map them to nflverse (gsis) ids.
    con.execute(
        """
        CREATE OR REPLACE TABLE player_snaps AS
        WITH ids AS (SELECT pfr_id, any_value(gsis_id) AS player_id FROM rosters
                     WHERE pfr_id IS NOT NULL AND gsis_id IS NOT NULL GROUP BY 1)
        SELECT ids.player_id, s.season, s.week, s.game_id, s.team, s.offense_snaps, s.offense_pct AS snap_pct
        FROM snap_counts s JOIN ids ON ids.pfr_id = s.pfr_player_id
        WHERE s.game_type IN ('REG', 'WC', 'DIV', 'CON', 'SB') AND s.offense_snaps > 0
        QUALIFY row_number() OVER (PARTITION BY ids.player_id, s.game_id ORDER BY s.offense_snaps DESC) = 1
        """
    )
    # Final game-status designation and the last practice report for each player-week.
    con.execute(
        """
        CREATE OR REPLACE TABLE injury_status AS
        SELECT gsis_id AS player_id, season, week, team, any_value(full_name) AS name,
               any_value(position) AS position, any_value(report_primary_injury) AS injury,
               CASE max(CASE report_status WHEN 'Out' THEN 3 WHEN 'Doubtful' THEN 2
                                           WHEN 'Questionable' THEN 1 ELSE 0 END)
                    WHEN 3 THEN 'Out' WHEN 2 THEN 'Doubtful' WHEN 1 THEN 'Questionable' END AS status,
               max(CASE WHEN practice_status ILIKE 'Did Not%' THEN 2 WHEN practice_status ILIKE 'Limited%' THEN 1
                        WHEN practice_status ILIKE 'Full%' THEN 0 END) AS practice
        FROM injuries WHERE gsis_id IS NOT NULL
        GROUP BY gsis_id, season, week, team
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE ngs_week AS
        WITH r AS (SELECT player_gsis_id AS player_id, season, week, avg_separation AS separation,
                          avg_cushion AS cushion, avg_intended_air_yards AS rec_iay,
                          avg_yac_above_expectation AS yac_oe
                   FROM ngs_receiving WHERE week > 0),
             u AS (SELECT player_gsis_id AS player_id, season, week, efficiency AS rush_eff,
                          percent_attempts_gte_eight_defenders AS stacked_pct,
                          rush_yards_over_expected_per_att AS ryoe
                   FROM ngs_rushing WHERE week > 0),
             p AS (SELECT player_gsis_id AS player_id, season, week, avg_time_to_throw AS ttt,
                          aggressiveness, avg_intended_air_yards AS pass_iay,
                          completion_percentage_above_expectation AS cpoe
                   FROM ngs_passing WHERE week > 0)
        SELECT * FROM r FULL OUTER JOIN u USING (player_id, season, week)
                        FULL OUTER JOIN p USING (player_id, season, week)
        WHERE player_id IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE player_bio AS
        SELECT gsis_id AS player_id,
               arg_max(TRY_CAST(birth_date AS DATE), season) AS birth_date,
               min(entry_year) AS entry_year
        FROM rosters WHERE gsis_id IS NOT NULL GROUP BY 1
        """
    )
    # Per player-game shares, and season-to-date shares through each game (inclusive).
    con.execute(
        """
        CREATE OR REPLACE TABLE player_shares AS
        WITH b AS (
            SELECT w.player_id, w.season, w.week, w.game_id, w.team, w.position,
                   COALESCE(w.target_share, 0) AS tgt_share,
                   COALESCE(w.carries, 0) / NULLIF(tg.carries, 0) AS carry_share,
                   s.snap_pct
            FROM player_week w
            LEFT JOIN team_games tg ON tg.game_id = w.game_id AND tg.team = w.team
            LEFT JOIN player_snaps s ON s.player_id = w.player_id AND s.game_id = w.game_id
            WHERE w.season_type IN ('REG', 'POST') AND w.position IN ('QB', 'RB', 'WR', 'TE', 'FB')
        )
        SELECT *,
               avg(tgt_share) OVER w AS cum_tgt_share,
               avg(COALESCE(carry_share, 0)) OVER w AS cum_carry_share
        FROM b
        WINDOW w AS (PARTITION BY player_id, season ORDER BY week ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        """
    )
    # Volume vacated by teammates ruled Out/Doubtful who played the team's previous game
    # (if they were already missing, the recent baseline has absorbed it).
    con.execute(
        """
        CREATE OR REPLACE TABLE vacancy AS
        WITH tg AS (
            SELECT team, season, week, lag(week) OVER (PARTITION BY team, season ORDER BY week) AS prev_week
            FROM team_games WHERE game_type IS DISTINCT FROM 'PRE'
        ),
        outs AS (
            SELECT i.player_id, i.name, i.status, i.team, i.season, i.week, tg.prev_week
            FROM injury_status i JOIN tg USING (team, season, week)
            WHERE i.status IN ('Out', 'Doubtful')
        ),
        last AS (
            SELECT o.*, ps.cum_tgt_share, ps.cum_carry_share, ps.position,
                   row_number() OVER (PARTITION BY o.player_id, o.season, o.week ORDER BY ps.week DESC) AS rn
            FROM outs o JOIN player_shares ps
              ON ps.player_id = o.player_id AND ps.season = o.season AND ps.week < o.week AND ps.team = o.team
        )
        SELECT team, season, week, player_id, name, status, position,
               cum_tgt_share AS tgt_share, cum_carry_share AS carry_share
        FROM last
        WHERE rn = 1 AND prev_week IS NOT NULL
          AND EXISTS (SELECT 1 FROM player_shares p2 WHERE p2.player_id = last.player_id
                      AND p2.season = last.season AND p2.week = last.prev_week)
          AND (cum_tgt_share >= 0.03 OR cum_carry_share >= 0.03)
        """
    )


def seasons_to_load(con: duckdb.DuckDBPyConnection, full: bool, explicit: list[int] | None) -> list[int]:
    current = nfl.get_current_season()
    if explicit:
        return sorted(set(explicit))
    everything = list(range(config.HISTORY_FIRST_SEASON, current + 1))
    if full or not table_exists(con, "pbp") or not table_exists(con, "injuries"):
        return everything
    loaded = {r[0] for r in con.execute("SELECT DISTINCT season FROM player_week").fetchall()}
    return [s for s in everything if s not in loaded] + [current]
