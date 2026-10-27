"""Match typed player names to nflverse ids and find the game a leg refers to."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import duckdb

from cajetan.db import AsOf

SKILL_POSITIONS = ("QB", "RB", "WR", "TE", "FB")


class ResolveError(LookupError):
    pass


@dataclass
class PlayerRef:
    player_id: str
    name: str
    position: str
    team: str


@dataclass
class GameContext:
    season: int
    week: int
    game_id: str
    team: str
    opponent: str
    is_home: bool
    exp_margin: float | None  # market-implied margin from `team`'s perspective (+ = favored)
    total_line: float | None
    gameday: str | None

    @property
    def asof(self) -> AsOf:
        return AsOf(self.season, self.week)

    @property
    def implied_points(self) -> float | None:
        if self.exp_margin is None or self.total_line is None:
            return None
        return self.total_line / 2 + self.exp_margin / 2

    @property
    def label(self) -> str:
        return f"{self.team} {'vs' if self.is_home else '@'} {self.opponent} (wk {self.week})"


def normalize_name(name: str) -> str:
    name = name.lower().replace("’", "'")
    name = re.sub(r"[.'\-]", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    return " ".join(name.split())


def find_player(con: duckdb.DuckDBPyConnection, name: str, season: int,
                positions: tuple[str, ...] = SKILL_POSITIONS) -> PlayerRef:
    rows = con.execute(
        f"""
        SELECT p.player_id, p.player_name, p.position, p.team, p.last_season,
               (SELECT count(*) FROM player_week w WHERE w.player_id = p.player_id AND w.season >= ?) AS games
        FROM players p
        WHERE p.last_season >= ? AND p.position IN ({",".join("?" * len(SKILL_POSITIONS))})
        """,
        [season - 1, season - 1, *SKILL_POSITIONS],
    ).fetchall()
    target = normalize_name(name)
    by_norm: dict[str, list[tuple]] = {}
    for r in rows:
        by_norm.setdefault(normalize_name(r[1] or ""), []).append(r)

    matches = by_norm.get(target, [])
    if not matches:
        parts = target.split()
        if len(parts) >= 2 and len(parts[0]) == 1:  # "j allen"
            matches = [r for k, rs in by_norm.items() for r in rs
                       if k.split()[-1] == parts[-1] and k.startswith(parts[0])]
    if not matches:
        close = difflib.get_close_matches(target, list(by_norm), n=3, cutoff=0.8)
        if close:
            matches = by_norm[close[0]]
    if not matches:
        hint = difflib.get_close_matches(target, list(by_norm), n=3, cutoff=0.5)
        raise ResolveError(f"no player found for '{name}'" + (f" (did you mean {', '.join(h.title() for h in hint)}?)" if hint else ""))

    # Prefer players at a position the market applies to, then recent and frequent players.
    matches.sort(key=lambda r: (r[2] in positions, r[4], r[5]), reverse=True)
    pid, pname, pos, team, _, _ = matches[0]
    return PlayerRef(pid, pname, pos, team)


def player_team_for_week(con: duckdb.DuckDBPyConnection, player: PlayerRef, season: int, week: int | None) -> str:
    """Team the player was on for a specific (already played) week, else the current team."""
    if week is not None:
        row = con.execute(
            "SELECT team FROM player_week WHERE player_id = ? AND season = ? AND week = ?",
            [player.player_id, season, week],
        ).fetchone()
        if row:
            return row[0]
    return player.team


def find_game(con: duckdb.DuckDBPyConnection, team: str, season: int, week: int | None = None) -> GameContext:
    if week is None:
        row = con.execute(
            """
            SELECT game_id, week, home_team, away_team, spread_line, total_line, gameday
            FROM schedules
            WHERE season = ? AND (home_team = ? OR away_team = ?) AND home_score IS NULL
            ORDER BY week LIMIT 1
            """,
            [season, team, team],
        ).fetchone()
        if not row:
            raise ResolveError(f"{team} has no unplayed games left in {season}; pass --week to rate a past game")
    else:
        row = con.execute(
            """
            SELECT game_id, week, home_team, away_team, spread_line, total_line, gameday
            FROM schedules WHERE season = ? AND week = ? AND (home_team = ? OR away_team = ?)
            """,
            [season, week, team, team],
        ).fetchone()
        if not row:
            raise ResolveError(f"{team} has no game in week {week} of {season} (bye week?)")
    game_id, wk, home, away, spread, total, gameday = row
    is_home = team == home
    margin = None if spread is None else (spread if is_home else -spread)
    return GameContext(season, wk, game_id, team, away if is_home else home, is_home, margin, total, gameday)


def current_season(con: duckdb.DuckDBPyConnection) -> int:
    return con.execute("SELECT max(season) FROM schedules").fetchone()[0]
