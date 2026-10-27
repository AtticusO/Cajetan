"""`cajetan player` and `cajetan defense`: look at the splits the ratings are built from."""

from __future__ import annotations

import duckdb
from rich.console import Console
from rich.table import Table

from cajetan.analysis.markets import DIM_LABELS, FAMILIES, bucket_label, plural
from cajetan.analysis.matchup import PlayStats, fmt_metric, ordinal
from cajetan.analysis.team_eval import situational
from cajetan.db import AsOf
from cajetan.resolve import current_season, find_game, find_player

PLAYER_VIEWS = {
    "QB": [("dropback", "passing_yards", "yds/att", ["personnel", "formation", "play_action", "motion", "coverage", "blitz"]),
           ("rush", "rushing_yards", "yds/carry", ["qb_run_type", "run_dir"])],
    "RB": [("rush", "rushing_yards", "yds/carry", ["run_dir", "formation", "personnel", "box"]),
           ("target", "receiving_yards", "yds/target", ["personnel", "formation", "coverage"])],
    "WR": [("target", "receiving_yards", "yds/target", ["personnel", "formation", "play_action", "motion", "coverage", "blitz"])],
    "TE": [("target", "receiving_yards", "yds/target", ["personnel", "formation", "play_action", "motion", "coverage", "blitz"])],
}
DEFENSE_VIEWS = [
    ("rush", "rushing_yards", "yds/carry", "RB", ["run_dir", "box", "formation", "personnel"]),
    ("dropback", "passing_yards", "yds/att", None, ["personnel", "formation", "play_action", "motion", "coverage", "blitz"]),
    ("target", "receiving_yards", "yds/target", "WR", ["personnel", "coverage"]),
    ("target", "receiving_yards", "yds/target", "TE", ["personnel", "coverage"]),
    ("target", "receiving_yards", "yds/target", "RB", []),
    ("rush", "rushing_yards", "yds/carry", "QB", ["qb_run_type"]),
]


def _asof(con: duckdb.DuckDBPyConnection, week: int | None) -> AsOf:
    season = current_season(con)
    if week is None:
        row = con.execute("SELECT min(week) FROM schedules WHERE season = ? AND home_score IS NULL", [season]).fetchone()
        week = row[0] if row and row[0] else 99
    return AsOf(season, week)


def player_profile(con: duckdb.DuckDBPyConnection, console: Console, name: str, week: int | None = None) -> None:
    season = current_season(con)
    p = find_player(con, name, season)
    asof = _asof(con, week)
    try:
        g = find_game(con, p.team, season, week)
        nxt = f"next: {g.label}, {g.team} spread {-g.exp_margin:+g} / total {g.total_line:g}" if g.exp_margin is not None else f"next: {g.label}"
    except LookupError:
        nxt = ""
    console.print(f"[bold]{p.name}[/bold] ({p.position}, {p.team})  {nxt}")
    logs = con.execute(
        """
        SELECT season, count(*) g, avg(attempts) att, avg(passing_yards) pyd, avg(passing_tds) ptd,
               avg(carries) car, avg(rushing_yards) ryd, avg(targets) tgt, avg(receptions) rec,
               avg(receiving_yards) recyd, avg(rushing_tds + receiving_tds) td
        FROM player_week WHERE player_id = ? AND season >= ? AND season_type IN ('REG', 'POST')
        GROUP BY 1 ORDER BY 1
        """,
        [p.player_id, season - 1],
    ).fetchall()
    t = Table(title="Per-game averages", header_style="bold cyan")
    for c in ["Season", "G", "Att", "Pass yds", "Pass TD", "Car", "Rush yds", "Tgt", "Rec", "Rec yds", "Rush+rec TD"]:
        t.add_column(c, justify="right")
    for r in logs:
        t.add_row(str(r[0]), str(r[1]), *[f"{v or 0:.1f}" for v in r[2:]])
    console.print(t)
    ps = PlayStats(con, asof)
    for fam, metric, label, dims in PLAYER_VIEWS.get(p.position, []):
        unit = FAMILIES[fam]["unit"]
        t = Table(title=f"Scheme splits: {plural(unit)} ({label}), {season - 1}-{season} weighted", header_style="bold cyan")
        for c in ["Dimension", "Bucket", f"Share of {plural(unit)}", label, "Plays (this season)"]:
            t.add_column(c)
        for dim in dims:
            b = ps.buckets(fam, metric, dim, player_id=p.player_id)
            total = sum(s.n for s in b.values())
            for k, s in sorted(b.items(), key=lambda kv: -kv[1].n):
                t.add_row(DIM_LABELS[dim], bucket_label(dim, k), f"{s.n / total:.0%}" if total else "-",
                          fmt_metric(metric, s.rate), f"{s.n_cur:.0f}")
        console.print(t)


def defense_profile(con: duckdb.DuckDBPyConnection, console: Console, team: str, week: int | None = None) -> None:
    asof = _asof(con, week)
    ps = PlayStats(con, asof)
    console.print(f"[bold]{team} defense[/bold]  (data before {asof.season} week {asof.week}; "
                  f"{asof.season - 1} weighted lower)")
    t = Table(title="Allowed by position (rank 1 = stingiest)", header_style="bold cyan")
    for c in ["Split", f"{team} allows", "League", "Rank", "Plays (this season)"]:
        t.add_column(c)
    for fam, metric, label, pos, _ in DEFENSE_VIEWS:
        table = ps.defense_table(fam, metric, pos, 100)
        if team in table.index:
            r = table.loc[team]
            who = pos or "QB"
            t.add_row(f"{label} to {who}s", fmt_metric(metric, r["rate"]), fmt_metric(metric, table.attrs["league"]),
                      f"{ordinal(int(r['rank']))}/{len(table)}", f"{r['n_cur']:.0f}")
    console.print(t)
    t = Table(title="Scheme & personnel splits allowed", header_style="bold cyan")
    for c in ["Split", "Bucket", f"{team} allows", "League", f"{team} frequency", "Plays (this season)"]:
        t.add_column(c)
    for fam, metric, label, pos, dims in DEFENSE_VIEWS:
        for dim in dims:
            D = ps.buckets(fam, metric, dim, pos=pos, extra=f"defteam = '{team}'")
            L = ps.buckets(fam, metric, dim, pos=pos)
            nD = sum(s.n for s in D.values())
            for k, s in sorted(D.items(), key=lambda kv: -kv[1].n):
                if k not in L:
                    continue
                t.add_row(f"{label} to {pos or 'QB'}s / {DIM_LABELS[dim]}", bucket_label(dim, k),
                          fmt_metric(metric, s.rate), fmt_metric(metric, L[k].rate),
                          f"{s.n / nD:.0%}", f"{s.n_cur:.0f}")
    console.print(t)
    sit = situational(con, asof).get(team, {})
    if sit.get("rz_def"):
        rate, rank, lg, trips = sit["rz_def"]
        console.print(f"Red zone: {team} allows TDs on {rate:.0%} of {trips} opponent trips "
                      f"({ordinal(rank)}-stingiest, league {lg:.0%})")
    if sit.get("late_def"):
        epa, rank, n = sit["late_def"]
        console.print(f"Late game (4th quarter/OT): {epa:+.3f} EPA/play allowed ({ordinal(rank)}-best of {n})")
