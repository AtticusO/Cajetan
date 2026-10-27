"""Command-line interface: `cajetan update | train | backtest | fit | grade | rate | player | defense | status | web`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from cajetan import config
from cajetan.db import DataMissingError, connect, table_exists

app = typer.Typer(add_completion=False, help="Cajetan: statistical confidence ratings for NFL parlays.",
                  no_args_is_help=True)
console = Console()


def _ro():
    try:
        return connect(read_only=True)
    except DataMissingError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command()
def update(
    full: Annotated[bool, typer.Option(help="Reload every season instead of just the current one.")] = False,
    seasons: Annotated[Optional[str], typer.Option(help="Comma-separated seasons to (re)load, e.g. 2025,2026.")] = None,
) -> None:
    """Download nflverse data into the local database (first run loads 2022-present)."""
    from cajetan import ingest

    con = connect()
    explicit = [int(s) for s in seasons.split(",")] if seasons else None
    todo = ingest.seasons_to_load(con, full, explicit)
    console.print(f"Loading seasons: {', '.join(map(str, todo))}")
    counts = ingest.update(con, todo, log=lambda m: console.print(f"  {m}"))
    for table, by_season in counts.items():
        console.print(f"  {table:14s} " + ", ".join(f"{s}: {n:,}" for s, n in by_season.items()))
    console.print("[green]Done.[/green] Run `cajetan train` to refresh the models and calibration.")


@app.command()
def train(
    tune: Annotated[bool, typer.Option(help="Also search the ML model settings across held-out seasons (~15 min).")] = False,
) -> None:
    """Fit the historical ML models and the dispersion/correlation/weather/injury calibration."""
    from cajetan import calibration
    from cajetan.analysis import ml

    con = _ro()
    data = calibration.compute(con, log=lambda m: console.print(f"  {m}"))
    calibration.save(data)
    console.print(f"  saved calibration: {len(data['cv'])} dispersion values, "
                  f"{len(data['corr_pair'])} player-pair correlations, {len(data['corr_env'])} game correlations")
    ml.train(con, log=lambda m: console.print(f"  {m}"), tune=tune)
    for old in config.MODEL_DIR.glob("ml_*_through_*"):
        old.unlink()  # backtest models are rebuilt from the new settings on the next backtest
    console.print(f"[green]Done.[/green] Models saved in {config.MODEL_DIR}")


@app.command()
def backtest(
    seasons: Annotated[Optional[str], typer.Option(help="Comma-separated seasons (default: last complete + current).")] = None,
    weeks: Annotated[Optional[str], typer.Option(help="Weeks to include, e.g. 3-18 (default: week 3 on).")] = None,
    workers: Annotated[Optional[int], typer.Option(help="Parallel worker processes (default: cores - 1).")] = None,
    max_legs: Annotated[Optional[int], typer.Option(help="Cap player legs per season (random sample).")] = None,
) -> None:
    """Rate past games leak-free and score the ratings; saves data/backtest.parquet for `cajetan fit`."""
    from cajetan import backtest as bt
    from cajetan import tuning

    con = _ro()
    if seasons:
        todo = [int(s) for s in seasons.split(",")]
    else:
        cur = con.execute("SELECT max(season) FROM schedules").fetchone()[0]
        todo = [cur - 1, cur]
    wk = None
    if weeks:
        lo, _, hi = weeks.partition("-")
        wk = list(range(int(lo), int(hi or lo) + 1))
    df = bt.run(con, todo, wk, workers, max_legs, log=lambda m: console.print(f"  {m}"))
    if df.empty:
        console.print("[yellow]No completed games to backtest.[/yellow]")
        return
    _print_scores(tuning.score(df), f"Backtest: {len(df):,} legs from {', '.join(map(str, todo))} (raw ratings)")
    console.print(f"Saved {config.BACKTEST_PATH}. Run `cajetan fit` to refit the weights and calibration.")


@app.command()
def fit() -> None:
    """Fit blend weights, team outcome spreads, game-log weighting and the probability calibration."""
    from cajetan import backtest as bt
    from cajetan import predlog, tuning

    df = bt.load()
    if df.empty:
        console.print("[red]No backtest yet: run `cajetan backtest` first.[/red]")
        raise typer.Exit(1)
    con = _ro()
    graded = predlog.load_grades(con)
    res = tuning.fit(con, df, graded, log=lambda m: console.print(f"  {m}"))
    console.print("[green]Saved[/green] " + ", ".join(
        f"{k} = {tuning.describe_value(k, v)}" for k, v in res["values"].items()))
    console.print(f"Settings in {config.TUNING_PATH}; calibration in {config.PROB_CALIBRATION_PATH}. "
                  "They take effect on the next rating.")


@app.command()
def grade(
    show: Annotated[int, typer.Option(help="How many of the most recent graded legs to list.")] = 15,
) -> None:
    """Settle logged ratings whose games are final, save them in the database, and report accuracy."""
    import duckdb
    from rich.table import Table

    from cajetan import predlog

    try:
        con = connect()
    except duckdb.IOException:
        console.print("[red]The database is in use (is `cajetan web` or another command running?). "
                      "Stop it and run `cajetan grade` again.[/red]")
        raise typer.Exit(1)
    graded = predlog.grade(con)
    if not graded.empty:
        new, changed = predlog.save_grades(con, graded)
        pending = int(graded["result"].isna().sum())
        console.print(f"Graded {new:,} new legs" + (f", {changed:,} re-graded after stat corrections" if changed else "")
                      + f"; {pending:,} still pending.")
    done = predlog.load_grades(con)
    con.close()
    if done.empty:
        console.print(f"No graded predictions yet. Ratings are logged automatically to {config.PREDICTION_LOG} "
                      "and graded here once their games are final.")
        return
    counts = done["result"].value_counts()
    console.print(f"Saved history: {len(done):,} legs ({', '.join(f'{n:,} {r}' for r, n in counts.items())})")
    s = predlog.summarize(done)
    if s["n"]:
        console.print(f"Decided legs {s['n']:,}: hit rate {s['hit_rate']:.1%} vs average rating {s['avg_prob']:.1%}; "
                      f"Brier {s['brier']:.4f} (coin flip {s['brier_coin']:.4f}), log loss {s['log_loss']:.4f}")
        t = Table(header_style="bold cyan")
        for c in ("Rated", "Legs", "Avg rating", "Hit rate"):
            t.add_column(c)
        for row in s["table"]:
            t.add_row(row["bin"], str(row["n"]), f"{row['predicted']:.1%}", f"{row['hit_rate']:.1%}")
        console.print(t)
        priced = done[done["odds"].notna() & done["result"].isin(["won", "lost"])]
        if len(priced):
            from cajetan.parsing import american_to_decimal

            profit = sum(american_to_decimal(int(o)) - 1 if r == "won" else -1
                         for o, r in zip(priced["odds"], priced["result"]))
            console.print(f"Legs with odds: {len(priced)}; 1-unit flat bets would return {profit:+.2f} units "
                          f"({profit / len(priced):+.1%} per bet)")
    recent = done.tail(show)
    for r in recent.itertuples():
        actual = "did not play" if r.result == "void" else f"actual {r.actual:g}"
        console.print(f"  {r.result:5s} {r.probability:5.0%}  {r.leg}  (wk {r.week}, {actual})")


@app.command()
def visits(
    limit: Annotated[int, typer.Option(help="How many recent connections to list.")] = 30,
    ip: Annotated[Optional[str], typer.Option(help="Only show connections from this IP.")] = None,
    by_ip: Annotated[bool, typer.Option("--by-ip", help="Summarise one row per visitor IP instead.")] = False,
) -> None:
    """Show who connected to the web UI (IP, time, legs entered), from data/access_log.sqlite."""
    import json

    from rich.table import Table

    from cajetan import access_log

    if by_ip:
        rows = access_log.visitors()
        t = Table(header_style="bold cyan", title="Visitors")
        for c in ("IP", "Country", "First seen (UTC)", "Last seen (UTC)", "Page loads", "Ratings"):
            t.add_column(c)
        for r in rows:
            t.add_row(r["ip"], r["country"] or "-", r["first_seen"], r["last_seen"], str(r["page_loads"]), str(r["ratings"]))
    else:
        rows = access_log.recent(limit, ip)
        t = Table(header_style="bold cyan", title=f"Last {len(rows)} connections")
        for c in ("Time (UTC)", "IP", "Country", "Event", "Legs", "Odds", "Week"):
            t.add_column(c)
        for r in rows:
            legs = "\n".join(json.loads(r["legs"])) if r["legs"] else "-"
            ip_label = r["ip"] + ("" if r["via_tunnel"] else " (local)")
            event = r["event"] + ("" if r["status"] in (None, 200) else f" ({r['status']})")
            t.add_row(r["time_utc"], ip_label, r["country"] or "-", event, legs, r["odds"] or "-", r["week"] or "-")
    if not rows:
        console.print(f"No connections logged yet ({access_log.PATH}).")
        return
    console.print(t)


def _print_scores(df, title: str) -> None:
    from rich.table import Table

    t = Table(title=title, header_style="bold cyan")
    for c in ("Market", "Legs", "Hit rate", "Avg rating", "Brier", "Brier (base)", "Log loss", "Log loss (base)"):
        t.add_column(c, justify="right" if c != "Market" else "left")
    for r in df.itertuples():
        t.add_row(r.market, f"{r.legs:,}", f"{r.hit_rate:.1%}", f"{r.avg_prob:.1%}", f"{r.brier:.4f}",
                  f"{r.brier_base:.4f}", f"{r.log_loss:.4f}", f"{r.log_loss_base:.4f}")
    console.print(t)
    console.print("[dim]Base = always predicting the overall hit rate for that market; lower is better.[/dim]")


@app.command()
def rate(
    legs: Annotated[Optional[list[str]], typer.Argument(help="Legs, each quoted: \"Josh Allen o 34.5 rush yds\".")] = None,
    file: Annotated[Optional[Path], typer.Option("--file", "-f", help="Text file with one leg per line.")] = None,
    week: Annotated[Optional[int], typer.Option(help="Week to rate (default: each team's next game).")] = None,
    odds: Annotated[Optional[str], typer.Option(help="Sportsbook odds for the whole parlay, e.g. +650.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
    reasons: Annotated[int, typer.Option(help="How many reasons to show per leg.")] = 5,
) -> None:
    """Rate each leg and the overall parlay."""
    from cajetan import engine, report

    raws = list(legs or [])
    if file:
        raws += [ln.strip() for ln in file.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not raws:
        console.print("[red]Give at least one leg, e.g. cajetan rate \"Josh Allen o 34.5 rush yds\"[/red]")
        raise typer.Exit(1)
    con = _ro()
    legs_out, parlay = engine.rate(con, raws, week, int(odds) if odds else None, log_source="cli")
    if as_json:
        print(report.to_json(legs_out, parlay))
        return
    report.render(console, legs_out, parlay, _data_note(con), reasons)


@app.command()
def player(name: str, week: Annotated[Optional[int], typer.Option()] = None) -> None:
    """Show a player's per-game stats and scheme/personnel splits."""
    from cajetan.profiles import player_profile

    player_profile(_ro(), console, name, week)


@app.command()
def defense(team: str, week: Annotated[Optional[int], typer.Option()] = None) -> None:
    """Show what a defense allows by position, scheme, personnel and situation."""
    from cajetan.profiles import defense_profile
    from cajetan.teams import team_abbr

    abbr = team_abbr(team)
    if not abbr:
        console.print(f"[red]Unknown team '{team}'[/red]")
        raise typer.Exit(1)
    defense_profile(_ro(), console, abbr, week)


@app.command()
def status() -> None:
    """Show what data and models are available locally."""
    from cajetan.analysis import ml

    con = _ro()
    console.print(_data_note(con))
    for t in ("pbp", "plays", "player_week", "schedules", "ftn", "participation"):
        if table_exists(con, t):
            rows = con.execute(f"SELECT season, count(*) FROM {t} GROUP BY 1 ORDER BY 1").fetchall()
            console.print(f"  {t:14s} " + ", ".join(f"{s}: {n:,}" for s, n in rows))
    for t in ("injuries", "snap_counts", "ngs_receiving"):
        if table_exists(con, t):
            rows = con.execute(f"SELECT min(season), max(season), count(*) FROM {t}").fetchone()
            console.print(f"  {t:14s} {rows[0]}-{rows[1]}: {rows[2]:,} rows")
    models, meta = ml._load()
    if meta:
        for m, info in meta.items():
            folds = ", ".join(map(str, info.get("folds") or [info.get("holdout_season")]))
            console.print(f"  model {m:13s} {info['rows']:,} rows ({info['seasons']}), validated MAE {info['mae']:.2f} "
                          f"(naive {info['naive_mae']:.2f}) on {folds}")
    else:
        console.print("  [yellow]No ML models yet: run `cajetan train`.[/yellow]")
    if not config.CALIBRATION_PATH.exists():
        console.print("  [yellow]No calibration yet: run `cajetan train` (defaults are used meanwhile).[/yellow]")
    if config.TUNED:
        from cajetan.tuning import describe_value

        console.print("  fitted settings: " + "; ".join(f"{k} {describe_value(k, v)}" for k, v in config.TUNED.items()))
    else:
        console.print("  [dim]No fitted settings: run `cajetan backtest` then `cajetan fit`.[/dim]")
    from cajetan.probcal import ProbCalibration

    groups = ProbCalibration.load().data.get("groups", {})
    if groups:
        console.print("  probability calibration: " + ", ".join(
            f"{g} {'on' if e.get('active') else 'off'}" for g, e in groups.items()))


@app.command()
def web(
    port: Annotated[int, typer.Option(help="Port to serve on.")] = 8765,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Open the page in a browser.")] = True,
) -> None:
    """Start the web UI: enter legs in the browser and see the ratings."""
    import webbrowser

    from cajetan.web import serve

    server = serve(port=port)
    url = f"http://127.0.0.1:{port}/"
    console.print(f"Cajetan web UI running at [bold]{url}[/bold]  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _data_note(con) -> str:
    season, week = con.execute(
        "SELECT season, max(week) FROM plays WHERE season = (SELECT max(season) FROM plays) GROUP BY 1").fetchone()
    updated = con.execute("SELECT value FROM meta WHERE key = 'last_update'").fetchone() if table_exists(con, "meta") else None
    return f"play-by-play through {season} week {week}" + (f", updated {updated[0]}" if updated else "")


if __name__ == "__main__":
    app()
