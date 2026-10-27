"""End-to-end checks against the local database (skipped until `cajetan update` has run)."""

import pytest

from cajetan import config

pytestmark = pytest.mark.skipif(not config.DB_PATH.exists(), reason="run `cajetan update` first")


@pytest.fixture(scope="module")
def con():
    from cajetan.db import connect

    c = connect(read_only=True)
    yield c
    c.close()


def _top_player(con, position, stat):
    return con.execute(
        f"""SELECT player_name FROM player_week WHERE season = (SELECT max(season) FROM player_week)
            AND position = '{position}' GROUP BY player_name HAVING count(*) >= 3
            ORDER BY avg({stat}) DESC LIMIT 1"""
    ).fetchone()[0]


def test_rates_player_and_team_legs(con):
    from cajetan import engine

    rb = _top_player(con, "RB", "rushing_yards")
    qb = _top_player(con, "QB", "passing_yards")
    legs, parlay = engine.rate(con, [f"{rb} o 60.5 rush yds", f"{qb} over 220.5 pass yds", f"{rb} anytime td"])
    for leg in legs:
        assert leg.error is None, leg.error
        assert 0 < leg.probability < 1
        assert leg.factors
    assert 0 < parlay.probability < 1


def test_lower_line_means_higher_over_confidence(con):
    from cajetan import engine

    rb = _top_player(con, "RB", "rushing_yards")
    legs, _ = engine.rate(con, [f"{rb} o 40.5 rush yds", f"{rb} o 120.5 rush yds"])
    assert legs[0].probability > legs[1].probability


def test_unknown_player_reports_error(con):
    from cajetan import engine

    legs, parlay = engine.rate(con, ["Notareal Playername o 50.5 rush yds"])
    assert legs[0].error
    assert parlay.probability == 0
