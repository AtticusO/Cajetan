import pytest

from cajetan.parsing import (LegParseError, american_to_decimal, american_to_prob, decimal_to_american,
                             parse_leg)


@pytest.mark.parametrize("raw, subject, market, side, line, odds", [
    ("Josh Allen o 34.5 rush yds", "Josh Allen", "rush_yds", "over", 34.5, None),
    ("Josh Allen over 34.5 rushing yards -115", "Josh Allen", "rush_yds", "over", 34.5, -115),
    ("Ja'Marr Chase u6.5 receptions", "Ja'Marr Chase", "rec", "under", 6.5, None),
    ("Travis Kelce under 55.5 rec yds (+105)", "Travis Kelce", "rec_yds", "under", 55.5, 105),
    ("Jared Goff 250+ pass yds", "Jared Goff", "pass_yds", "over", 249.5, None),
    ("Jalen Hurts 2+ passing tds", "Jalen Hurts", "pass_tds", "over", 1.5, None),
    ("Bijan Robinson o 119.5 rush + rec yds", "Bijan Robinson", "rush_rec_yds", "over", 119.5, None),
    ("Saquon Barkley over 21.5 carries", "Saquon Barkley", "rush_att", "over", 21.5, None),
    ("Patrick Mahomes over 24.5 completions", "Patrick Mahomes", "pass_comp", "over", 24.5, None),
    ("Patrick Mahomes under 0.5 ints", "Patrick Mahomes", "pass_int", "under", 0.5, None),
    ("Kenneth Walker III o 89.5 rush yds", "Kenneth Walker III", "rush_yds", "over", 89.5, None),
])
def test_player_props(raw, subject, market, side, line, odds):
    leg = parse_leg(raw)
    assert (leg.kind, leg.subject, leg.market, leg.side, leg.line, leg.odds) == ("player", subject, market, side, line, odds)


@pytest.mark.parametrize("raw", ["Jahmyr Gibbs anytime td", "Jahmyr Gibbs ATD +120", "Jahmyr Gibbs to score a touchdown"])
def test_anytime_td(raw):
    leg = parse_leg(raw)
    assert (leg.subject, leg.market, leg.side) == ("Jahmyr Gibbs", "anytime_td", "yes")


@pytest.mark.parametrize("raw, team, market, side, line, odds, opp", [
    ("BUF ML", "BUF", "ml", "win", None, None, None),
    ("Bills moneyline -150", "BUF", "ml", "win", None, -150, None),
    ("Kansas City Chiefs", "KC", "ml", "win", None, None, None),
    ("DET -3.5", "DET", "spread", "cover", -3.5, None, None),
    ("Lions +3 -110", "DET", "spread", "cover", 3.0, -110, None),
    ("BUF/MIA o 47.5", "BUF", "game_total", "over", 47.5, None, "MIA"),
    ("Packers @ Lions under 52.5 total", "GB", "game_total", "under", 52.5, None, "DET"),
    ("BUF team total over 26.5", "BUF", "team_total", "over", 26.5, None, None),
])
def test_team_markets(raw, team, market, side, line, odds, opp):
    leg = parse_leg(raw)
    assert (leg.kind, leg.subject, leg.market, leg.side, leg.line, leg.odds, leg.opponent) == \
        ("team", team, market, side, line, odds, opp)


@pytest.mark.parametrize("raw", ["", "Josh Allen rush yds", "Josh Allen o 30.5 tackles"])
def test_bad_legs(raw):
    with pytest.raises(LegParseError):
        parse_leg(raw)


def test_odds_conversions():
    assert american_to_prob(-110) == pytest.approx(0.5238, abs=1e-4)
    assert american_to_prob(150) == pytest.approx(0.4)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    assert decimal_to_american(2.5) == 150
    assert decimal_to_american(1.5) == -200
