import pytest

from cajetan.analysis.distributions import count_outcome, gamma_outcome, normal_outcome
from cajetan.analysis.explain import Factor, LegResult
from cajetan.analysis.matchup import shrink
from cajetan.analysis.parlay import combine
from cajetan.analysis.player_eval import cap_factors, product
from cajetan.calibration import Calibration
from cajetan.parsing import Leg


def test_shrink_moves_toward_prior():
    assert shrink(10, 2, 4.0, 0) == 5.0
    assert shrink(10, 2, 4.0, 1e9) == pytest.approx(4.0)
    assert 4.0 < shrink(10, 2, 4.0, 2) < 5.0


def test_gamma_outcome_half_line_has_no_push():
    out = gamma_outcome(80, 0.5, 79.5)
    assert out.p_push == pytest.approx(0, abs=1e-12)
    assert out.p_over + out.p_under == pytest.approx(1)


def test_gamma_outcome_whole_line_has_push_and_symmetry_of_sides():
    out = gamma_outcome(80, 0.5, 80)
    assert out.p_push > 0
    assert out.p_side("over") + out.p_side("under") == pytest.approx(1)


def test_higher_mean_raises_over_probability():
    assert gamma_outcome(100, 0.5, 80.5).p_over > gamma_outcome(80, 0.5, 80.5).p_over


def test_count_outcome_poisson_matches_closed_form():
    out = count_outcome(1.0, 0.5, None)  # P(X >= 1) for Poisson(1)
    assert out.p_over == pytest.approx(1 - 0.36787944, abs=1e-6)


def test_negative_binomial_is_wider_than_poisson():
    nb = count_outcome(5, 8.5, 5.0)
    po = count_outcome(5, 8.5, None)
    assert nb.p_over > po.p_over


def test_normal_outcome_spread():
    # Favored by 7 covering -3.5 should be well above 50%.
    assert normal_outcome(7, 13.5, 3.5).p_side("over") > 0.55


def test_cap_factors_bounds_product():
    fs = [Factor("a", 1.3, ""), Factor("b", 1.2, "")]
    cap_factors(fs, 0.8, 1.25)
    assert product(fs) == pytest.approx(1.25)


def _leg(p, game, team, market="rush_yds", pos="RB", pid="x", side="over"):
    return LegResult(leg=Leg("raw", "player", "P", market, side, 50.5), description="", market_label="",
                     side=side, line=50.5, projection=55, probability=p, market=market, game_id=game,
                     team=team, opponent="OPP", position=pos, player_id=pid)


def test_parlay_independent_legs_multiply():
    legs = [_leg(0.6, "g1", "A", pid="a"), _leg(0.5, "g2", "B", pid="b")]
    res = combine(legs, Calibration())
    assert res.probability == pytest.approx(0.3)
    assert res.independent_probability == pytest.approx(0.3)


def test_parlay_positive_correlation_raises_joint_probability():
    qb = _leg(0.5, "g1", "A", market="pass_yds", pos="QB", pid="qb")
    wr = _leg(0.5, "g1", "A", market="rec_yds", pos="WR", pid="wr")
    res = combine([qb, wr], Calibration())
    assert res.probability > 0.25 + 0.03


def test_parlay_over_under_flip_correlation():
    qb = _leg(0.5, "g1", "A", market="pass_yds", pos="QB", pid="qb")
    wr = _leg(0.5, "g1", "A", market="rec_yds", pos="WR", pid="wr", side="under")
    res = combine([qb, wr], Calibration())
    assert res.probability < 0.25 - 0.03
