"""Unit tests for grading, probability calibration, weather buckets and tuned-setting overrides."""

import json

import numpy as np
import pandas as pd
import pytest

from cajetan import config, probcal
from cajetan.backtest import proxy_line
from cajetan.calibration import Calibration
from cajetan.predlog import settle, summarize
from cajetan.weather import Conditions, temp_bucket, wind_bucket


@pytest.mark.parametrize("market,side,line,actual,expected", [
    ("rush_yds", "over", 79.5, 80, "won"),
    ("rush_yds", "under", 79.5, 80, "lost"),
    ("rec", "over", 5.0, 5, "push"),
    ("anytime_td", "yes", 0.5, 2, "won"),
    ("anytime_td", "yes", 0.5, 0, "lost"),
    ("ml", "win", None, -3, "lost"),
    ("ml", "win", None, 0, "push"),
    ("spread", "cover", -3.5, 4, "won"),
    ("spread", "cover", -3.0, 3, "push"),
    ("spread", "cover", 6.5, -6, "won"),
    ("game_total", "under", 47.5, 51, "lost"),
    ("rush_yds", "over", 79.5, None, None),
])
def test_settle(market, side, line, actual, expected):
    assert settle(market, side, line, actual) == expected


def test_proxy_line_is_half_point_below_average():
    assert proxy_line(64.2) == 64.5
    assert proxy_line(3.0) == 3.5
    assert proxy_line(0.4) == 0.5


def test_weather_buckets():
    assert wind_bucket(None) is None
    assert [wind_bucket(w) for w in (5, 12, 17, 25)] == ["calm", "breezy", "windy", "very_windy"]
    assert [temp_bucket(t) for t in (10, 35, 70)] == ["frigid", "cold", "mild"]


def test_calibration_weather_multiplier():
    cell = lambda m: [m, 500]
    calib = Calibration({"weather": {"pass_yds": {
        "indoors": cell(1.02),
        "wind": {"calm": cell(1.0), "breezy": cell(0.98), "windy": cell(0.93), "very_windy": cell(0.92)},
        "temp": {"frigid": cell(0.95), "cold": cell(0.96), "mild": cell(1.0)}}}})
    mult, _ = calib.weather("pass_yds", Conditions(20, 22, False, "forecast"))
    assert mult == pytest.approx(0.92 * 0.95)
    assert calib.weather("pass_yds", Conditions(None, None, True, "dome"))[0] == pytest.approx(1.02)
    assert calib.weather("rush_yds", Conditions(20, 22, False, "forecast")) is None


def test_inactive_calibration_is_identity_and_sides_are_symmetric():
    pc = probcal.ProbCalibration({"groups": {"yardage": {"active": False, "y": list(probcal.GRID)}}})
    assert pc.apply("rush_yds", "over", 0.62) == 0.62
    squash = list(0.5 + (probcal.GRID - 0.5) * 0.5)  # halve every distance from 50%
    pc = probcal.ProbCalibration({"groups": {"yardage": {"active": True, "y": squash}}})
    over, under = pc.apply("rush_yds", "over", 0.70), pc.apply("rush_yds", "under", 0.30)
    assert over == pytest.approx(0.60, abs=1e-6)
    assert under == pytest.approx(1 - over)


def test_probcal_fit_turns_on_for_overconfident_ratings():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.2, 0.8, 4000)
    truth = 0.5 + (p - 0.5) * 0.4  # ratings are 2.5x too far from 50%
    won = (rng.uniform(size=p.size) < truth).astype(int)
    rows = pd.DataFrame({"market": "rec_yds", "side": "over", "p": p, "won": won, "fold": np.arange(p.size) % 17})
    out = probcal.fit(rows, log=lambda m: None)["groups"]["yardage"]
    assert out["active"]
    assert out["cv_log_loss"] < out["raw_log_loss"]
    pc = probcal.ProbCalibration({"groups": {"yardage": out}})
    assert 0.55 < pc.apply("rec_yds", "over", 0.8) < 0.75  # truth is 0.62; shrinkage keeps part of the raw 0.8


def test_summarize_reports_brier_against_coin_flip():
    df = pd.DataFrame({"probability": [0.8, 0.7, 0.3, 0.6], "result": ["won", "won", "lost", "push"]})
    s = summarize(df)
    assert s["n"] == 3
    assert s["brier"] < s["brier_coin"]


def test_grades_are_saved_and_regraded_in_place():
    import duckdb

    from cajetan.predlog import load_grades, save_grades

    con = duckdb.connect()
    leg = {"leg_id": "abc-0", "parlay_id": "abc", "logged_at": "2026-10-20T12:00:00+00:00", "leg": "X over 49.5",
           "market": "rush_yds", "side": "over", "line": 49.5, "season": 2026, "week": 6, "probability": 0.6,
           "raw_probability": 0.6, "game_id": "g1"}
    pending = {**leg, "leg_id": "abc-1", "result": None, "actual": None}
    assert save_grades(con, pd.DataFrame([{**leg, "actual": 52.0, "result": "won"}, pending])) == (1, 0)
    assert list(load_grades(con)["leg_id"]) == ["abc-0"]  # pending legs aren't saved
    con.execute("UPDATE prediction_grades SET graded_at = '2026-10-21T09:00:00+00:00'")
    assert save_grades(con, pd.DataFrame([{**leg, "actual": 48.0, "result": "lost"}])) == (0, 1)
    again = load_grades(con)
    assert len(again) == 1 and again["result"][0] == "lost" and again["actual"][0] == 48.0
    assert again["graded_at"][0] == "2026-10-21T09:00:00+00:00"  # first grading time is kept


def test_tuning_file_overrides_config(tmp_path, monkeypatch):
    path = tmp_path / "tuning.json"
    path.write_text(json.dumps({"values": {"MARGIN_SD": 12.5, "ML_BLEND_BY_MARKET": {"rec": 0.4}, "NOT_A_SETTING": 1}}))
    monkeypatch.setattr(config, "TUNING_PATH", path)
    monkeypatch.delenv("CAJETAN_NO_TUNING", raising=False)
    for name in ("MARGIN_SD", "ML_BLEND_BY_MARKET"):
        monkeypatch.setattr(config, name, getattr(config, name))
    monkeypatch.setattr(config, "TUNED", {})
    config._load_tuning()
    assert config.MARGIN_SD == 12.5
    assert config.ML_BLEND_BY_MARKET == {"rec": 0.4}
    assert not hasattr(config, "NOT_A_SETTING")
