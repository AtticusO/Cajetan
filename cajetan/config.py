"""Paths and tunable constants used across Cajetan.

The constants below are defaults. `cajetan fit` estimates several of them from a
backtest and saves them to data/models/tuning.json, which overrides them on import.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("CAJETAN_HOME", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "cajetan.duckdb"
MODEL_DIR = DATA_DIR / "models"
CALIBRATION_PATH = MODEL_DIR / "calibration.json"
TUNING_PATH = MODEL_DIR / "tuning.json"
PROB_CALIBRATION_PATH = MODEL_DIR / "prob_calibration.json"
BACKTEST_PATH = DATA_DIR / "backtest.parquet"
PREDICTION_LOG = DATA_DIR / "predictions.jsonl"
WEATHER_CACHE = DATA_DIR / "weather_cache.json"

# FTN charting (play action, motion, blitz, box counts) starts in 2022, so play-by-play
# and the matchup engine start there too.
FIRST_SEASON = 2022
# Weekly stats, schedules, rosters, snaps, injuries and Next Gen Stats are also loaded
# from this season on, to give the ML model more history (Next Gen Stats start in 2016).
HISTORY_FIRST_SEASON = 2016

# Weight given to the previous season relative to the current one when
# pooling play-level samples. Everything older than one season is ignored
# by the matchup engine (the ML model still trains on every season).
PRIOR_SEASON_WEIGHT = 0.25

# Half-life (in games) of the recency decay applied to a player's game log.
RECENCY_HALF_LIFE = 4.0

# Share of the final projection that comes from the gradient-boosting model
# when one has been trained; the rest comes from the matchup engine. Markets
# where the model failed to beat the naive season-average baseline on the
# holdout season get the smaller weight. `cajetan fit` replaces these with a
# per-market weight (ML_BLEND_BY_MARKET) estimated from the backtest.
ML_BLEND_WEIGHT = 0.35
ML_BLEND_WEIGHT_WEAK = 0.15
ML_BLEND_BY_MARKET: dict[str, float] = {}

# Share of team-market projections taken from the betting-market consensus
# line stored in the nflverse schedule (when present).
MARKET_BLEND_WEIGHT = 0.5

# Standard deviations used for team markets (NFL historical values).
MARGIN_SD = 13.5
TOTAL_SD = 13.0
TEAM_POINTS_SD = 9.5

PARLAY_SIMULATIONS = 200_000

# Names `cajetan fit` may override, and the fitted values currently in effect.
TUNABLE = ("PRIOR_SEASON_WEIGHT", "RECENCY_HALF_LIFE", "ML_BLEND_BY_MARKET", "MARKET_BLEND_WEIGHT",
           "MARGIN_SD", "TOTAL_SD", "TEAM_POINTS_SD")
TUNED: dict = {}


def _load_tuning() -> None:
    if os.environ.get("CAJETAN_NO_TUNING") or not TUNING_PATH.exists():
        return
    try:
        values = json.loads(TUNING_PATH.read_text()).get("values", {})
    except (OSError, ValueError):
        return
    for name, value in values.items():
        if name in TUNABLE:
            globals()[name] = value
            TUNED[name] = value


_load_tuning()
