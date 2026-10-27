"""How each player market is built: volume x efficiency, which plays it comes from, how it is distributed."""

from __future__ import annotations

from dataclasses import dataclass

# Families of plays the efficiency side of a market is measured on. `pid` is the
# player-id column the family is keyed on; `pos_col` is the column used for
# position-specific defensive splits.
FAMILIES: dict[str, dict[str, str]] = {
    "rush": {"filter": "rush_attempt = 1", "pid": "rusher_player_id", "pos_col": "rusher_pos", "unit": "carry"},
    "target": {"filter": "pass_attempt = 1 AND sack = 0 AND receiver_player_id IS NOT NULL",
               "pid": "receiver_player_id", "pos_col": "receiver_pos", "unit": "target"},
    "dropback": {"filter": "pass_attempt = 1 AND sack = 0", "pid": "passer_player_id", "pos_col": "", "unit": "attempt"},
}


def plural(unit: str) -> str:
    return "carries" if unit == "carry" else f"{unit}s"


@dataclass(frozen=True)
class Market:
    key: str
    label: str
    stat: str  # SQL over player_week giving the per-game value
    volume: str | None  # player_week volume column
    family: str | None  # play family for efficiency
    metric: str | None  # SQL over plays giving the per-play value
    metric_label: str
    dist: str  # gamma / nbinom / poisson
    k: float  # shrinkage pseudo-sample for efficiency (plays)
    positions: tuple[str, ...]
    offense_dims: tuple[str, ...] = ()
    defense_dims: tuple[str, ...] = ()


_RUSH_OFF = ("run_dir", "formation", "personnel")
_RUSH_DEF = ("box",)
_PASS_OFF = ("personnel", "formation", "play_action", "motion")
_PASS_DEF = ("coverage", "blitz")

MARKETS: dict[str, Market] = {
    m.key: m
    for m in [
        Market("pass_yds", "passing yards", "passing_yards", "attempts", "dropback", "passing_yards",
               "yds/att", "gamma", 100, ("QB",), _PASS_OFF, _PASS_DEF),
        Market("pass_comp", "completions", "completions", "attempts", "dropback", "complete_pass",
               "comp %", "nbinom", 100, ("QB",), _PASS_OFF, _PASS_DEF),
        Market("pass_tds", "passing TDs", "passing_tds", "attempts", "dropback", "pass_touchdown",
               "TD/att", "nbinom", 400, ("QB",)),
        Market("pass_int", "interceptions", "passing_interceptions", "attempts", "dropback", "interception",
               "INT/att", "poisson", 500, ("QB",), (), _PASS_DEF),
        Market("pass_att", "pass attempts", "attempts", "attempts", None, None, "", "nbinom", 0, ("QB",)),
        Market("rush_yds", "rushing yards", "rushing_yards", "carries", "rush", "rushing_yards",
               "yds/carry", "gamma", 45, ("RB", "QB", "WR", "TE", "FB"), _RUSH_OFF, _RUSH_DEF),
        Market("rush_att", "rush attempts", "carries", "carries", None, None, "", "nbinom", 0,
               ("RB", "QB", "WR", "TE", "FB")),
        Market("rec", "receptions", "receptions", "targets", "target", "complete_pass",
               "catch %", "nbinom", 20, ("WR", "TE", "RB", "FB"), _PASS_OFF, _PASS_DEF),
        Market("rec_yds", "receiving yards", "receiving_yards", "targets", "target", "receiving_yards",
               "yds/target", "gamma", 30, ("WR", "TE", "RB", "FB"), _PASS_OFF, _PASS_DEF),
        Market("rush_rec_yds", "rush + rec yards", "rushing_yards + receiving_yards", None, None, None,
               "", "gamma", 0, ("RB", "WR", "TE", "QB", "FB")),
        Market("anytime_td", "anytime TD", "rushing_tds + receiving_tds", None, None, None,
               "", "poisson", 0, ("RB", "WR", "TE", "QB", "FB")),
    ]
}

DIM_LABELS: dict[str, str] = {
    "run_dir": "run direction", "formation": "formation", "personnel": "personnel",
    "box": "box count", "play_action": "play action", "motion": "pre-snap motion",
    "coverage": "man/zone coverage", "blitz": "blitz", "qb_run_type": "QB run type",
}

BUCKET_LABELS: dict[str, str] = {
    "outside": "outside runs (end/edge: outside-zone, stretch & toss proxy)",
    "off_tackle": "off-tackle runs",
    "inside": "inside runs (guard/middle: inside-zone, duo & power proxy)",
    "shotgun": "shotgun", "under_center": "under center", "pistol": "pistol",
    "light_box": "light boxes (6 or fewer)", "neutral_box": "7-man boxes", "heavy_box": "stacked boxes (8+)",
    "play_action": "play action", "no_play_action": "non-play-action",
    "motion": "motion at the snap", "no_motion": "no motion",
    "man": "man coverage", "zone": "zone coverage", "blitz": "blitzes", "no_blitz": "non-blitz",
    "designed": "designed QB runs", "scramble": "scrambles", "sneak": "QB sneaks",
}


def bucket_label(dim: str, bucket: str) -> str:
    if dim == "personnel":
        return f"{bucket} personnel"
    return BUCKET_LABELS.get(bucket, bucket)
