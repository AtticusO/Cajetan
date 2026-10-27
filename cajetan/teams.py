"""Team abbreviations (nflverse style) and the names people type for them."""

from __future__ import annotations

TEAMS: dict[str, tuple[str, ...]] = {
    "ARI": ("arizona", "cardinals", "cards", "arizona cardinals"),
    "ATL": ("atlanta", "falcons", "atlanta falcons"),
    "BAL": ("baltimore", "ravens", "baltimore ravens"),
    "BUF": ("buffalo", "bills", "buffalo bills"),
    "CAR": ("carolina", "panthers", "carolina panthers"),
    "CHI": ("chicago", "bears", "chicago bears"),
    "CIN": ("cincinnati", "bengals", "cincinnati bengals"),
    "CLE": ("cleveland", "browns", "cleveland browns"),
    "DAL": ("dallas", "cowboys", "dallas cowboys"),
    "DEN": ("denver", "broncos", "denver broncos"),
    "DET": ("detroit", "lions", "detroit lions"),
    "GB": ("green bay", "packers", "green bay packers", "gnb"),
    "HOU": ("houston", "texans", "houston texans"),
    "IND": ("indianapolis", "colts", "indianapolis colts"),
    "JAX": ("jacksonville", "jaguars", "jags", "jacksonville jaguars", "jac"),
    "KC": ("kansas city", "chiefs", "kansas city chiefs", "kan"),
    "LA": ("rams", "los angeles rams", "la rams", "lar"),
    "LAC": ("chargers", "los angeles chargers", "la chargers"),
    "LV": ("las vegas", "raiders", "las vegas raiders", "lvr", "oak"),
    "MIA": ("miami", "dolphins", "miami dolphins"),
    "MIN": ("minnesota", "vikings", "minnesota vikings"),
    "NE": ("new england", "patriots", "pats", "new england patriots", "nwe"),
    "NO": ("new orleans", "saints", "new orleans saints", "nor"),
    "NYG": ("giants", "new york giants", "ny giants"),
    "NYJ": ("jets", "new york jets", "ny jets"),
    "PHI": ("philadelphia", "eagles", "philly", "philadelphia eagles"),
    "PIT": ("pittsburgh", "steelers", "pittsburgh steelers"),
    "SEA": ("seattle", "seahawks", "seattle seahawks"),
    "SF": ("san francisco", "49ers", "niners", "san francisco 49ers", "sfo"),
    "TB": ("tampa bay", "tampa", "buccaneers", "bucs", "tampa bay buccaneers", "tam"),
    "TEN": ("tennessee", "titans", "tennessee titans"),
    "WAS": ("washington", "commanders", "washington commanders", "wsh"),
}

_LOOKUP: dict[str, str] = {}
for abbr, names in TEAMS.items():
    _LOOKUP[abbr.lower()] = abbr
    for n in names:
        _LOOKUP[n] = abbr


def team_abbr(text: str) -> str | None:
    """Return the nflverse abbreviation for a team name/abbreviation, or None."""
    return _LOOKUP.get(" ".join(text.lower().replace(".", "").split()))
