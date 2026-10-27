"""Turn free-text betting legs ("Josh Allen o 34.5 rush yds -115") into structured legs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from cajetan.teams import team_abbr


class LegParseError(ValueError):
    pass


PLAYER_MARKETS: dict[str, str] = {
    "pass_yds": "passing yards",
    "pass_tds": "passing TDs",
    "pass_att": "pass attempts",
    "pass_comp": "completions",
    "pass_int": "interceptions",
    "rush_yds": "rushing yards",
    "rush_att": "rush attempts",
    "rec": "receptions",
    "rec_yds": "receiving yards",
    "rush_rec_yds": "rush + rec yards",
    "anytime_td": "anytime TD",
}

TEAM_MARKETS: dict[str, str] = {
    "ml": "moneyline",
    "spread": "spread",
    "game_total": "game total",
    "team_total": "team total",
}

# Checked in order, so composite markets come before their parts.
_MARKET_PATTERNS: list[tuple[str, str]] = [
    ("rush_rec_yds", r"(rush(ing)?\s*(\+|&|and|/)?\s*rec(eiving)?|rec(eiving)?\s*(\+|&|and|/)?\s*rush(ing)?|scrimmage)\s*(yards|yds|yd)?"),
    ("pass_yds", r"pass(ing)?\s*(yards|yds|yd)|\bpy\b"),
    ("pass_tds", r"pass(ing)?\s*(touchdowns|tds|td)|\bptds?\b"),
    ("pass_att", r"pass(ing)?\s*(attempts|atts|att)|\battempts\b"),
    ("pass_comp", r"(pass(ing)?\s*)?(completions|comps|comp|cmp)\b"),
    ("pass_int", r"(interceptions|ints|int)\b(\s*thrown)?"),
    ("rush_yds", r"rush(ing)?\s*(yards|yds|yd)|\bry\b"),
    ("rush_att", r"rush(ing)?\s*(attempts|atts|att)|carries"),
    ("rec_yds", r"rec(eiving)?\s*(yards|yds|yd)"),
    ("rec", r"receptions|catches|recs\b|\brec\b"),
]
_ANYTIME = r"(anytime\s*(td|touchdown)(\s*scorer)?|\batd\b|anytime\s*scorer|to\s*score(\s*a)?(\s*(td|touchdown))?|td\s*scorer|\banytime\b)"
_ODDS = re.compile(r"(?:^|[\s(@])\s*([+-]\d{3,5})\s*\)?\s*$")
_SIDE = re.compile(r"(?:^|\s)(over|under|o|u)\s*(\d+(?:\.\d+)?)(?=\s|$)")
_PLUS = re.compile(r"(?:^|\s)(\d+(?:\.\d+)?)\s*\+(?=\s|$|[a-z])")
_TOTAL = re.compile(r"^(.+?)\s*(?:/|@|\bvs\.?|\bv\.?|\bat\b)\s*(.+?)\s+(over|under|o|u)\s*(\d+(?:\.\d+)?)\s*(total|pts|points)?$")


@dataclass
class Leg:
    raw: str
    kind: str  # "player" or "team"
    subject: str  # player name as typed, or team abbreviation
    market: str
    side: str  # over / under / yes / win / cover
    line: float | None
    odds: int | None = None
    opponent: str | None = None  # second team for game totals

    @property
    def market_label(self) -> str:
        return PLAYER_MARKETS.get(self.market) or TEAM_MARKETS[self.market]

    def describe(self) -> str:
        if self.market == "anytime_td":
            return f"{self.subject} anytime TD"
        if self.market == "ml":
            return f"{self.subject} moneyline"
        if self.market == "spread":
            return f"{self.subject} {self.line:+g}"
        if self.market == "game_total":
            return f"{self.subject}/{self.opponent} {self.side} {self.line:g}"
        if self.market == "team_total":
            return f"{self.subject} team total {self.side} {self.line:g}"
        return f"{self.subject} {self.side} {self.line:g} {self.market_label}"


def _clean(text: str) -> str:
    return " ".join(text.replace(",", " ").split())


def _leading_team(text: str) -> tuple[str, str] | None:
    """Longest prefix of `text` that names a team -> (abbr, rest)."""
    words = text.split()
    for n in range(min(4, len(words)), 0, -1):
        abbr = team_abbr(" ".join(words[:n]))
        if abbr:
            return abbr, " ".join(words[n:])
    return None


def parse_leg(raw: str) -> Leg:
    text = _clean(raw)
    if not text:
        raise LegParseError("empty leg")

    odds = None
    m = _ODDS.search(text)
    if m:
        odds = int(m.group(1))
        text = text[: m.start()].strip()
    s = text.lower().replace("’", "'")

    # Game total: "BUF/MIA o 47.5", "Bills @ Dolphins under 44"
    m = _TOTAL.match(s)
    if m and team_abbr(m.group(1)) and team_abbr(m.group(2)):
        return Leg(raw, "team", team_abbr(m.group(1)), "game_total",
                   "over" if m.group(3).startswith("o") else "under", float(m.group(4)), odds,
                   opponent=team_abbr(m.group(2)))

    team = _leading_team(s)
    if team:
        abbr, rest = team
        rest = rest.strip()
        if rest in ("", "ml", "moneyline", "money line", "to win", "win", "straight up", "su"):
            return Leg(raw, "team", abbr, "ml", "win", None, odds)
        if rest in ("pk", "pick", "pickem", "pick'em"):
            return Leg(raw, "team", abbr, "spread", "cover", 0.0, odds)
        sm = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*(spread|ats)?", rest)
        if sm:
            return Leg(raw, "team", abbr, "spread", "cover", float(sm.group(1)), odds)
        tm = re.fullmatch(r"(team\s*total|tt)?\s*(over|under|o|u)\s*(\d+(?:\.\d+)?)\s*(team\s*total|tt|pts|points)?", rest)
        if tm and (tm.group(1) or tm.group(4)):
            return Leg(raw, "team", abbr, "team_total",
                       "over" if tm.group(2).startswith("o") else "under", float(tm.group(3)), odds)

    # Player props
    am = re.search(_ANYTIME, s)
    if am:
        subject = _clean(s[: am.start()] + " " + s[am.end():])
        return Leg(raw, "player", _title(subject, raw), "anytime_td", "yes", 0.5, odds)

    market = None
    for key, pattern in _MARKET_PATTERNS:
        mm = re.search(pattern, s)
        if mm:
            market = key
            s = s[: mm.start()] + " " + s[mm.end():]
            break
    if market is None:
        raise LegParseError(
            f"couldn't find a market in '{raw}'. Try e.g. 'Josh Allen o 34.5 rush yds', "
            "'Bijan Robinson anytime td', 'BUF -3.5', 'BUF ML' or 'BUF/MIA o 47.5'."
        )

    side_m = _SIDE.search(s)
    if side_m:
        side = "over" if side_m.group(1).startswith("o") else "under"
        line = float(side_m.group(2))
        s = s[: side_m.start()] + " " + s[side_m.end():]
    else:
        plus = _PLUS.search(s)
        if not plus:
            raise LegParseError(f"couldn't find an over/under line in '{raw}' (e.g. 'o 64.5', 'under 5.5', '5+').")
        side, line = "over", float(plus.group(1)) - 0.5
        s = s[: plus.start()] + " " + s[plus.end():]

    subject = _clean(s)
    if not subject:
        raise LegParseError(f"couldn't find a player name in '{raw}'.")
    return Leg(raw, "player", _title(subject, raw), market, side, line, odds)


def _title(subject: str, raw: str) -> str:
    """Recover the player's name with its original capitalisation where possible."""
    idx = raw.lower().find(subject)
    return raw[idx: idx + len(subject)] if idx >= 0 else subject.title()


def american_to_prob(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def american_to_decimal(odds: int) -> float:
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def decimal_to_american(dec: float) -> int:
    return round((dec - 1) * 100) if dec >= 2 else round(-100 / (dec - 1))
