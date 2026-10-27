"""Terminal (Rich) and JSON rendering of leg and parlay ratings."""

from __future__ import annotations

import json

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cajetan.analysis.explain import LegResult, ParlayResult
from cajetan.parsing import decimal_to_american


def _color(conf: int) -> str:
    return "bold green" if conf >= 65 else "green" if conf >= 55 else "yellow" if conf >= 45 else "red"


def _fmt_proj(leg: LegResult) -> str:
    if leg.projection is None:
        return "-"
    if leg.market == "anytime_td":
        return f"{leg.projection:.2f} TDs"
    if leg.market in ("ml", "spread"):
        return f"{leg.projection:+.1f} margin"
    return f"{leg.projection:.1f}"


def render(console: Console, legs: list[LegResult], parlay: ParlayResult, data_note: str, reasons: int = 5) -> None:
    console.print(Text(f"Cajetan parlay rating  -  {data_note}", style="bold"))
    table = Table(show_lines=False, header_style="bold cyan")
    for col, kw in [("#", {"justify": "right"}), ("Leg", {}), ("Matchup", {}), ("Proj", {"justify": "right"}),
                    ("Confidence", {"justify": "right"}), ("Data", {"justify": "center"}), ("Verdict", {})]:
        table.add_column(col, **kw)
    for i, leg in enumerate(legs, 1):
        if leg.error:
            table.add_row(str(i), leg.leg.raw if hasattr(leg.leg, "raw") else leg.description, "-", "-",
                          Text("n/a", style="dim"), "-", Text("error", style="red"))
            continue
        table.add_row(str(i), leg.leg.describe(), leg.description, _fmt_proj(leg),
                      Text(f"{leg.confidence}%", style=_color(leg.confidence)), leg.grade, leg.verdict)
    console.print(table)

    for i, leg in enumerate(legs, 1):
        if leg.error:
            console.print(Panel(Text(leg.error, style="red"), title=f"Leg {i}: {leg.description}", title_align="left"))
            continue
        lines: list[Text] = [Text(leg.summary, style="bold")]
        why = _why(leg)
        if why:
            lines.append(Text(why, style="italic"))
        for text, sign in leg.reasons(reasons):
            mark, style = {1: ("[+] ", "green"), -1: ("[-] ", "red"), 0: (" .  ", "dim")}[sign]
            lines.append(Text(mark, style=style) + Text(text, style="" if sign else "dim"))
        if leg.p_push > 0.01:
            lines.append(Text(f" .  push chance {leg.p_push:.0%} (refunded, excluded from confidence)", style="dim"))
        for note in leg.notes:
            lines.append(Text(f" !  {note}", style="yellow"))
        lines.append(Text(f"Data reliability {leg.grade} ({leg.reliability}/100)", style="dim"))
        console.print(Panel(Group(*lines), title=f"Leg {i}: {leg.leg.describe()}  -  {leg.confidence}%",
                            title_align="left", border_style=_color(leg.confidence).replace("bold ", "")))

    rated = [l for l in legs if not l.error]
    if len(rated) >= 1:
        body = [
            Text.assemble(("Parlay hit probability: ", "bold"),
                          (f"{parlay.probability:.1%}", _color(parlay.confidence) if len(rated) == 1 else "bold"),
                          (f"   (fair odds {decimal_to_american(1 / max(parlay.probability, 1e-6)):+d})", "dim")),
            Text(f"If legs were independent: {parlay.independent_probability:.1%}   "
                 f"Average leg strength: {parlay.leg_strength:.0%}"),
        ]
        if parlay.book_odds is not None:
            edge = parlay.probability - parlay.book_probability
            body.append(Text.assemble(
                f"Book odds {parlay.book_odds:+d} imply {parlay.book_probability:.1%}; edge ",
                (f"{edge:+.1%}", "green" if edge > 0 else "red"),
                f", expected value {parlay.expected_value:+.1%} per unit staked"))
        body.append(Text(_parlay_verdict(parlay, len(rated)), style="italic"))
        for n in parlay.notes:
            body.append(Text(f" .  {n}", style="dim"))
        console.print(Panel(Group(*body), title="Overall parlay", title_align="left", border_style="cyan"))


def _why(leg: LegResult) -> str:
    """One sentence on what drives the rating: the baseline vs the line, then the biggest adjustments."""
    helps = [t.split(" (")[0] for t, s in leg.reasons(3) if s == 1]
    hurts = [t.split(" (")[0] for t, s in leg.reasons(3) if s == -1]
    high, low = leg.confidence >= 55, leg.confidence < 45
    if not (high or low):
        return "Close to a coin flip: the projection sits near the line."
    base = ""
    if leg.market == "anytime_td" and leg.baseline is not None:
        base = f"{leg.baseline:.2f} expected TDs before matchup adjustments"
    elif leg.median is not None and leg.line is not None:
        side = "above" if leg.median > leg.line else "below" if leg.median < leg.line else "right at"
        base = f"median outcome {leg.median:g} is {side} the {leg.line:g} line"
        if (leg.market in ("pass_yds", "rush_yds", "rec_yds", "rush_rec_yds") and leg.projection is not None
                and (leg.projection - leg.median) / max(leg.line, 1) > 0.1):
            base += f" (the {leg.projection:.1f} average is pulled up by big games)"
    with_, against = (helps, hurts) if high else (hurts, helps)
    parts = [base] if base else []
    if with_:
        parts.append(f"{'boosted' if high else 'dragged down'} most by {', '.join(with_)}")
    if against:
        parts.append(f"partly offset by {', '.join(against)}")
    return f"Rated {'high' if high else 'low'}: " + "; ".join(parts) + "."


def _parlay_verdict(p: ParlayResult, n: int) -> str:
    if p.expected_value is not None:
        if p.expected_value > 0.05:
            return "The model sees positive expected value at these odds."
        if p.expected_value < -0.15:
            return "The payout doesn't compensate for the risk according to the model."
        return "Roughly fair price according to the model; the book's margin likely eats any edge."
    if n == 1:
        return "Single leg."
    return (f"{n}-leg parlay: average leg strength {p.leg_strength:.0%}. Add sportsbook odds "
            f"(per leg like '-115' or --odds) to see expected value.")


def to_json(legs: list[LegResult], parlay: ParlayResult) -> str:
    def leg_dict(l: LegResult) -> dict:
        return {
            "input": getattr(l.leg, "raw", l.description), "leg": l.leg.describe() if not l.error else None,
            "matchup": l.description, "market": l.market, "side": l.side, "line": l.line,
            "projection": l.projection, "confidence": l.confidence, "probability": l.probability,
            "push_probability": l.p_push, "reliability": l.reliability, "grade": l.grade, "verdict": l.verdict,
            "summary": l.summary, "error": l.error, "notes": l.notes,
            "factors": [{"name": f.name, "effect_pct": None if f.points is not None else round(f.pct, 2),
                         "effect_points": f.points, "kind": f.applies_to, "detail": f.detail} for f in l.factors],
        }
    return json.dumps({
        "legs": [leg_dict(l) for l in legs],
        "parlay": {"probability": parlay.probability, "independent_probability": parlay.independent_probability,
                   "confidence": parlay.confidence, "leg_strength": parlay.leg_strength,
                   "book_odds": parlay.book_odds, "book_probability": parlay.book_probability,
                   "expected_value": parlay.expected_value, "notes": parlay.notes},
    }, indent=2)
