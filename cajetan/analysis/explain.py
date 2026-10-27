"""Result containers shared by the evaluators, the parlay combiner and the report."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Factor:
    """One adjustment to a projection, with the evidence behind it."""

    name: str
    multiplier: float  # applied to the projection (1.0 = no effect)
    detail: str
    applies_to: str = "efficiency"  # volume / efficiency / projection / info
    points: float | None = None  # team markets: shift in points instead of a multiplier

    @property
    def pct(self) -> float:
        return (self.multiplier - 1.0) * 100

    @property
    def magnitude(self) -> float:
        """Comparable size of the effect (1 point of margin ~ 3% of a player projection)."""
        return abs(self.points) * 3 if self.points is not None else abs(self.pct)

    def label(self) -> str:
        effect = f"{self.points:+.1f} pts" if self.points is not None else f"{self.pct:+.1f}%"
        return f"{self.name} ({effect}): {self.detail}"


@dataclass
class LegResult:
    leg: object
    description: str
    market_label: str
    side: str
    line: float | None
    projection: float | None
    probability: float  # chance the leg wins (pushes removed)
    baseline: float | None = None  # projection before matchup adjustments
    median: float | None = None  # median outcome (lines are usually set near it)
    p_push: float = 0.0
    reliability: int = 50  # 0-100: how much data the rating stands on
    factors: list[Factor] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    summary: str = ""
    # Used by the parlay correlation model.
    market: str | None = None
    game_id: str | None = None
    team: str | None = None
    opponent: str | None = None
    position: str | None = None
    player_id: str | None = None
    error: str | None = None
    season: int | None = None
    week: int | None = None
    # Pieces of the projection, kept so `cajetan fit` can re-weight them without re-rating.
    components: dict = field(default_factory=dict)
    # Probability before the final calibration step (equal to `probability` when none applies).
    raw_probability: float | None = None

    @property
    def confidence(self) -> int:
        return round(self.probability * 100)

    @property
    def grade(self) -> str:
        r = self.reliability
        return "A" if r >= 80 else "B" if r >= 60 else "C" if r >= 40 else "D"

    @property
    def verdict(self) -> str:
        if self.error:
            return "error"
        c = self.confidence
        return ("strong" if c >= 65 else "lean" if c >= 55 else "coin flip" if c >= 45
                else "fade" if c >= 35 else "avoid")

    def reasons(self, n: int = 5) -> list[tuple[str, int]]:
        """Most important factors as (text, sign): +1 helps this side of the bet, -1 hurts it, 0 is context."""
        helps_up = self.side in ("over", "yes", "win", "cover")
        ranked = sorted(
            (f for f in self.factors if f.applies_to != "info" and f.magnitude >= 0.5),
            key=lambda f: -f.magnitude,
        )
        out = []
        for f in ranked[:n]:
            up = f.points > 0 if f.points is not None else f.multiplier > 1
            out.append((f.label(), 1 if up == helps_up else -1))
        for f in self.factors:
            if f.applies_to == "info":
                out.append((f"{f.name}: {f.detail}", 0))
        return out


@dataclass
class ParlayResult:
    legs: list[LegResult]
    probability: float  # correlated joint probability
    independent_probability: float
    leg_strength: float  # geometric mean of leg probabilities
    book_odds: int | None = None
    book_probability: float | None = None
    expected_value: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> int:
        return round(self.probability * 100)
