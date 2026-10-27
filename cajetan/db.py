"""DuckDB connection helpers and the as-of filter shared by every query."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from cajetan import config


class DataMissingError(RuntimeError):
    """Raised when the local database has not been populated yet."""


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if read_only and not config.DB_PATH.exists():
        raise DataMissingError("No local data found. Run `cajetan update` first.")
    return duckdb.connect(str(config.DB_PATH), read_only=read_only)


def table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return bool(row and row[0])


@dataclass(frozen=True)
class AsOf:
    """Point in time a rating is made from: only games before `week` of `season` are used."""

    season: int
    week: int

    def clause(self, alias: str = "", min_season: int | None = None) -> str:
        a = f"{alias}." if alias else ""
        lo = self.season - 1 if min_season is None else min_season
        return (
            f"({a}season >= {lo} AND ({a}season < {self.season} "
            f"OR ({a}season = {self.season} AND {a}week < {self.week})))"
        )

    def weight(self, alias: str = "") -> str:
        """SQL expression weighting current-season rows 1.0 and prior-season rows lower."""
        a = f"{alias}." if alias else ""
        return f"(CASE WHEN {a}season = {self.season} THEN 1.0 ELSE {config.PRIOR_SEASON_WEIGHT} END)"
