"""Game-day conditions: recorded weather for played games, an Open-Meteo forecast for upcoming ones.

nflverse records kickoff temperature and wind for outdoor games once they are played.
For upcoming outdoor games (and retractable roofs listed as open) the forecast comes
from Open-Meteo, which is free and needs no key. Domes and closed roofs have no
weather effect. Forecasts are cached in data/weather_cache.json for a few hours.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime

import duckdb

from cajetan import config

# Approximate stadium coordinates by nflverse stadium id (outdoor, retractable and international venues).
STADIUMS: dict[str, tuple[float, float]] = {
    "BAL00": (39.278, -76.623), "BUF00": (42.774, -78.787), "CAR00": (35.226, -80.853),
    "CHI98": (41.862, -87.617), "CIN00": (39.095, -84.516), "CLE00": (41.506, -81.700),
    "DEN00": (39.744, -105.020), "GNB00": (44.501, -88.062), "JAX00": (30.324, -81.637),
    "KAN00": (39.049, -94.484), "BOS00": (42.091, -71.264), "NYC01": (40.814, -74.074),
    "PHI00": (39.901, -75.168), "PIT00": (40.447, -80.016), "SEA00": (47.595, -122.332),
    "SFO01": (37.403, -121.970), "TAM00": (27.976, -82.503), "NAS00": (36.166, -86.771),
    "WAS00": (38.908, -76.864), "MIA00": (25.958, -80.239), "HOU00": (29.685, -95.411),
    "IND00": (39.760, -86.164), "PHO00": (33.528, -112.263), "ATL97": (33.755, -84.401),
    "DAL00": (32.748, -97.093), "LON00": (51.556, -0.280), "LON01": (51.456, -0.342),
    "LON02": (51.604, -0.066), "MUN01": (48.219, 11.625), "MAD01": (40.453, -3.688),
    "PAR00": (48.924, 2.360), "MEX00": (19.303, -99.150), "RIO00": (-22.912, -43.230),
    "MEL00": (-37.820, 144.983), "FRA00": (50.069, 8.646), "SAO00": (-23.545, -46.474),
}
OPEN_ROOFS = ("outdoors", "open")
FORECAST_DAYS = 15
CACHE_SECONDS = 3 * 3600


@dataclass
class Conditions:
    temp: float | None  # Fahrenheit
    wind: float | None  # mph
    indoors: bool
    source: str  # recorded / forecast / dome

    def describe(self) -> str:
        if self.indoors:
            return "indoors"
        parts = []
        if self.temp is not None:
            parts.append(f"{self.temp:.0f}°F")
        if self.wind is not None:
            parts.append(f"wind {self.wind:.0f} mph")
        return ", ".join(parts) + f" ({self.source})"


def wind_bucket(wind: float | None) -> str | None:
    if wind is None:
        return None
    return "calm" if wind < 10 else "breezy" if wind < 15 else "windy" if wind < 20 else "very_windy"


def temp_bucket(temp: float | None) -> str | None:
    if temp is None:
        return None
    return "frigid" if temp <= 25 else "cold" if temp <= 40 else "mild"


WIND_LABELS = {"calm": "under 10 mph", "breezy": "10-14 mph", "windy": "15-19 mph", "very_windy": "20+ mph"}
TEMP_LABELS = {"frigid": "25°F or colder", "cold": "26-40°F", "mild": "over 40°F"}


def game_conditions(con: duckdb.DuckDBPyConnection, game_id: str, allow_forecast: bool = True) -> Conditions | None:
    row = con.execute(
        "SELECT roof, temp, wind, stadium_id, gameday, gametime, home_score FROM schedules WHERE game_id = ?",
        [game_id],
    ).fetchone()
    if not row:
        return None
    roof, temp, wind, stadium, gameday, gametime, score = row
    if roof in ("dome", "closed"):
        return Conditions(None, None, True, "dome")
    if temp is not None or wind is not None:
        return Conditions(_num(temp), _num(wind), False, "recorded")
    if score is not None or roof not in OPEN_ROOFS or not allow_forecast:
        return None
    return _forecast(game_id, stadium, gameday, gametime)


def _num(x) -> float | None:
    return None if x is None or x != x else float(x)


def _forecast(game_id: str, stadium: str | None, gameday: str | None, gametime: str | None) -> Conditions | None:
    if not stadium or stadium not in STADIUMS or not gameday:
        return None
    try:
        day = date.fromisoformat(gameday)
    except ValueError:
        return None
    if not 0 <= (day - date.today()).days <= FORECAST_DAYS:
        return None
    cache = _read_cache()
    hit = cache.get(game_id)
    if hit and time.time() - hit["at"] < CACHE_SECONDS:
        return Conditions(hit["temp"], hit["wind"], False, "forecast")
    lat, lon = STADIUMS[stadium]
    url = ("https://api.open-meteo.com/v1/forecast?"
           f"latitude={lat}&longitude={lon}&hourly=temperature_2m,wind_speed_10m"
           "&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=America%2FNew_York"
           f"&start_date={day.isoformat()}&end_date={day.isoformat()}")
    try:
        with urllib.request.urlopen(url, timeout=4) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None  # no network or service down: rate without weather
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    try:
        kickoff = int((gametime or "13:00").split(":")[0])  # nflverse kickoff times are US Eastern
    except ValueError:
        kickoff = 13
    idx = [i for i, t in enumerate(times) if kickoff <= datetime.fromisoformat(t).hour <= kickoff + 3]
    if not idx:
        return None
    temp = sum(hourly["temperature_2m"][i] for i in idx) / len(idx)
    wind = sum(hourly["wind_speed_10m"][i] for i in idx) / len(idx)
    cache[game_id] = {"at": time.time(), "temp": round(temp, 1), "wind": round(wind, 1)}
    _write_cache(cache)
    return Conditions(round(temp, 1), round(wind, 1), False, "forecast")


def _read_cache() -> dict:
    try:
        return json.loads(config.WEATHER_CACHE.read_text())
    except (OSError, ValueError):
        return {}


def _write_cache(cache: dict) -> None:
    try:
        config.WEATHER_CACHE.write_text(json.dumps(cache))
    except OSError:
        pass
