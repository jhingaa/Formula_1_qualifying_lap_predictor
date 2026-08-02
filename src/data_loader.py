"""Load F1 sessions via FastF1 and flatten them into clean, model-ready DataFrames.

A "clean" lap row combines three things FastF1 keeps separate:
  1. Lap timing + tyre data (compound, tyre age, stint, sectors)
  2. Weather measured closest to that lap (air/track temp, wind, rain)
  3. Session metadata (year, round, circuit, session type)

Usage
-----
    from data_loader import load_session_laps
    df = load_session_laps(2023, 1, "R")   # 2023 Bahrain race
    print(df.shape)
"""
from __future__ import annotations

import pandas as pd

import config  # noqa: F401  (importing enables the FastF1 cache)
import fastf1

# Weather columns we keep and align to each lap.
_WEATHER_COLS = [
    "AirTemp", "TrackTemp", "Humidity", "Pressure",
    "WindSpeed", "WindDirection", "Rainfall",
]

# Lap columns worth keeping (others are dropped to keep the frame lean).
_LAP_COLS = [
    "Driver", "Team", "LapNumber", "LapTime", "Stint",
    "Compound", "TyreLife", "FreshTyre",
    "Sector1Time", "Sector2Time", "Sector3Time",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "IsAccurate", "TrackStatus", "Position",
    "PitInTime", "PitOutTime",
]


def _td_to_seconds(series: pd.Series) -> pd.Series:
    """Convert a Timedelta column to float seconds (NaT -> NaN)."""
    return series.dt.total_seconds()


def load_session(year: int, rnd: int, session_type: str) -> fastf1.core.Session:
    """Load a single FastF1 session with laps + weather.

    session_type: "Q" (qualifying), "R" (race), "FP1"/"FP2"/"FP3", "S" (sprint).
    """
    session = fastf1.get_session(year, rnd, session_type)
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    return session


def load_session_laps(year: int, rnd: int, session_type: str) -> pd.DataFrame:
    """Return a flat DataFrame: one row per lap, with tyres, weather, metadata."""
    session = load_session(year, rnd, session_type)
    laps = session.laps

    if laps is None or len(laps) == 0:
        raise ValueError(f"No lap data for {year} R{rnd} {session_type}")

    # Keep only columns that actually exist (FastF1 schema varies by season).
    cols = [c for c in _LAP_COLS if c in laps.columns]
    df = laps[cols].copy()

    # Align weather to each lap (measured closest to the lap's start time).
    weather = laps.get_weather_data()
    weather = weather[[c for c in _WEATHER_COLS if c in weather.columns]]
    df = df.reset_index(drop=True)
    weather = weather.reset_index(drop=True)
    df = pd.concat([df, weather], axis=1)

    # Convert all Timedelta columns to float seconds for modelling.
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in df.columns:
            df[col] = _td_to_seconds(df[col])
    # Pit times: keep only a boolean "did this lap involve a pit stop".
    df["IsPitLap"] = False
    for col in ("PitInTime", "PitOutTime"):
        if col in df.columns:
            df["IsPitLap"] = df["IsPitLap"] | df[col].notna()
            df = df.drop(columns=col)

    # Session metadata.
    ev = session.event
    df.insert(0, "Year", year)
    df.insert(1, "Round", rnd)
    df.insert(2, "Session", session_type)
    df.insert(3, "EventName", ev["EventName"])
    df.insert(4, "Circuit", ev["Location"])

    return df


if __name__ == "__main__":
    # Quick sanity check: load one race and eyeball it.
    demo = load_session_laps(2023, 1, "R")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(f"\nLoaded {len(demo)} laps, {demo.shape[1]} columns")
    print("\nColumns:", list(demo.columns))
    print("\nSample (fastest 5 accurate laps):")
    sample = demo[demo["IsAccurate"]].nsmallest(5, "LapTime")
    print(sample[["Driver", "LapNumber", "LapTime", "Compound", "TyreLife",
                  "TrackTemp", "AirTemp", "Rainfall"]].to_string(index=False))
