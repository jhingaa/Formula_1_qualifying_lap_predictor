"""Build model-ready datasets by aggregating many FastF1 sessions.

Two builders:
  build_quali_dataset(years)  -> one row per driver per event: their best clean lap
  build_race_dataset(years)   -> one row per green-flag racing lap

Both are resilient: a session that fails to load (cancelled round, missing
timing data, sprint-only weekend) is skipped with a warning, not fatal.
Results are cached to data/processed/ as CSV so we build once and reuse.
"""
from __future__ import annotations

import warnings
import logging

import numpy as np
import pandas as pd

import config
import fastf1
from data_loader import load_session_laps

fastf1.set_log_level("ERROR")
warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.ERROR)

# Weather features carried into both datasets.
WEATHER_FEATURES = ["AirTemp", "TrackTemp", "Humidity", "WindSpeed", "Rainfall"]


def _completed_rounds(year: int) -> list[int]:
    """Round numbers whose race has already happened (or all, for past seasons)."""
    from datetime import datetime
    sched = fastf1.get_event_schedule(year, include_testing=False)
    now = datetime.now()
    done = sched[sched["Session5DateUtc"] < now]
    return [int(r) for r in done["RoundNumber"].tolist() if r > 0]


# ---------------------------------------------------------------------------
# Qualifying dataset
# ---------------------------------------------------------------------------
def _quali_rows_for_event(year: int, rnd: int) -> pd.DataFrame | None:
    try:
        laps = load_session_laps(year, rnd, "Q")
    except Exception as e:  # noqa: BLE001
        print(f"  skip {year} R{rnd} Q: {type(e).__name__}: {e}")
        return None

    clean = laps[(laps["IsAccurate"]) & (laps["LapTime"].notna())]
    if clean.empty:
        return None

    # Best clean lap per driver, and the conditions during that lap.
    idx = clean.groupby("Driver")["LapTime"].idxmin()
    best = clean.loc[idx].copy()

    keep = (["Year", "Round", "EventName", "Circuit", "Driver", "Team",
             "Compound", "LapTime"] + WEATHER_FEATURES)
    keep = [c for c in keep if c in best.columns]
    out = best[keep].rename(columns={"LapTime": "BestLapTime"})
    return out


def build_quali_dataset(years: list[int], save: bool = True) -> pd.DataFrame:
    frames = []
    for year in years:
        for rnd in _completed_rounds(year):
            df = _quali_rows_for_event(year, rnd)
            if df is not None:
                frames.append(df)
                print(f"  quali {year} R{rnd}: {len(df)} drivers")
    data = pd.concat(frames, ignore_index=True)
    if save:
        path = config.PROCESSED_DIR / "quali_dataset.csv"
        data.to_csv(path, index=False)
        print(f"Saved {len(data)} quali rows -> {path}")
    return data


# ---------------------------------------------------------------------------
# Practice dataset
# ---------------------------------------------------------------------------
# Sprint weekends only run FP1, so a missing FP2/FP3 is normal, not an error.
PRACTICE_SESSIONS = ["FP1", "FP2", "FP3"]


def _practice_rows_for_session(year: int, rnd: int, sess: str) -> pd.DataFrame | None:
    """One row per driver: their best clean lap in this practice session.

    Practice laps are a mix of low-fuel qualifying simulations and heavy-fuel
    race runs. The best clean lap is the closest thing to a low-fuel effort, so
    that's what we keep — plus the compound it was set on, because a best lap on
    HARD says something very different about pace than the same time on SOFT.
    """
    try:
        laps = load_session_laps(year, rnd, sess)
    except Exception as e:  # noqa: BLE001
        print(f"  skip {year} R{rnd} {sess}: {type(e).__name__}: {e}")
        return None

    clean = laps[(laps["IsAccurate"]) & (laps["LapTime"].notna()) & (~laps["IsPitLap"])]
    if clean.empty:
        return None

    idx = clean.groupby("Driver")["LapTime"].idxmin()
    best = clean.loc[idx].copy()
    # How much running each driver actually did — a driver with 3 clean laps
    # tells us far less than one with 20, and confidence should know that.
    best["CleanLaps"] = best["Driver"].map(clean.groupby("Driver").size())

    keep = (["Year", "Round", "EventName", "Circuit", "Session", "Driver", "Team",
             "Compound", "TyreLife", "CleanLaps", "LapTime"] + WEATHER_FEATURES)
    keep = [c for c in keep if c in best.columns]
    return best[keep].rename(columns={"LapTime": "BestLapTime"})


def build_practice_dataset(years: list[int], save: bool = True) -> pd.DataFrame:
    frames = []
    for year in years:
        for rnd in _completed_rounds(year):
            for sess in PRACTICE_SESSIONS:
                df = _practice_rows_for_session(year, rnd, sess)
                if df is not None:
                    frames.append(df)
                    print(f"  practice {year} R{rnd} {sess}: {len(df)} drivers")
    if not frames:
        raise SystemExit("No practice sessions could be loaded.")
    data = pd.concat(frames, ignore_index=True)
    if save:
        path = config.PROCESSED_DIR / "practice_dataset.csv"
        data.to_csv(path, index=False)
        print(f"Saved {len(data)} practice rows -> {path}")
    return data


# ---------------------------------------------------------------------------
# Race stint dataset
# ---------------------------------------------------------------------------
def _race_rows_for_event(year: int, rnd: int) -> pd.DataFrame | None:
    try:
        laps = load_session_laps(year, rnd, "R")
    except Exception as e:  # noqa: BLE001
        print(f"  skip {year} R{rnd} R: {type(e).__name__}: {e}")
        return None

    df = laps.copy()
    # Keep only true green-flag racing laps:
    #   - accurate timing, real lap time
    #   - not an in/out (pit) lap
    #   - green track status ('1'); anything else = SC/VSC/yellow/red
    mask = (
        df["IsAccurate"]
        & df["LapTime"].notna()
        & (~df["IsPitLap"])
        & (df["TrackStatus"].astype(str) == "1")
    )
    df = df[mask].copy()
    if df.empty:
        return None

    # LapNumber is our fuel-load proxy (fuel burns ~linearly through the race).
    keep = (["Year", "Round", "EventName", "Circuit", "Driver", "Team",
             "LapNumber", "Stint", "Compound", "TyreLife", "Position",
             "LapTime"] + WEATHER_FEATURES)
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def build_race_dataset(years: list[int], save: bool = True) -> pd.DataFrame:
    frames = []
    for year in years:
        for rnd in _completed_rounds(year):
            df = _race_rows_for_event(year, rnd)
            if df is not None:
                frames.append(df)
                print(f"  race {year} R{rnd}: {len(df)} laps")
    data = pd.concat(frames, ignore_index=True)
    if save:
        path = config.PROCESSED_DIR / "race_dataset.csv"
        data.to_csv(path, index=False)
        print(f"Saved {len(data)} race laps -> {path}")
    return data


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["quali", "race", "practice", "both", "all"],
                   default="both")
    p.add_argument("--years", type=int, nargs="+", default=[2023, 2024])
    args = p.parse_args()

    if args.kind in ("quali", "both", "all"):
        print(f"Building QUALI dataset for {args.years} ...")
        build_quali_dataset(args.years)
    if args.kind in ("practice", "all"):
        print(f"Building PRACTICE dataset for {args.years} ...")
        build_practice_dataset(args.years)
    if args.kind in ("race", "both", "all"):
        print(f"Building RACE dataset for {args.years} ...")
        build_race_dataset(args.years)
