"""Keep the processed datasets current while a race weekend is in progress.

`dataset.py` builds a whole season in one go and only looks at rounds whose RACE
has finished. That is far too coarse for a live weekend: a qualifying prediction
is only worth anything BEFORE qualifying runs, so the practice data has to land
the moment the last practice session ends.

This module works one session at a time. It asks the schedule what has finished,
compares that against what is already in data/processed/, and fetches only the
gap — so a mid-weekend call costs one session load, not a season.

Which practice sessions to wait for is read from the schedule, not hard-coded:
a conventional weekend runs FP1-FP3, a sprint weekend runs FP1 only.

Run:  python refresh.py                  # bring the current season up to date
      python refresh.py --loop 300       # ...and keep doing it every 5 minutes
"""
from __future__ import annotations

import time

import pandas as pd

import config
import dataset as DS
import fastf1
import standings as ST

PRACTICE_PATH = config.PROCESSED_DIR / "practice_dataset.csv"
QUALI_PATH = config.PROCESSED_DIR / "quali_dataset.csv"

# Scheduled length of each session. Used only to decide when data should exist,
# so it errs on the generous side rather than trying to be exact.
_DURATION_H = {
    "Practice 1": 1.0, "Practice 2": 1.0, "Practice 3": 1.0,
    "Qualifying": 1.0, "Sprint Qualifying": 1.0, "Sprint Shootout": 1.0,
    "Sprint": 1.0, "Race": 2.5,
}
# Schedule name -> the session key FastF1 loads by.
_PRACTICE = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3"}


def _now() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize(None)


def current_season() -> int:
    """The season the processed data is about (latest year we hold)."""
    if QUALI_PATH.exists():
        return int(pd.read_csv(QUALI_PATH, usecols=["Year"])["Year"].max())
    return int(_now().year)


def _schedule(year: int) -> pd.DataFrame:
    return fastf1.get_event_schedule(year, include_testing=False)


def _ended(row, i: int, grace_min: int) -> bool:
    """Has session `i` of this event finished (plus a grace period for publishing)?"""
    name, dt = str(row.get(f"Session{i}") or ""), row.get(f"Session{i}DateUtc")
    if not name or pd.isna(dt):
        return False
    end = (pd.to_datetime(dt)
           + pd.Timedelta(hours=_DURATION_H.get(name, 1.0))
           + pd.Timedelta(minutes=grace_min))
    return _now() >= end


def expected_practice(row) -> list[str]:
    """Practice sessions this weekend's format actually runs, in order."""
    return [_PRACTICE[str(row.get(f"Session{i}"))] for i in range(1, 6)
            if str(row.get(f"Session{i}")) in _PRACTICE]


def practice_complete(year: int, rnd: int, grace_min: int = 5) -> bool:
    """True once every practice session of the weekend has run."""
    sched = _schedule(year)
    m = sched[sched["RoundNumber"] == rnd]
    if not len(m):
        return False
    row = m.iloc[0]
    idx = [i for i in range(1, 6) if str(row.get(f"Session{i}")) in _PRACTICE]
    return bool(idx) and all(_ended(row, i, grace_min) for i in idx)


def _append(path, existing: pd.DataFrame, frames: list[pd.DataFrame],
            keys: list[str]) -> None:
    data = pd.concat(([existing] if len(existing) else []) + frames,
                     ignore_index=True)
    data = data.drop_duplicates(keys, keep="last")
    data = data.sort_values([k for k in ("Year", "Round", "Session") if k in data])
    data.to_csv(path, index=False)


def refresh_practice(year: int, grace_min: int = 5, verbose: bool = True) -> list[str]:
    """Fetch any finished practice session missing from the practice dataset."""
    have = pd.read_csv(PRACTICE_PATH) if PRACTICE_PATH.exists() else pd.DataFrame()
    seen = (set(zip(have["Year"], have["Round"], have["Session"]))
            if len(have) else set())

    frames, added = [], []
    for _, row in _schedule(year).iterrows():
        rnd = int(row["RoundNumber"])
        for i in range(1, 6):
            abbr = _PRACTICE.get(str(row.get(f"Session{i}")))
            if not abbr or (year, rnd, abbr) in seen:
                continue
            if not _ended(row, i, grace_min):
                continue
            df = DS._practice_rows_for_session(year, rnd, abbr)
            if df is None or df.empty:
                continue
            frames.append(df)
            added.append(f"R{rnd} {abbr}")
            if verbose:
                print(f"  + practice {year} R{rnd} {abbr}: {len(df)} drivers")

    if frames:
        _append(PRACTICE_PATH, have, frames, ["Year", "Round", "Session", "Driver"])
    return added


def refresh_quali(year: int, grace_min: int = 5, verbose: bool = True) -> list[str]:
    """Fetch any finished qualifying session missing from the quali dataset."""
    have = pd.read_csv(QUALI_PATH) if QUALI_PATH.exists() else pd.DataFrame()
    seen = set(zip(have["Year"], have["Round"])) if len(have) else set()

    frames, added = [], []
    for _, row in _schedule(year).iterrows():
        rnd = int(row["RoundNumber"])
        if (year, rnd) in seen:
            continue
        idx = next((i for i in range(1, 6)
                    if str(row.get(f"Session{i}")) == "Qualifying"), None)
        if idx is None or not _ended(row, idx, grace_min):
            continue
        df = DS._quali_rows_for_event(year, rnd)
        if df is None or df.empty:
            continue
        frames.append(df)
        added.append(f"R{rnd}")
        if verbose:
            print(f"  + quali {year} R{rnd}: {len(df)} drivers")

    if frames:
        _append(QUALI_PATH, have, frames, ["Year", "Round", "Driver"])
    return added


def auto_refresh(year: int = None, grace_min: int = 5,
                 verbose: bool = True) -> dict:
    """Bring practice, qualifying and championship results up to date.

    Ordering matters: practice first, so that a prediction becomes available at
    the earliest possible moment rather than waiting on the rest.
    """
    year = year or current_season()
    before = len(ST.load_season_results(year))

    practice = refresh_practice(year, grace_min, verbose)
    quali = refresh_quali(year, grace_min, verbose)
    try:
        results = len(ST.fetch_season_results(year)) - before
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"  results refresh failed: {type(e).__name__}: {e}")
        results = 0

    parts = []
    if practice:
        parts.append("practice " + ", ".join(practice))
    if quali:
        parts.append("quali " + ", ".join(quali))
    if results > 0:
        parts.append(f"{results} result rows")
    return {"year": year, "practice": practice, "quali": quali,
            "results": max(0, results), "changed": bool(parts),
            "summary": "; ".join(parts) or "already up to date"}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--grace", type=int, default=5,
                   help="minutes to wait after a session ends before fetching")
    p.add_argument("--loop", type=int, default=0,
                   help="seconds between checks; 0 = run once and exit")
    a = p.parse_args()

    while True:
        r = auto_refresh(a.year, a.grace)
        print(f"{time.strftime('%H:%M:%S')}  {r['year']}: {r['summary']}")
        if not a.loop:
            break
        time.sleep(a.loop)
