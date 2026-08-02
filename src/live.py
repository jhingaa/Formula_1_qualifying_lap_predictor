"""Live race-weekend tracking + live lap-time predictions.

FastF1's live timing is a two-process design:
  1. A RECORDER captures the live SignalR stream to a file while a session runs.
  2. A READER loads that (growing) file and works with the laps so far.

Workflow during a live session (e.g. Dutch GP, 2026-08-23):

    # Terminal 1 — start recording when the session goes live:
    python live.py record --out ../data/live/dutch_race.txt

    # Terminal 2 — watch predictions update as laps come in:
    python live.py monitor --file ../data/live/dutch_race.txt \
        --year 2026 --round 12 --session R

No session live right now? Test the exact same logic against a finished race:

    python live.py replay --year 2024 --round 1 --session R
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

# Windows consoles default to cp1252 and choke on symbols like Δ; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import config
import fastf1
from data_loader import load_session_laps
from predict import load_model, predict_laps, predict_next_lap

LIVE_DIR = config.DATA_DIR / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Recording (run while a session is live)
# ---------------------------------------------------------------------------
def record(out_file: str, timeout: int = 0) -> None:
    from fastf1.livetiming.client import SignalRClient
    print(f"Recording live timing -> {out_file}\nPress Ctrl+C to stop.")
    kwargs = {"filename": out_file}
    if timeout:
        kwargs["timeout"] = timeout
    client = SignalRClient(**kwargs)
    client.start()  # blocks until the session ends / Ctrl+C


# ---------------------------------------------------------------------------
# Loading laps — from a live recording OR (for replay/testing) a finished race
# ---------------------------------------------------------------------------
def _laps_from_recording(out_file: str, year: int, rnd: int, session_type: str) -> pd.DataFrame:
    from fastf1.livetiming.data import LiveTimingData
    livedata = LiveTimingData(out_file)
    session = fastf1.get_session(year, rnd, session_type)
    session.load(livedata=livedata, telemetry=False, weather=True, messages=False)
    from data_loader import load_session_laps as _  # noqa
    # Reuse the same flattening logic by re-implementing the minimal path:
    return _flatten(session, year, rnd, session_type)


def _flatten(session, year, rnd, session_type) -> pd.DataFrame:
    """Flatten a loaded Session the same way data_loader does (for live data)."""
    import data_loader as dl
    laps = session.laps
    cols = [c for c in dl._LAP_COLS if c in laps.columns]
    df = laps[cols].copy().reset_index(drop=True)
    weather = laps.get_weather_data()
    weather = weather[[c for c in dl._WEATHER_COLS if c in weather.columns]].reset_index(drop=True)
    df = pd.concat([df, weather], axis=1)
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in df.columns:
            df[col] = df[col].dt.total_seconds()
    df["IsPitLap"] = False
    for col in ("PitInTime", "PitOutTime"):
        if col in df.columns:
            df["IsPitLap"] = df["IsPitLap"] | df[col].notna()
            df = df.drop(columns=col)
    ev = session.event
    df.insert(0, "Circuit", ev["Location"])
    return df


# ---------------------------------------------------------------------------
# The monitor: what you actually watch during the weekend
# ---------------------------------------------------------------------------
def snapshot(laps: pd.DataFrame, bundle: dict) -> None:
    """Print current pace vs model expectation, and predicted next laps."""
    racing = laps[laps["LapTime"].notna() & laps.get("IsAccurate", True)]
    if "IsPitLap" in racing:
        racing = racing[~racing["IsPitLap"]]
    if racing.empty:
        print("  (no completed racing laps yet)")
        return

    # Predicted next lap per driver (fastest expected first).
    nxt = predict_next_lap(bundle, racing)
    print("\n  PREDICTED NEXT LAP (fastest expected first):")
    print("  " + "-" * 60)
    for _, r in nxt.head(10).iterrows():
        comp = str(r.get("Compound", "?"))[:4]
        print(f"   {r['Driver']:<4} {r['PredNextLap']:7.3f}s  "
              f"{comp:<5} tyre {int(r.get('TyreLife', 0)):>2}  "
              f"lap {int(r.get('LastLap', 0)):>2}")

    # Latest completed lap: predicted vs actual (who's over/under-performing).
    latest = racing.sort_values("LapNumber").groupby("Driver").tail(1).copy()
    latest["Pred"] = predict_laps(bundle, latest).values
    latest["Delta"] = latest["LapTime"] - latest["Pred"]
    latest = latest.sort_values("Delta")  # most under-model (overperforming) first
    print("\n  LATEST LAP  actual vs model (Δ<0 = faster than expected):")
    print("  " + "-" * 60)
    for _, r in latest.head(10).iterrows():
        print(f"   {r['Driver']:<4} actual {r['LapTime']:7.3f}  "
              f"model {r['Pred']:7.3f}  Δ {r['Delta']:+.3f}s")


def monitor(file: str, year: int, rnd: int, session_type: str,
            interval: int = 30, once: bool = False) -> None:
    bundle = load_model("race" if session_type in ("R", "S") else "quali")
    while True:
        try:
            laps = _laps_from_recording(file, year, rnd, session_type)
            print(f"\n{'='*64}\n Live snapshot @ {time.strftime('%H:%M:%S')}  "
                  f"({len(laps)} laps recorded)\n{'='*64}")
            snapshot(laps, bundle)
        except Exception as e:  # noqa: BLE001
            print(f"  waiting for data... ({type(e).__name__}: {e})")
        if once:
            break
        time.sleep(interval)


def replay(year: int, rnd: int, session_type: str) -> None:
    """Test the monitor against a finished race (loaded via normal FastF1)."""
    bundle = load_model("race" if session_type in ("R", "S") else "quali")
    laps = load_session_laps(year, rnd, session_type)
    print(f"\n{'='*64}\n REPLAY {year} R{rnd} {session_type} "
          f"({len(laps)} laps) - simulating a live snapshot\n{'='*64}")
    snapshot(laps, bundle)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("record"); pr.add_argument("--out", required=True)
    pr.add_argument("--timeout", type=int, default=0)

    pm = sub.add_parser("monitor")
    pm.add_argument("--file", required=True)
    pm.add_argument("--year", type=int, required=True)
    pm.add_argument("--round", type=int, required=True)
    pm.add_argument("--session", default="R")
    pm.add_argument("--interval", type=int, default=30)
    pm.add_argument("--once", action="store_true")

    pp = sub.add_parser("replay")
    pp.add_argument("--year", type=int, required=True)
    pp.add_argument("--round", type=int, required=True)
    pp.add_argument("--session", default="R")

    a = p.parse_args()
    if a.cmd == "record":
        record(a.out, a.timeout)
    elif a.cmd == "monitor":
        monitor(a.file, a.year, a.round, a.session, a.interval, a.once)
    elif a.cmd == "replay":
        replay(a.year, a.round, a.session)
