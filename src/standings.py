"""Drivers' championship standings, as they stood after any given round.

The dashboard shows the table "as of" the race week you are looking at, not the
end-of-season table, so every round's results are stored individually and summed
up to the round you ask for.

Points come from FastF1's session results (which already apply the current
scoring system, including sprint points) — we never hard-code a points table.

Run:  python standings.py            # refresh the cache, print the table
"""
from __future__ import annotations

import pandas as pd

import config
import fastf1

RESULTS_COLS = ["Year", "Round", "EventName", "Session", "Driver", "Team",
                "Position", "Points", "Status"]

# Sessions that award championship points, in the order they are run.
_POINT_SESSIONS = {"S": "Sprint", "R": "Race"}


def _cache_path(year: int):
    return config.PROCESSED_DIR / f"results_{year}.csv"


def _session_results(year: int, rnd: int, kind: str) -> pd.DataFrame | None:
    """Results of one points-scoring session, or None if it hasn't run."""
    try:
        s = fastf1.get_session(year, rnd, kind)
        s.load(laps=False, telemetry=False, weather=False, messages=False)
        res = s.results
    except Exception:  # noqa: BLE001  (not run yet / no data / no sprint)
        return None
    if res is None or len(res) == 0 or res["Points"].isna().all():
        return None
    out = pd.DataFrame({
        "Year": year,
        "Round": rnd,
        "EventName": s.event["EventName"],
        "Session": kind,
        "Driver": res["Abbreviation"].astype(str),
        "Team": res["TeamName"].astype(str),
        "Position": pd.to_numeric(res["Position"], errors="coerce"),
        "Points": pd.to_numeric(res["Points"], errors="coerce").fillna(0.0),
        "Status": res["Status"].astype(str) if "Status" in res else "",
    })
    return out[RESULTS_COLS].reset_index(drop=True)


def fetch_season_results(year: int, force: bool = False) -> pd.DataFrame:
    """Every points-scoring session of a season, cached on disk.

    Only rounds that are missing from the cache are fetched, so calling this
    after each race weekend costs one or two session loads rather than a full
    season re-download.
    """
    path = _cache_path(year)
    have = pd.read_csv(path) if (path.exists() and not force) else \
        pd.DataFrame(columns=RESULTS_COLS)

    sched = fastf1.get_event_schedule(year, include_testing=False)
    now = pd.Timestamp.utcnow().tz_localize(None)
    done = {(int(r), str(s)) for r, s in zip(have["Round"], have["Session"])} \
        if len(have) else set()

    frames = [have] if len(have) else []
    for _, ev in sched.iterrows():
        rnd = int(ev["RoundNumber"])
        # Skip weekends that are still ahead. EventDate is race day, so allow a
        # day of slack: mid-weekend, Saturday's sprint has already scored.
        if pd.notna(ev["EventDate"]) and \
                pd.to_datetime(ev["EventDate"]) > now + pd.Timedelta(days=1):
            continue
        for kind in _POINT_SESSIONS:
            if kind == "S" and "sprint" not in str(ev["EventFormat"]).lower():
                continue
            if (rnd, kind) in done:
                continue
            got = _session_results(year, rnd, kind)
            if got is not None:
                frames.append(got)

    df = pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=RESULTS_COLS)
    if len(df):
        df = df.drop_duplicates(["Round", "Session", "Driver"], keep="last")
        df = df.sort_values(["Round", "Session", "Position"]).reset_index(drop=True)
        df.to_csv(path, index=False)
    return df


def load_season_results(year: int) -> pd.DataFrame:
    """Cached results only — never hits the network (used by the web request path)."""
    path = _cache_path(year)
    if not path.exists():
        return pd.DataFrame(columns=RESULTS_COLS)
    return pd.read_csv(path)


def _countback(sub: pd.DataFrame, places: int = 12) -> tuple:
    """FIA tie-break: most wins, then most 2nds, then 3rds, and so on.

    Returned as a sort key (negated counts, so ascending sort = better).
    """
    finished = sub.loc[sub["Session"] == "R", "Position"].dropna()
    return tuple(-int((finished == p).sum()) for p in range(1, places + 1))


def standings(year: int, upto_round: int | None = None,
              results: pd.DataFrame | None = None) -> list[dict]:
    """Drivers' championship as it stood after `upto_round`.

    Rounds after `upto_round` are ignored, so selecting an earlier race week in
    the dashboard shows the table as it looked at the time — not today's.
    """
    df = results if results is not None else load_season_results(year)
    if df is None or not len(df):
        return []
    df = df[df["Year"] == year]
    if upto_round is not None:
        df = df[df["Round"] <= int(upto_round)]
    if not len(df):
        return []

    rows = []
    for code, sub in df.groupby("Driver"):
        races = sub[sub["Session"] == "R"]
        pos = races["Position"].dropna()
        # The team a driver is scoring for is whichever they last raced for.
        last = sub.sort_values(["Round", "Session"]).iloc[-1]
        rows.append({
            "code": str(code),
            "team": str(last["Team"]),
            "points": float(sub["Points"].sum()),
            "wins": int((pos == 1).sum()),
            "podiums": int((pos <= 3).sum()),
            "best": int(pos.min()) if len(pos) else None,
            "starts": int(len(races)),
            "_key": _countback(sub),
        })

    rows.sort(key=lambda r: (-r["points"], r["_key"]))
    for i, r in enumerate(rows):
        r.pop("_key")
        r["pos"] = i + 1
    return rows


def rounds_scored(year: int, upto_round: int | None = None,
                  results: pd.DataFrame | None = None) -> list[int]:
    """Rounds up to `upto_round` that have actually awarded points, in order."""
    df = results if results is not None else load_season_results(year)
    if df is None or not len(df):
        return []
    r = df.loc[df["Year"] == year, "Round"].astype(int).unique()
    if upto_round is not None:
        r = [x for x in r if x <= int(upto_round)]
    return sorted(int(x) for x in r)


def movement(year: int, upto_round: int | None = None,
             results: pd.DataFrame | None = None) -> dict[str, int]:
    """Places gained (+) or lost (-) across the most recent counted round.

    Compares against the previous round that scored, not `upto_round - 1`, so a
    cancelled or not-yet-run round in between doesn't blank out the column.
    """
    df = results if results is not None else load_season_results(year)
    done = rounds_scored(year, upto_round, df)
    if len(done) < 2:
        return {}
    prev = {r["code"]: r["pos"] for r in standings(year, done[-2], df)}
    now = {r["code"]: r["pos"] for r in standings(year, done[-1], df)}
    return {c: prev[c] - p for c, p in now.items() if c in prev}


if __name__ == "__main__":
    import sys
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    res = fetch_season_results(yr)
    rounds = sorted(res["Round"].unique())
    print(f"{yr}: cached {len(res)} result rows over rounds {rounds[0]}-{rounds[-1]}")
    table = standings(yr, results=res)
    print(f"\n{'':3} {'DRV':4} {'TEAM':16} {'PTS':>6} {'W':>3} {'POD':>4}")
    for r in table:
        print(f"{r['pos']:>2}. {r['code']:4} {r['team'][:16]:16} "
              f"{r['points']:>6.0f} {r['wins']:>3} {r['podiums']:>4}")
