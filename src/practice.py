"""Turn FP1/FP2/FP3 lap times into a usable pace signal.

Raw practice times are not comparable across sessions. The track rubbers in
through the weekend, air and track temperature swing by 15-20 degrees, and teams
run wildly different fuel loads. A 1:18.9 in FP1 and a 1:18.9 in FP3 mean
completely different things.

What IS comparable is a driver's gap to the fastest lap of the SAME session.
That cancels the session-wide effects (track state, weather, red flags) and
leaves relative pace, which is the thing that actually carries over to
qualifying. Everything here is built on that normalisation.

Run `python practice.py` to re-run the evaluation that picked the metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

PRACTICE_PATH = config.PROCESSED_DIR / "practice_dataset.csv"
QUALI_PATH = config.PROCESSED_DIR / "quali_dataset.csv"

# A qualifying simulation is run on the softest tyre available. A best lap set
# on a harder compound understates that driver's real one-lap pace, so we track
# which compound the lap came from rather than comparing times blindly.
COMPOUND_RANK = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}

# Practice sessions in the order they run, so "latest session" is well defined.
SESSION_ORDER = {"FP1": 1, "FP2": 2, "FP3": 3}


def load_practice(path=None) -> pd.DataFrame:
    """Practice dataset with team gaps repaired and per-session normalisation applied."""
    path = path or PRACTICE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No practice dataset at {path}. Run:\n"
            f"    python dataset.py --kind practice --years 2026")
    return add_session_gaps(fill_missing_teams(pd.read_csv(path)))


def fill_missing_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Recover the Team column where FastF1 returns it empty.

    Some practice sessions come back with no team entered at all (2026 rounds 5
    and 7 at the time of writing). Anything that groups by team — the top-4
    comparison especially — silently loses those events otherwise. Drivers do not
    change team mid-season, so the driver's team from any other session that year
    is a safe repair.
    """
    if "Team" not in df.columns or not df["Team"].isna().any():
        return df
    df = df.copy()
    known = (df.dropna(subset=["Team"])
               .groupby(["Year", "Driver"])["Team"]
               .agg(lambda s: s.mode().iloc[0]))
    keys = pd.MultiIndex.from_arrays([df["Year"], df["Driver"]])
    df["Team"] = df["Team"].fillna(pd.Series(keys.map(known), index=df.index))
    return df


def add_session_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Add each lap's gap to the fastest lap of the same session, in percent.

    Percent rather than seconds because circuits differ hugely in lap length:
    0.3s at Monaco (72s lap) is a much bigger deficit than 0.3s at Spa (104s).
    """
    df = df.copy()
    key = ["Year", "Round", "Session"]
    df["SessionBest"] = df.groupby(key)["BestLapTime"].transform("min")
    df["GapPct"] = (df["BestLapTime"] / df["SessionBest"] - 1.0) * 100.0
    df["CompoundRank"] = df["Compound"].map(COMPOUND_RANK).fillna(2.0)
    df["SessionOrder"] = df["Session"].map(SESSION_ORDER).fillna(0).astype(int)
    # A session where the fastest lap was set in the wet tells us nothing about
    # dry qualifying pace, so mark it and let callers drop it.
    wet = df.groupby(key)["Rainfall"].transform("max").astype(bool)
    df["SessionWet"] = wet
    return df


def event_practice(df: pd.DataFrame, year: int, event: str) -> pd.DataFrame:
    """All practice rows for one event."""
    return df[(df["Year"] == year) & (df["EventName"] == event)]


def driver_pace(ev: pd.DataFrame, dry_only: bool = True) -> pd.DataFrame:
    """Per-driver practice form at one event: their single best relative lap.

    Taking the minimum gap across the weekend (rather than an average) is
    deliberate: it picks out each driver's best low-fuel effort and ignores the
    heavy-fuel race runs and aborted laps that pollute a mean.
    """
    if dry_only and not ev["SessionWet"].all():
        ev = ev[~ev["SessionWet"]]
    if ev.empty:
        return pd.DataFrame()
    idx = ev.groupby("Driver")["GapPct"].idxmin()
    best = ev.loc[idx].copy()
    best["Sessions"] = best["Driver"].map(ev.groupby("Driver")["Session"].nunique())
    best["TotalCleanLaps"] = best["Driver"].map(ev.groupby("Driver")["CleanLaps"].sum())
    return best.sort_values("GapPct").reset_index(drop=True)


def session_breakdown(ev: pd.DataFrame) -> dict:
    """{driver: {FP1: {...}, FP2: {...}}} — the per-session detail for the UI."""
    out: dict[str, dict] = {}
    for _, r in ev.iterrows():
        out.setdefault(str(r["Driver"]), {})[str(r["Session"])] = {
            "time": round(float(r["BestLapTime"]), 3),
            "gapPct": round(float(r["GapPct"]), 3),
            "gap": round(float(r["BestLapTime"] - r["SessionBest"]), 3),
            "compound": str(r["Compound"]),
            "laps": int(r["CleanLaps"]) if pd.notna(r["CleanLaps"]) else 0,
        }
    return out


# ---------------------------------------------------------------------------
# Team strength — which teams are actually the top 4, from the data
# ---------------------------------------------------------------------------
def team_strength(quali: pd.DataFrame, year: int, before_round: int | None = None,
                  n: int = 4) -> pd.DataFrame:
    """Rank teams by season-to-date one-lap pace.

    Uses each driver's qualifying gap to pole, in percent, then takes the team's
    MEDIAN across the season. Median rather than mean so one wet session or one
    crashed-out lap doesn't drag a team's rating around.

    `before_round` restricts to earlier rounds, so ranking the top 4 for an event
    never uses that event's own result.
    """
    q = quali[quali["Year"] == year].copy()
    if before_round is not None:
        q = q[q["Round"] < before_round]
    if q.empty:  # first round of a season — fall back to the whole dataset
        q = quali.copy()
    q["Pole"] = q.groupby(["Year", "Round"])["BestLapTime"].transform("min")
    q["GapPct"] = (q["BestLapTime"] / q["Pole"] - 1.0) * 100.0
    agg = (q.groupby("Team")
             .agg(paceGapPct=("GapPct", "median"), rounds=("Round", "nunique"))
             .sort_values("paceGapPct").reset_index())
    agg["rank"] = np.arange(1, len(agg) + 1)
    agg["isTop"] = agg["rank"] <= n
    return agg


# ---------------------------------------------------------------------------
# Metric evaluation — how well does each practice metric predict qualifying?
# ---------------------------------------------------------------------------
def _quali_gaps(quali: pd.DataFrame) -> pd.DataFrame:
    q = quali.copy()
    q["Pole"] = q.groupby(["Year", "Round"])["BestLapTime"].transform("min")
    q["QualiGapPct"] = (q["BestLapTime"] / q["Pole"] - 1.0) * 100.0
    return q[["Year", "Round", "Driver", "QualiGapPct", "BestLapTime"]]


def _candidate_metrics(ev: pd.DataFrame) -> dict[str, pd.Series]:
    """Different ways of reducing a weekend's practice laps to one number."""
    dry = ev[~ev["SessionWet"]]
    if dry.empty:
        dry = ev
    last = dry[dry["SessionOrder"] == dry["SessionOrder"].max()]
    soft = dry[dry["Compound"] == "SOFT"]
    return {
        "best_all": dry.groupby("Driver")["GapPct"].min(),
        "last_session": last.groupby("Driver")["GapPct"].min(),
        "mean_all": dry.groupby("Driver")["GapPct"].mean(),
        "soft_only": (soft.groupby("Driver")["GapPct"].min()
                      if not soft.empty else dry.groupby("Driver")["GapPct"].min()),
    }


def evaluate_metrics() -> pd.DataFrame:
    """Score each candidate metric against what actually happened in qualifying.

    Reported per metric:
      corr      Spearman correlation with the real qualifying gap
      mae       mean absolute error predicting quali gap % (after a linear fit)
      poleHit   how often the practice-fastest driver took pole
      top3Hit   how often the practice-fastest driver qualified in the top 3
    """
    prac = load_practice()
    quali = pd.read_csv(QUALI_PATH)
    qg = _quali_gaps(quali)

    rows = []
    events = prac[["Year", "Round", "EventName"]].drop_duplicates()
    scores: dict[str, list] = {}
    for _, e in events.iterrows():
        ev = event_practice(prac, int(e["Year"]), str(e["EventName"]))
        truth = qg[(qg["Year"] == e["Year"]) & (qg["Round"] == e["Round"])]
        if ev.empty or truth.empty:
            continue
        truth = truth.set_index("Driver")
        for name, metric in _candidate_metrics(ev).items():
            common = metric.index.intersection(truth.index)
            if len(common) < 6:
                continue
            x = metric.loc[common].to_numpy()
            y = truth.loc[common, "QualiGapPct"].to_numpy()
            pole_driver = truth["QualiGapPct"].idxmin()
            order = truth.loc[common, "QualiGapPct"].rank()
            scores.setdefault(name, []).append({
                "x": x, "y": y,
                "pick": common[int(np.argmin(x))],
                "pole": pole_driver,
                "pickRank": float(order.loc[common[int(np.argmin(x))]]),
            })

    for name, recs in scores.items():
        x = np.concatenate([r["x"] for r in recs])
        y = np.concatenate([r["y"] for r in recs])
        # Spearman without scipy: Pearson correlation of the ranks.
        rx = pd.Series(x).rank().to_numpy(); ry = pd.Series(y).rank().to_numpy()
        corr = float(np.corrcoef(rx, ry)[0, 1])
        slope, intercept = np.polyfit(x, y, 1)
        mae = float(np.abs(y - (slope * x + intercept)).mean())
        rows.append({
            "metric": name,
            "corr": round(corr, 3),
            "mae": round(mae, 3),
            "poleHit": round(float(np.mean([r["pick"] == r["pole"] for r in recs])) * 100, 1),
            "top3Hit": round(float(np.mean([r["pickRank"] <= 3 for r in recs])) * 100, 1),
            "slope": round(float(slope), 3),
            "events": len(recs),
        })
    return pd.DataFrame(rows).sort_values("corr", ascending=False).reset_index(drop=True)


def reference_session(ev: pd.DataFrame, min_drivers: int = 8) -> str | None:
    """The practice session to judge the pole battle in.

    A driver's `GapPct` is measured against the best lap of their OWN session,
    so the minimum across the weekend is not comparable between drivers —
    whoever topped FP1, FP2 and FP3 all show 0.000%. To rank drivers against
    each other they have to be compared inside a single session.

    We take the latest dry session with enough of the field running, because the
    track is fastest and the fuel loads lowest by then, which is what qualifying
    actually resembles.
    """
    dry = ev[~ev["SessionWet"]]
    if dry.empty:
        dry = ev
    counts = dry.groupby("Session")["Driver"].nunique()
    ok = counts[counts >= min_drivers]
    if ok.empty:
        ok = counts
    if ok.empty:
        return None
    order = {s: SESSION_ORDER.get(s, 0) for s in ok.index}
    return max(order, key=order.get)


def pole_battle(ev: pd.DataFrame, teams: list[str],
                lineup: set[str] | None = None) -> list[dict]:
    """Head-to-head practice record for the drivers who can realistically take pole.

    Restricted to the top teams because the pace cliff behind them is large
    enough that a midfield pole would be a wet-session or red-flag accident, not
    something practice pace predicts. `teams` comes from `team_strength`, so
    which teams those are is measured, not assumed.

    `refGapPct` is the head-to-head number: every driver's gap to the fastest lap
    of one common reference session. `bestTime`/`bestSession` stay as supporting
    detail for drivers whose strongest lap came elsewhere in the weekend.
    """
    ev = ev[ev["Team"].isin(teams)]
    if lineup:
        # Drop rookies on mandatory FP1 outings — they are not qualifying.
        ev = ev[ev["Driver"].isin(lineup)]
    dry = ev[~ev["SessionWet"]]
    if dry.empty:
        dry = ev
    if dry.empty:
        return []

    ref = reference_session(dry)
    ref_rows = dry[dry["Session"] == ref] if ref else dry.iloc[0:0]
    # Re-normalise inside the reference session against the top teams present,
    # so the benchmark is the quickest car that actually ran, not a rookie's lap.
    ref_best = float(ref_rows["BestLapTime"].min()) if len(ref_rows) else None
    ref_gap = {}
    if ref_best:
        for _, r in ref_rows.iterrows():
            ref_gap[str(r["Driver"])] = (float(r["BestLapTime"]) / ref_best - 1.0) * 100.0

    detail = session_breakdown(dry)
    out = []
    for drv, grp in dry.groupby("Driver"):
        best = grp.loc[grp["GapPct"].idxmin()]
        out.append({
            "code": str(drv),
            "team": str(best["Team"]),
            "refSession": ref,
            "refGapPct": (round(ref_gap[str(drv)], 3) if str(drv) in ref_gap else None),
            "refTime": (round(float(ref_rows[ref_rows["Driver"] == drv]["BestLapTime"].iloc[0]), 3)
                        if str(drv) in ref_gap else None),
            "bestTime": round(float(best["BestLapTime"]), 3),
            "bestSession": str(best["Session"]),
            "bestCompound": str(best["Compound"]),
            "cleanLaps": int(grp["CleanLaps"].sum()) if grp["CleanLaps"].notna().any() else 0,
            "ranOnSoft": bool((grp["Compound"] == "SOFT").any()),
            "sessions": detail.get(str(drv), {}),
        })
    # Drivers absent from the reference session sort last — we have no
    # comparable lap for them.
    return sorted(out, key=lambda r: (r["refGapPct"] is None, r["refGapPct"] or 0))


if __name__ == "__main__":
    res = evaluate_metrics()
    print("\nHow well does each practice metric predict qualifying pace?")
    print(res.to_string(index=False))
    prac = load_practice()
    quali = pd.read_csv(QUALI_PATH)
    print("\nTeam pace ranking (season to date, median quali gap to pole):")
    print(team_strength(quali, int(quali['Year'].max())).to_string(index=False))
