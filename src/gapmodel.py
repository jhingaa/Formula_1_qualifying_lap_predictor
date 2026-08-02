"""Predict qualifying pace from practice pace, validated leave-one-round-out.

Why a second model at all? The existing quali model predicts an ABSOLUTE lap
time from circuit identity and weather. That is fine for showing a number, but
it cannot be honestly cross-validated with a single season of data: hold out a
round and the circuit becomes unseen, so the model falls back to a global mean
and the error explodes. It also has no idea what happened in practice.

So for anything that depends on knowing how wrong we are — the confidence
figures, and the call on who takes pole — we model a RELATIVE target instead:
each driver's qualifying gap to pole, in percent. That target is
circuit-agnostic, which makes leave-one-round-out validation meaningful, and it
is exactly the quantity that decides the running order.

Absolute times are recovered at the end by predicting the pole lap time from the
weekend's practice benchmark and scaling the predicted gaps onto it.

Run `python gapmodel.py` to reproduce the model comparison below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

import config
import practice as P

QUALI_PATH = config.PROCESSED_DIR / "quali_dataset.csv"

CAT_FEATURES = ["Driver", "Team"]
NUM_FEATURES = [
    "PracGapPct",       # best relative practice lap of the weekend
    "PracGapLast",      # relative pace in the final dry practice session
    "PracGapMean",      # average across sessions (heavy-fuel runs included)
    "PracGapSoft",      # best lap restricted to soft-tyre runs (quali sims)
    "PracCompoundRank", # 0 = the best lap was on softs, higher = harder tyre
    "PracSessions",     # how many practice sessions the driver appeared in
    "PracLaps",         # total clean laps — how much evidence we have
    "TeamFormPct",      # team's season-to-date median quali gap to pole
    "DriverFormPct",    # driver's season-to-date median quali gap to pole
]
TARGET = "QualiGapPct"


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------
def _season_form(quali: pd.DataFrame, year: int, before_round: int) -> tuple[dict, dict]:
    """Median quali gap to pole per driver and per team, using earlier rounds only.

    Restricting to `before_round` is what keeps this honest: a driver's form
    feature for round 8 never sees round 8 or anything after it.
    """
    q = quali[(quali["Year"] == year) & (quali["Round"] < before_round)]
    if q.empty:
        return {}, {}
    q = q.copy()
    q["Pole"] = q.groupby("Round")["BestLapTime"].transform("min")
    q["GapPct"] = (q["BestLapTime"] / q["Pole"] - 1.0) * 100.0
    return (q.groupby("Driver")["GapPct"].median().to_dict(),
            q.groupby("Team")["GapPct"].median().to_dict())


def _race_lineup(quali: pd.DataFrame, year: int, rnd: int) -> set[str]:
    """Drivers who actually take part in qualifying at this round.

    Practice entry lists are not the race entry list: teams must hand FP1 to a
    rookie several times a season, so the practice data contains drivers who
    will never set a qualifying lap. Including them would invent grid slots.
    Prefer this round's own qualifying entry list; for a round that hasn't run
    yet, fall back to the most recent completed one.
    """
    here = quali[(quali["Year"] == year) & (quali["Round"] == rnd)]["Driver"]
    if len(here):
        return set(here)
    prior = quali[(quali["Year"] == year) & (quali["Round"] < rnd)]
    if prior.empty:
        prior = quali
    if prior.empty:
        return set()
    return set(prior[prior["Round"] == prior["Round"].max()]["Driver"])


def build_feature_table(prac: pd.DataFrame = None,
                        quali: pd.DataFrame = None) -> pd.DataFrame:
    """One row per driver per event: practice-derived features + season form."""
    prac = P.load_practice() if prac is None else prac
    quali = pd.read_csv(QUALI_PATH) if quali is None else quali

    q = quali.copy()
    q["Pole"] = q.groupby(["Year", "Round"])["BestLapTime"].transform("min")
    q[TARGET] = (q["BestLapTime"] / q["Pole"] - 1.0) * 100.0

    rows = []
    events = prac[["Year", "Round", "EventName"]].drop_duplicates()
    for _, e in events.iterrows():
        year, rnd = int(e["Year"]), int(e["Round"])
        ev = P.event_practice(prac, year, str(e["EventName"]))
        dry = ev[~ev["SessionWet"]]
        if dry.empty:
            dry = ev  # a fully wet weekend — use it, but flag it below
        last_order = dry["SessionOrder"].max()
        last = dry[dry["SessionOrder"] == last_order]

        best_idx = dry.groupby("Driver")["GapPct"].idxmin()
        best = dry.loc[best_idx].set_index("Driver")
        last_gap = last.groupby("Driver")["GapPct"].min()
        mean_gap = dry.groupby("Driver")["GapPct"].mean()
        soft = dry[dry["Compound"] == "SOFT"]
        soft_gap = soft.groupby("Driver")["GapPct"].min() if not soft.empty else {}
        sessions = dry.groupby("Driver")["Session"].nunique()
        laps = dry.groupby("Driver")["CleanLaps"].sum()

        drv_form, team_form = _season_form(quali, year, rnd)
        truth = q[(q["Year"] == year) & (q["Round"] == rnd)].set_index("Driver")
        lineup = _race_lineup(quali, year, rnd)

        for drv in best.index:
            if lineup and drv not in lineup:
                continue  # FP1-only rookie, not a qualifying entrant
            b = best.loc[drv]
            rows.append({
                "Year": year, "Round": rnd, "EventName": str(e["EventName"]),
                "Circuit": str(b["Circuit"]), "Driver": drv, "Team": str(b["Team"]),
                "PracGapPct": float(b["GapPct"]),
                "PracGapLast": float(last_gap.get(drv, b["GapPct"])),
                "PracGapMean": float(mean_gap.get(drv, b["GapPct"])),
                # NaN when the driver never ran softs — HGB handles it natively
                # and the imputer covers it for ridge.
                "PracGapSoft": float(soft_gap.get(drv, np.nan)) if len(soft_gap) else np.nan,
                "PracCompoundRank": float(b["CompoundRank"]),
                "PracCompound": str(b["Compound"]),
                "PracSessions": int(sessions.get(drv, 1)),
                "PracLaps": float(laps.get(drv, np.nan)),
                "PracBestTime": float(b["BestLapTime"]),
                "WeekendWet": bool(ev["SessionWet"].any()),
                "TeamFormPct": team_form.get(str(b["Team"]), np.nan),
                "DriverFormPct": drv_form.get(drv, np.nan),
                TARGET: float(truth.loc[drv, TARGET]) if drv in truth.index else np.nan,
                "QualiTime": (float(truth.loc[drv, "BestLapTime"])
                              if drv in truth.index else np.nan),
                "PoleTime": (float(truth.loc[drv, "Pole"])
                             if drv in truth.index else np.nan),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Candidate estimators
# ---------------------------------------------------------------------------
def _pipeline(kind: str) -> Pipeline:
    """Estimators worth trying on ~240 rows. Small data rewards simple models,
    so a regularised linear fit is a genuine contender, not just a baseline."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, TargetEncoder

    if kind == "ridge":
        pre = ColumnTransformer([
            ("cat", TargetEncoder(target_type="continuous", random_state=42), CAT_FEATURES),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), NUM_FEATURES),
        ])
        return Pipeline([("pre", pre), ("model", Ridge(alpha=3.0))])

    if kind == "gbm":
        pre = ColumnTransformer([
            ("cat", TargetEncoder(target_type="continuous", random_state=42), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),  # HGB handles NaN natively
        ])
        model = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.05, max_iter=300,
            max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=1.0,
            early_stopping=False, random_state=42)
        return Pipeline([("pre", pre), ("model", model)])

    raise ValueError(kind)


def _baseline_practice(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Practice gap, rescaled by the slope that best maps it onto quali gap.

    Practice gaps are usually LARGER than qualifying gaps (not everyone runs a
    clean low-fuel lap), so a straight copy is biased — the slope corrects that.
    """
    x, y = train["PracGapPct"].to_numpy(), train[TARGET].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    return slope * test["PracGapPct"].to_numpy() + intercept


def _baseline_form(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Season form only — no practice data at all. This is the bar practice
    has to clear to be worth using."""
    med = train.groupby("Driver")[TARGET].median()
    glob = float(train[TARGET].median())
    return test["Driver"].map(med).fillna(glob).to_numpy()


# ---------------------------------------------------------------------------
# Leave-one-round-out validation
# ---------------------------------------------------------------------------
def loro_predictions(tab: pd.DataFrame, kind: str = "ridge") -> pd.DataFrame:
    """Out-of-fold predictions: for each round, train on every OTHER round.

    Each round is predicted by a model that has never seen it, which is the
    closest offline analogue of predicting an upcoming weekend.
    """
    df = tab.dropna(subset=[TARGET]).reset_index(drop=True).copy()
    feats = CAT_FEATURES + NUM_FEATURES
    df["oof"] = np.nan
    df["oof_prac"] = np.nan
    df["oof_form"] = np.nan
    for rnd in sorted(df["Round"].unique()):
        te = df["Round"] == rnd
        tr = ~te
        if tr.sum() < 20:
            continue
        pipe = _pipeline(kind)
        pipe.fit(df.loc[tr, feats], df.loc[tr, TARGET])
        df.loc[te, "oof"] = pipe.predict(df.loc[te, feats])
        df.loc[te, "oof_prac"] = _baseline_practice(df[tr], df[te])
        df.loc[te, "oof_form"] = _baseline_form(df[tr], df[te])
    return df


def compare_models(tab: pd.DataFrame = None) -> pd.DataFrame:
    """Leave-one-round-out scores for every candidate, on the same folds."""
    tab = build_feature_table() if tab is None else tab
    out = []
    ridge = loro_predictions(tab, "ridge")
    gbm = loro_predictions(tab, "gbm")
    cands = {
        "practice only (rescaled)": ridge["oof_prac"],
        "season form only": ridge["oof_form"],
        "ridge (practice + form)": ridge["oof"],
        "gbm (practice + form)": gbm["oof"],
    }
    y = ridge[TARGET].to_numpy()
    for name, pred in cands.items():
        p = np.asarray(pred, dtype=float)
        ok = np.isfinite(p) & np.isfinite(y)
        err = y[ok] - p[ok]
        # Did it put the right driver on pole? Judged per round.
        d = ridge.loc[ok].copy(); d["p"] = p[ok]
        hits, n = 0, 0
        for _, g in d.groupby("Round"):
            if len(g) < 5:
                continue
            n += 1
            hits += int(g.loc[g["p"].idxmin(), "Driver"] == g.loc[g[TARGET].idxmin(), "Driver"])
        out.append({
            "model": name,
            "MAE_gap%": round(float(np.abs(err).mean()), 4),
            "RMSE_gap%": round(float(np.sqrt((err ** 2).mean())), 4),
            "poleHit%": round(100 * hits / max(n, 1), 1),
            "n": int(ok.sum()),
        })
    return pd.DataFrame(out).sort_values("MAE_gap%").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pole lap time: practice benchmark -> qualifying benchmark
# ---------------------------------------------------------------------------
def pole_time_ratio(tab: pd.DataFrame) -> float:
    """Median pole_time / best_practice_time across events.

    Qualifying is almost always quicker than anything seen in practice — the
    track is at its most rubbered-in and engines run maximum power — so this
    ratio sits just below 1 and converts a practice benchmark into a pole
    estimate.
    """
    ev = tab.dropna(subset=["PoleTime"]).groupby(["Year", "Round"]).agg(
        pole=("PoleTime", "first"), prac=("PracBestTime", "min"))
    return float((ev["pole"] / ev["prac"]).median())


def fit(tab: pd.DataFrame = None, kind: str = "ridge") -> dict:
    """Train on everything available and return a bundle for live prediction."""
    tab = build_feature_table() if tab is None else tab
    df = tab.dropna(subset=[TARGET])
    pipe = _pipeline(kind)
    pipe.fit(df[CAT_FEATURES + NUM_FEATURES], df[TARGET])
    return {"pipeline": pipe, "features": CAT_FEATURES + NUM_FEATURES,
            "kind": kind, "poleRatio": pole_time_ratio(tab)}


if __name__ == "__main__":
    tab = build_feature_table()
    print(f"\nFeature table: {len(tab)} driver-events, "
          f"{tab['Round'].nunique()} rounds\n")
    print("Leave-one-round-out comparison (target = quali gap to pole, %):")
    print(compare_models(tab).to_string(index=False))
    print(f"\npole/practice time ratio: {pole_time_ratio(tab):.5f}")
