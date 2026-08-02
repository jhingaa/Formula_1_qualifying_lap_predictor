"""Load a trained model and score laps — works for historical OR live data.

The same `predict_laps` function powers offline evaluation and the live monitor,
because both feed a DataFrame shaped like data_loader's output.
"""
from __future__ import annotations

import joblib
import pandas as pd

import config


def load_model(kind: str) -> dict:
    """Load a trained model bundle ('quali' or 'race')."""
    path = config.MODELS_DIR / f"{kind}_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No trained {kind} model at {path}. Run train.py first.")
    return joblib.load(path)


def predict_laps(bundle: dict, laps: pd.DataFrame) -> pd.Series:
    """Predict lap time (seconds) for each row of a data_loader-shaped frame."""
    feats = bundle["features"]
    missing = [c for c in feats if c not in laps.columns]
    if missing:
        raise ValueError(f"laps frame missing feature columns: {missing}")
    preds = bundle["pipeline"].predict(laps[feats])
    return pd.Series(preds, index=laps.index, name="PredLapTime")


def predict_next_lap(bundle: dict, laps: pd.DataFrame) -> pd.DataFrame:
    """For each driver, predict their NEXT lap from their latest completed lap.

    We roll conditions forward one lap: LapNumber+1, TyreLife+1 (fuel burns,
    tyre ages), holding compound/position/weather at their latest values.
    Returns one row per driver with LatestLap and PredNextLap.
    """
    feats = bundle["features"]
    if laps.empty:
        return pd.DataFrame()

    latest = (laps.sort_values("LapNumber")
                  .groupby("Driver", as_index=False)
                  .tail(1)
                  .copy())
    nxt = latest.copy()
    if "LapNumber" in nxt:
        nxt["LapNumber"] = nxt["LapNumber"] + 1
    if "TyreLife" in nxt:
        nxt["TyreLife"] = nxt["TyreLife"] + 1

    latest["PredNextLap"] = bundle["pipeline"].predict(nxt[feats])
    cols = [c for c in ["Driver", "Team", "Compound", "TyreLife",
                        "LapNumber", "Position"] if c in latest.columns]
    out = latest[cols].rename(columns={"LapNumber": "LastLap"})
    out["PredNextLap"] = latest["PredNextLap"].values
    return out.sort_values("PredNextLap").reset_index(drop=True)
