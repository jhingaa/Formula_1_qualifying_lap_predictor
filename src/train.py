"""Train and evaluate a lap-time model (quali or race) with honest validation.

Validation is TIME-BASED: train on earlier seasons, test on a later one. A
random split would leak the future into the past (the same event appears in
both sets) and flatter the model dishonestly — the #1 mistake in motorsport ML.

Run:
    python train.py --kind quali --train-years 2023 --test-years 2024
    python train.py --kind race  --train-years 2023 --test-years 2024
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

import config
from features import build_pipeline, feature_columns

DATASET_PATHS = {
    "quali": config.PROCESSED_DIR / "quali_dataset.csv",
    "race": config.PROCESSED_DIR / "race_dataset.csv",
}


def _naive_baseline(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    """Predict the per-circuit mean lap time learned from training data."""
    per_track = train.groupby("Circuit")[target].mean()
    global_mean = train[target].mean()
    return test["Circuit"].map(per_track).fillna(global_mean).to_numpy()


def train_and_evaluate(kind: str, train_years: list[int], test_years: list[int]):
    cat, num, target = feature_columns(kind)
    df = pd.read_csv(DATASET_PATHS[kind])

    # Drop rows missing the target or any feature we rely on.
    df = df.dropna(subset=[target])
    features = cat + num

    train = df[df["Year"].isin(train_years)].copy()
    test = df[df["Year"].isin(test_years)].copy()
    if train.empty or test.empty:
        raise SystemExit(
            f"Empty split: train={len(train)} test={len(test)}. "
            f"Check the dataset covers {train_years} and {test_years}."
        )

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    pipe = build_pipeline(kind)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    base = _naive_baseline(train, test, target)

    mae = mean_absolute_error(y_test, pred)
    base_mae = mean_absolute_error(y_test, base)
    r2 = r2_score(y_test, pred)

    print(f"\n{'='*56}\n {kind.upper()} MODEL  |  train {train_years} -> test {test_years}\n{'='*56}")
    print(f" rows: train={len(train):,}  test={len(test):,}")
    print(f" naive per-track baseline MAE : {base_mae:6.3f} s")
    print(f" model MAE                    : {mae:6.3f} s   ({(1-mae/base_mae)*100:4.1f}% better)")
    print(f" model R^2                    : {r2:6.3f}")

    # Per-track breakdown (worst 8 by model MAE).
    tt = test.copy()
    tt["pred"] = pred
    tt["abs_err"] = (tt[target] - tt["pred"]).abs()
    by_track = (tt.groupby("Circuit")["abs_err"].mean()
                  .sort_values(ascending=False).head(8))
    print("\n Worst tracks by MAE (s):")
    for trk, v in by_track.items():
        print(f"   {trk:<28} {v:5.3f}")

    # Save model + metadata.
    out = config.MODELS_DIR / f"{kind}_model.joblib"
    joblib.dump({"pipeline": pipe, "features": features, "target": target,
                 "kind": kind, "train_years": train_years}, out)
    print(f"\n Saved model -> {out}")
    return pipe


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["quali", "race"], required=True)
    p.add_argument("--train-years", type=int, nargs="+", required=True)
    p.add_argument("--test-years", type=int, nargs="+", required=True)
    args = p.parse_args()
    train_and_evaluate(args.kind, args.train_years, args.test_years)
