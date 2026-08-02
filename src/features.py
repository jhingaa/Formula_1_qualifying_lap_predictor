"""Feature definitions and the model pipeline, shared by quali and race models.

Design choices:
  * Categoricals (Driver, Team, Circuit, Compound) go through TargetEncoder.
    It cross-fits internally to avoid leakage, and — crucially for live use —
    maps an unseen category (e.g. a rookie mid-season) to the global mean
    instead of crashing.
  * The estimator is HistGradientBoostingRegressor: fast, strong on tabular
    data, ships with scikit-learn (no xgboost/lightgbm wheels needed).
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

# --- Feature schemas ---------------------------------------------------------
QUALI_CAT = ["Driver", "Team", "Circuit", "Compound"]
QUALI_NUM = ["AirTemp", "TrackTemp", "Humidity", "WindSpeed", "Rainfall"]
QUALI_TARGET = "BestLapTime"

RACE_CAT = ["Driver", "Team", "Circuit", "Compound"]
RACE_NUM = [
    "LapNumber",   # fuel-load proxy (fuel burns ~linearly)
    "TyreLife",    # tyre degradation
    "Position",    # track position / traffic
    "AirTemp", "TrackTemp", "Humidity", "WindSpeed", "Rainfall",
]
RACE_TARGET = "LapTime"


def feature_columns(kind: str) -> tuple[list[str], list[str], str]:
    """Return (categorical, numeric, target) for 'quali' or 'race'."""
    if kind == "quali":
        return QUALI_CAT, QUALI_NUM, QUALI_TARGET
    if kind == "race":
        return RACE_CAT, RACE_NUM, RACE_TARGET
    raise ValueError(f"unknown kind: {kind}")


def build_pipeline(kind: str) -> Pipeline:
    cat, num, _ = feature_columns(kind)
    pre = ColumnTransformer(
        transformers=[
            # Seeded: TargetEncoder shuffles its internal cross-fitting, so
            # without this the reported accuracy drifts between runs.
            ("cat", TargetEncoder(target_type="continuous", random_state=42), cat),
            ("num", "passthrough", num),
        ]
    )
    model = HistGradientBoostingRegressor(
        loss="absolute_error",   # optimise MAE directly (robust to outliers)
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("model", model)])
