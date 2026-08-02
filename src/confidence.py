"""Per-circuit confidence, computed from how wrong the model actually is.

The old dashboard derived confidence from grid position alone
(`93 - pos * 2`), so every circuit produced an identical 93/91/89/... ladder and
the gauge always read 86%. It carried no information.

This module replaces that with numbers measured from leave-one-round-out
residuals:

  * sigma per circuit  — how far off the predicted qualifying gap actually lands
    at that circuit, split into the part that moves the whole field together and
    the part that reshuffles the order (only the second one affects ranking).
  * position confidence — Monte Carlo over the field: perturb every driver by
    that circuit's ordering sigma, re-rank, and count how often each driver lands
    where we predicted.
  * gauge value        — a documented blend of circuit accuracy, how cleanly the
    order separates, how much practice running we have, and conditions.

Run `python confidence.py` to print the per-circuit table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gapmodel as G

# Shrinkage strength for per-circuit sigma. Each circuit contributes only ~22
# residuals, which is far too few for a stable variance estimate, so we pull it
# toward the global value. k is in units of "pseudo-observations": at k=10 a
# circuit's own data outweighs the prior roughly 2:1.
SHRINK_K = 10.0

# Weights for the composite gauge. These are a judgement call, not fitted —
# stated here so the number can be argued with rather than reverse-engineered.
GAUGE_WEIGHTS = {
    "accuracy": 0.40,    # how well the model predicts pace at this circuit
    "resolution": 0.30,  # how cleanly the predicted order separates
    "data": 0.20,        # how much practice running we actually have
    "conditions": 0.10,  # dry and stable, or wet and unrepresentative
}

N_SIMS = 20000
SEED = 42


# ---------------------------------------------------------------------------
# Residual structure
# ---------------------------------------------------------------------------
def residual_stats(oof: pd.DataFrame) -> pd.DataFrame:
    """Per-circuit sigma from out-of-fold residuals.

    Two sigmas, because they answer different questions:

      sigmaTotal    spread of the raw residual — drives "will the lap time be
                    close to what we said".
      sigmaOrder    spread after removing each event's mean residual. If the
                    model is 0.3% slow for everyone at a circuit, the ORDER is
                    untouched; only the driver-to-driver scatter can reshuffle
                    it. This is the one the ranking simulation uses.
    """
    df = oof.dropna(subset=["oof", G.TARGET]).copy()
    df["resid"] = df[G.TARGET] - df["oof"]
    df["eventResid"] = df.groupby(["Year", "Round"])["resid"].transform("mean")
    df["idio"] = df["resid"] - df["eventResid"]

    g_total = float(df["resid"].std(ddof=1))
    g_order = float(df["idio"].std(ddof=1))

    rows = []
    for circuit, grp in df.groupby("Circuit"):
        n = len(grp)
        # Shrink the circuit's variance toward the global variance.
        v_tot = (n * grp["resid"].var(ddof=1) + SHRINK_K * g_total ** 2) / (n + SHRINK_K)
        v_ord = (n * grp["idio"].var(ddof=1) + SHRINK_K * g_order ** 2) / (n + SHRINK_K)
        # Out-of-fold R^2 at this circuit: the share of the field's pace spread
        # the model actually explains here. Negative means worse than the mean.
        ss_res = float((grp["resid"] ** 2).sum())
        ss_tot = float(((grp[G.TARGET] - grp[G.TARGET].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rows.append({
            "Circuit": circuit, "n": n,
            "sigmaTotal": float(np.sqrt(v_tot)),
            "sigmaOrder": float(np.sqrt(v_ord)),
            "mae": float(grp["resid"].abs().mean()),
            "r2": float(r2),
            "fieldSpread": float(grp[G.TARGET].std(ddof=1)),
            # Empirical 80% interval. Deliberately NOT +/- 1.28 sigma: lap-time
            # error is one-sided in practice, because a driver can lose half a
            # second to traffic or a mistake far more easily than they can find
            # one, so the upper tail is fatter. Taken globally — 22 residuals per
            # circuit cannot support a stable tail estimate.
            "q10": float(np.quantile(df["resid"], 0.10)),
            "q90": float(np.quantile(df["resid"], 0.90)),
        })
    out = pd.DataFrame(rows)
    out.attrs["globalSigmaTotal"] = g_total
    out.attrs["globalSigmaOrder"] = g_order
    return out


def circuit_sigma(stats: pd.DataFrame, circuit: str) -> dict:
    """Sigmas for one circuit, falling back to the global values if unseen."""
    row = stats[stats["Circuit"] == circuit]
    if row.empty:
        return {"sigmaTotal": stats.attrs["globalSigmaTotal"],
                "sigmaOrder": stats.attrs["globalSigmaOrder"],
                "r2": float(stats["r2"].median()),
                "mae": float(stats["mae"].median()),
                "fieldSpread": float(stats["fieldSpread"].median()),
                "q10": float(stats["q10"].median()),
                "q90": float(stats["q90"].median()),
                "n": 0, "known": False}
    r = row.iloc[0].to_dict()
    r["known"] = True
    return r


# ---------------------------------------------------------------------------
# Monte Carlo over the running order
# ---------------------------------------------------------------------------
def simulate_order(pred_gaps: np.ndarray, sigma_order: float,
                   n_sims: int = N_SIMS, rng=None) -> dict:
    """Perturb every driver by sigma and re-rank, many times.

    Returns, per driver: probability of pole, probability of landing exactly on
    the predicted position, and probability of landing within one place of it.

    Simulating is better than a closed-form gap formula here because position is
    a property of the whole field at once — a driver's finishing slot depends on
    what everyone else does, not just the nearest rival.
    """
    # Fresh generator per call rather than a shared module-level one: with a
    # shared RNG the numbers for a given event would depend on how many other
    # events happened to be simulated first.
    rng = rng or np.random.default_rng(SEED)
    n = len(pred_gaps)
    if n == 0:
        return {"pole": np.array([]), "exact": np.array([]), "within1": np.array([])}
    pred_rank = np.argsort(np.argsort(pred_gaps))          # 0 = predicted fastest
    draws = pred_gaps[None, :] + rng.normal(0, sigma_order, size=(n_sims, n))
    sim_rank = np.argsort(np.argsort(draws, axis=1), axis=1)
    diff = np.abs(sim_rank - pred_rank[None, :])
    return {
        "pole": (sim_rank == 0).mean(axis=0),
        "exact": (diff == 0).mean(axis=0),
        "within1": (diff <= 1).mean(axis=0),
    }


# ---------------------------------------------------------------------------
# Composite gauge
# ---------------------------------------------------------------------------
def _score_accuracy(sig: dict) -> float:
    """How well the model predicts pace at this circuit, as out-of-fold R^2.

    R^2 is already 'share of variance explained', so it maps to a 0-1 score
    directly. Clipped at 0 because a negative R^2 (worse than guessing the mean)
    is simply no confidence, not negative confidence.
    """
    return float(np.clip(sig["r2"], 0.0, 1.0))


def _score_resolution(sim: dict, top_n: int = 10) -> float:
    """Mean probability that a front-running driver lands within one place of
    the prediction. Low when the top of the field is bunched relative to sigma."""
    w1 = sim["within1"]
    if len(w1) == 0:
        return 0.0
    return float(np.mean(np.sort(w1)[::-1][:min(top_n, len(w1))]))


def _score_data(ev_practice: pd.DataFrame) -> float:
    """How much practice evidence this weekend actually provides.

    Three components, equally weighted: sessions run (a sprint weekend with only
    FP1 genuinely tells us less), median clean laps per driver against a
    reference of 12, and whether soft-tyre runs exist to read quali pace from.
    """
    if ev_practice is None or ev_practice.empty:
        return 0.0
    sessions = ev_practice["Session"].nunique() / 3.0
    laps = float(ev_practice.groupby("Driver")["CleanLaps"].sum().median())
    lap_score = min(1.0, laps / 12.0)
    soft = 1.0 if (ev_practice["Compound"] == "SOFT").any() else 0.4
    return float(np.clip((sessions + lap_score + soft) / 3.0, 0.0, 1.0))


def _score_conditions(ev_practice: pd.DataFrame) -> float:
    """Dry, representative practice scores 1. Wet sessions tell us little about
    dry qualifying pace, so each one costs a third of the score."""
    if ev_practice is None or ev_practice.empty:
        return 0.5
    per_session = ev_practice.groupby("Session")["SessionWet"].max()
    wet_frac = float(per_session.mean())
    return float(np.clip(1.0 - wet_frac, 0.0, 1.0))


def event_confidence(pred_gaps: np.ndarray, sig: dict,
                     ev_practice: pd.DataFrame) -> dict:
    """Everything the dashboard needs for one event.

    `drivers` holds per-driver probabilities in prediction order; `gauge` is the
    headline percentage with its components exposed so the UI can explain it.
    """
    sim = simulate_order(np.asarray(pred_gaps, dtype=float), sig["sigmaOrder"])
    parts = {
        "accuracy": _score_accuracy(sig),
        "resolution": _score_resolution(sim),
        "data": _score_data(ev_practice),
        "conditions": _score_conditions(ev_practice),
    }
    value = sum(GAUGE_WEIGHTS[k] * v for k, v in parts.items())
    return {
        "value": float(np.clip(value * 100.0, 0.0, 99.0)),
        "parts": {k: round(v * 100, 1) for k, v in parts.items()},
        "weights": GAUGE_WEIGHTS,
        "sigmaOrder": sig["sigmaOrder"],
        "sigmaTotal": sig["sigmaTotal"],
        "r2": sig["r2"],
        "mae": sig["mae"],
        "drivers": {
            "pole": sim["pole"], "exact": sim["exact"], "within1": sim["within1"],
        },
    }


def label_for(value: float) -> str:
    return ("High Confidence" if value >= 80 else
            "Moderate Confidence" if value >= 60 else "Low Confidence")


if __name__ == "__main__":
    tab = G.build_feature_table()
    oof = G.loro_predictions(tab, "ridge")
    stats = residual_stats(oof)
    print(f"\nGlobal sigma: total={stats.attrs['globalSigmaTotal']:.3f}%  "
          f"order={stats.attrs['globalSigmaOrder']:.3f}%\n")
    print("Per-circuit residual structure (leave-one-round-out):")
    print(stats.sort_values("sigmaOrder").to_string(index=False))

    prac = __import__("practice").load_practice()
    print("\nPer-circuit gauge confidence:")
    rows = []
    for _, e in tab[["Year", "Round", "EventName", "Circuit"]].drop_duplicates().iterrows():
        sub = oof[(oof["Round"] == e["Round"])]
        if sub.empty:
            continue
        ev_prac = prac[(prac["Year"] == e["Year"]) & (prac["Round"] == e["Round"])]
        sig = circuit_sigma(stats, str(e["Circuit"]))
        c = event_confidence(sub["oof"].to_numpy(), sig, ev_prac)
        rows.append({"round": int(e["Round"]), "event": str(e["EventName"])[:24],
                     "gauge": round(c["value"], 1), **c["parts"],
                     "sigmaOrder%": round(c["sigmaOrder"], 3)})
    print(pd.DataFrame(rows).to_string(index=False))
