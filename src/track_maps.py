"""Generate real circuit outlines from FastF1 position telemetry.

The fastest lap's X/Y trace *is* the track shape. We load one session per
circuit, take the fastest lap's position data, normalise it into a fixed
viewBox, and emit an SVG path. Cached to web/data/track_maps.json so the
frontend just reads a path string.

Run:  python track_maps.py            # builds maps for all circuits in the dataset
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

import config
import fastf1

fastf1.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT = config.PROJECT_ROOT / "web" / "data" / "track_maps.json"
VIEW_W, VIEW_H, PAD = 200.0, 120.0, 12.0


def _svg_path(x: np.ndarray, y: np.ndarray) -> str:
    """Normalise X/Y into the viewBox (preserving aspect) and build a path."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    # Flip Y: telemetry Y is 'up', SVG Y is 'down'.
    y = -y
    x -= x.min(); y -= y.min()
    w = max(x.max(), 1e-6); h = max(y.max(), 1e-6)
    scale = min((VIEW_W - 2 * PAD) / w, (VIEW_H - 2 * PAD) / h)
    x = x * scale; y = y * scale
    # Centre inside the viewBox.
    x += (VIEW_W - x.max()) / 2
    y += (VIEW_H - y.max()) / 2
    # Downsample to keep the path light (~250 points max).
    n = len(x)
    step = max(1, n // 250)
    pts = [f"{x[i]:.1f},{y[i]:.1f}" for i in range(0, n, step)]
    return "M" + " L".join(pts) + " Z"


def _latest_session_per_circuit(quali: pd.DataFrame) -> pd.DataFrame:
    """One representative (Circuit, Year, Round, EventName) — most recent."""
    idx = quali.sort_values(["Circuit", "Year", "Round"]).groupby("Circuit").tail(1)
    return idx[["Circuit", "Year", "Round", "EventName"]].reset_index(drop=True)


def build_track_maps() -> dict:
    quali = pd.read_csv(config.PROCESSED_DIR / "quali_dataset.csv")
    reps = _latest_session_per_circuit(quali)

    # Reuse any maps we already built (so re-runs skip cached circuits).
    maps: dict = {}
    if OUT.exists():
        maps = json.loads(OUT.read_text())

    for _, r in reps.iterrows():
        circuit = r["Circuit"]
        if circuit in maps and maps[circuit].get("path"):
            print(f"  have {circuit}")
            continue
        try:
            s = fastf1.get_session(int(r["Year"]), int(r["Round"]), "Q")
            s.load(telemetry=True, weather=False, messages=False, laps=True)
            lap = s.laps.pick_fastest()
            tel = lap.get_pos_data()
            path = _svg_path(tel["X"].to_numpy(), tel["Y"].to_numpy())
            maps[circuit] = {"path": path, "viewBox": f"0 0 {int(VIEW_W)} {int(VIEW_H)}",
                             "event": r["EventName"]}
            print(f"  built {circuit} ({len(tel)} pts)")
            OUT.write_text(json.dumps(maps, indent=1))  # save incrementally
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {circuit}: {type(e).__name__}: {str(e)[:60]}")

    OUT.write_text(json.dumps(maps, indent=1))
    print(f"\nSaved {len(maps)} track maps -> {OUT}")
    return maps


if __name__ == "__main__":
    build_track_maps()
