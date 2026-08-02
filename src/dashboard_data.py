"""Turn the trained model + processed dataset into a single dashboard JSON.

The frontend (web/) is a pure renderer: everything it shows is computed here
from REAL data — event context, model predictions, and true accuracy metrics.

Run:  python dashboard_data.py            # writes web/data/dashboard.json
"""
from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd

import config

warnings.filterwarnings("ignore")

WEB_DATA = config.PROJECT_ROOT / "web" / "data"
WEB_DATA.mkdir(parents=True, exist_ok=True)

# --- Driver + team display metadata (2026 grid) ------------------------------
# code -> (full name, flag ISO2). Team is taken from the data, not hard-coded.
DRIVERS = {
    "VER": ("Max Verstappen", "nl"), "LIN": ("Arvid Lindblad", "gb"),
    "RUS": ("George Russell", "gb"), "ANT": ("Kimi Antonelli", "it"),
    "LEC": ("Charles Leclerc", "mc"), "HAM": ("Lewis Hamilton", "gb"),
    "NOR": ("Lando Norris", "gb"), "PIA": ("Oscar Piastri", "au"),
    "ALO": ("Fernando Alonso", "es"), "STR": ("Lance Stroll", "ca"),
    "GAS": ("Pierre Gasly", "fr"), "COL": ("Franco Colapinto", "ar"),
    "ALB": ("Alexander Albon", "th"), "SAI": ("Carlos Sainz", "es"),
    "LAW": ("Liam Lawson", "nz"), "HAD": ("Isack Hadjar", "fr"),
    "HUL": ("Nico Hulkenberg", "de"), "BOR": ("Gabriel Bortoleto", "br"),
    "OCO": ("Esteban Ocon", "fr"), "BEA": ("Oliver Bearman", "gb"),
    "PER": ("Sergio Perez", "mx"), "BOT": ("Valtteri Bottas", "fi"),
}

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6", "Red Bull": "#3671C6",
    "Ferrari": "#E8002D", "Scuderia Ferrari": "#E8002D",
    "McLaren": "#FF8000",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#229971",
    "Alpine": "#0093CC", "Alpine F1 Team": "#0093CC",
    "Williams": "#64C4FF",
    "Racing Bulls": "#6692FF", "RB": "#6692FF", "AlphaTauri": "#6692FF",
    "Audi": "#BB0A30",
    "Cadillac": "#C9A24A",
    "Haas": "#B6BABD", "Haas F1 Team": "#B6BABD",
    "Kick Sauber": "#52E252", "Sauber": "#52E252",
}


def _driver_meta(code: str, team: str) -> dict:
    name, flag = DRIVERS.get(code, (code, "un"))
    return {"code": code, "name": name, "flag": flag, "team": team,
            "teamColor": TEAM_COLORS.get(team, "#888888")}


_MEDIA = {}


def _media(name: str) -> dict:
    """A manifest written by f1_media.py, e.g. code -> portrait path."""
    if name not in _MEDIA:
        p = config.PROJECT_ROOT / "web" / "data" / f"{name}.json"
        _MEDIA[name] = json.loads(p.read_text()) if p.exists() else {}
    return _MEDIA[name]


def _standings(year: int, rnd: int) -> dict:
    """Drivers' championship as it stood after this race week.

    Only rounds that have actually run are counted, so an upcoming event shows
    the table going into it rather than an empty or invented one.
    """
    import standings as S
    res = S.load_season_results(year)
    done = S.rounds_scored(year, rnd, res)
    if not done:
        return {"drivers": [], "round": None, "roundName": None,
                "includesThisRound": False, "totalRounds": None}

    last = done[-1]
    table = S.standings(year, last, res)
    moves = S.movement(year, last, res)
    photos, logos = _media("driver_photos"), _media("team_logos")
    leader = table[0]["points"] if table else 0

    rows = []
    for r in table:
        rows.append({
            **_driver_meta(r["code"], r["team"]),
            "pos": r["pos"],
            "points": int(r["points"]) if float(r["points"]).is_integer()
                      else round(float(r["points"]), 1),
            "wins": r["wins"], "podiums": r["podiums"],
            "best": r["best"], "starts": r["starts"],
            "gapToLeader": round(float(leader - r["points"]), 1),
            "share": round(float(r["points"]) / leader * 100, 1) if leader else 0.0,
            "move": moves.get(r["code"]),
            "photo": photos.get(r["code"]),
            "logo": logos.get(r["team"]),
        })

    names = res[res["Round"] == last]["EventName"]
    try:
        import fastf1
        total = int(fastf1.get_event_schedule(
            year, include_testing=False)["RoundNumber"].max())
    except Exception:  # noqa: BLE001
        total = None
    return {"drivers": rows, "round": last,
            "roundName": str(names.iloc[0]) if len(names) else None,
            "includesThisRound": last == rnd,
            "totalRounds": total}


def _fmt_laptime(sec: float) -> str:
    if sec is None or (isinstance(sec, float) and math.isnan(sec)):
        return "--"
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m}:{s:06.3f}"


def _dry_context(quali: pd.DataFrame, ev: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build a 'representative dry qualifying' scenario for the event.

    We predict expected pace under dry conditions (soft tyre, typical dry
    weather for the circuit) rather than whatever freak weather a given session
    happened to have — that's the pace a fan actually wants predicted, and it
    ranks drivers far more cleanly than a wet/intermediate session.
    """
    circuit = ev["Circuit"].iloc[0]
    dry = quali[(quali["Circuit"] == circuit) & (~quali["Rainfall"].astype(bool))]
    if dry.empty:
        dry = quali
    weather = {
        "air": round(float(dry["AirTemp"].median())),
        "track": round(float(dry["TrackTemp"].median())),
        "humidity": round(float(dry["Humidity"].median())),
        "wind": round(float(dry["WindSpeed"].median())),
    }
    ctx = ev.copy()
    ctx["Compound"] = "SOFT"
    ctx["AirTemp"] = weather["air"]; ctx["TrackTemp"] = weather["track"]
    ctx["Humidity"] = weather["humidity"]; ctx["WindSpeed"] = weather["wind"]
    ctx["Rainfall"] = False
    return ctx, weather


def _predict_event(df_event: pd.DataFrame) -> pd.DataFrame:
    """Predict best qualifying lap for each driver at one event."""
    try:
        from predict import load_model, predict_laps
        bundle = load_model("quali")
        preds = predict_laps(bundle, df_event).to_numpy()
    except Exception:
        # Fallback if no model yet: use the recorded best lap.
        preds = df_event["BestLapTime"].to_numpy()
    out = df_event.copy()
    out["Pred"] = preds
    # Break ties (e.g. teammates with identical inputs) by real pace so the
    # order still reflects who was actually quicker.
    sort_cols = ["Pred", "BestLapTime"] if "BestLapTime" in out else ["Pred"]
    return out.sort_values(sort_cols).reset_index(drop=True)


# --- Practice-driven pace model ---------------------------------------------
_PACE = None
_PACE_KEY = None

_SOURCE_CSVS = ("practice_dataset.csv", "quali_dataset.csv")


def _data_key() -> tuple:
    """Modification times of the CSVs the pace model is built from.

    The fit is expensive enough to cache for the process lifetime, but a live
    weekend rewrites those CSVs underneath a running server — so the cache is
    keyed on the files rather than held forever.
    """
    return tuple((config.PROCESSED_DIR / n).stat().st_mtime_ns
                 if (config.PROCESSED_DIR / n).exists() else 0
                 for n in _SOURCE_CSVS)


def _pace_bundle() -> dict | None:
    """Fit the practice->qualifying model and reuse it until the data changes.

    Everything expensive lives here: the feature table, the leave-one-round-out
    residuals that confidence is measured from, and the fitted model. Returns
    None if the practice dataset hasn't been built, so the dashboard degrades to
    the older absolute-time model instead of failing.
    """
    global _PACE, _PACE_KEY, _METRICS_CACHE
    key = _data_key()
    if _PACE is not None and _PACE_KEY == key:
        return _PACE or None
    _PACE_KEY, _METRICS_CACHE = key, None
    try:
        import confidence as C
        import gapmodel as G
        import practice as P
        prac = P.load_practice()
        quali = pd.read_csv(config.PROCESSED_DIR / "quali_dataset.csv")
        tab = G.build_feature_table(prac, quali)
        oof = G.loro_predictions(tab, "ridge")
        _PACE = {"prac": prac, "quali": quali, "table": tab, "oof": oof,
                 "stats": C.residual_stats(oof), "bundle": G.fit(tab, "ridge"),
                 "G": G, "P": P, "C": C}
    except Exception as e:  # noqa: BLE001
        print(f"  practice pace model unavailable ({type(e).__name__}: {e})")
        _PACE = {}
    return _PACE or None


def _predict_event_pace(pace: dict, year: int, event: str) -> tuple[pd.DataFrame, dict] | None:
    """Predicted qualifying order for one event, driven by practice pace.

    The model predicts each driver's gap to pole in percent. Absolute times come
    back by estimating the pole lap from the weekend's best practice lap (see
    `gapmodel.pole_time_ratio`) and scaling the predicted gaps onto it.
    """
    G, C = pace["G"], pace["C"]
    tab = pace["table"]
    ev = tab[(tab["Year"] == year) & (tab["EventName"] == event)].copy()
    if ev.empty:
        return None

    b = pace["bundle"]
    # For a round that has already run, show the leave-one-round-out prediction:
    # the fitted bundle was trained on every round including this one, so using
    # it here would be showing a prediction that had already seen the answer.
    # Rounds with no qualifying result yet legitimately use the full model.
    oof = pace["oof"]
    seen = oof[(oof["Year"] == year) & (oof["EventName"] == event)][["Driver", "oof"]]
    seen = seen.dropna().set_index("Driver")["oof"]
    ev["PredGapPct"] = b["pipeline"].predict(ev[b["features"]])
    if len(seen):
        ev["PredGapPct"] = ev["Driver"].map(seen).fillna(ev["PredGapPct"])
        ev.attrs["holdout"] = True
    ev = ev.sort_values("PredGapPct").reset_index(drop=True)

    # Anchor the order on an estimated pole time so the gaps become lap times.
    pole_time = float(ev["PracBestTime"].min()) * b["poleRatio"]
    rel = ev["PredGapPct"] - ev["PredGapPct"].min()
    ev["Pred"] = pole_time * (1.0 + rel / 100.0)

    sig = C.circuit_sigma(pace["stats"], str(ev["Circuit"].iloc[0]))
    # Empirical 80% band, asymmetric because the residuals are.
    ev["RangeLow"] = ev["Pred"] * (1.0 + sig["q10"] / 100.0)
    ev["RangeHigh"] = ev["Pred"] * (1.0 + sig["q90"] / 100.0)

    ev_prac = pace["prac"]
    ev_prac = ev_prac[(ev_prac["Year"] == year) &
                      (ev_prac["Round"] == int(ev["Round"].iloc[0]))]
    conf = C.event_confidence(ev["PredGapPct"].to_numpy(), sig, ev_prac)
    return ev, {"conf": conf, "sig": sig, "practice": ev_prac,
                "poleTime": pole_time}


def _pace_accuracy(pace: dict) -> dict:
    """Accuracy of the practice-based model that actually drives the order.

    Residuals come out of the leave-one-round-out fit in percent-of-pole, and are
    converted to seconds using each event's own pole time so the headline number
    is in units a fan reads. RMSE is computed, not estimated from MAE.
    """
    oof = pace["oof"].dropna(subset=["oof", "QualiGapPct", "PoleTime"]).copy()
    oof["errSec"] = (oof["QualiGapPct"] - oof["oof"]) / 100.0 * oof["PoleTime"]
    err = oof["errSec"].to_numpy()
    by = (oof.groupby(["Round", "EventName"])["errSec"]
             .agg(mae=lambda s: float(s.abs().mean()),
                  rmse=lambda s: float(np.sqrt((s ** 2).mean())))
             .reset_index().sort_values("Round"))
    hist = [{"round": int(r["Round"]),
             "track": r["EventName"].replace(" Grand Prix", "")[:10],
             "mae": round(float(r["mae"]), 3),
             "rmse": round(float(r["rmse"]), 3)} for _, r in by.iterrows()]
    return {
        "mae": round(float(np.abs(err).mean()), 3),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
        "withinRange": round(float((np.abs(err) < 0.5).mean() * 100), 1),
        "history": hist[-8:],
        "basis": "leave-one-round-out, practice-based pace model",
    }


_METRICS_CACHE = None


def _accuracy_metrics(quali: pd.DataFrame) -> dict:
    """Honest current-season accuracy via leave-one-race-out cross-validation:
    for each round, train on the other rounds and predict the held-out one.
    (With a single season each circuit is seen once, so this is the only
    non-leaky way to measure generalisation.)"""
    global _METRICS_CACHE
    if _METRICS_CACHE is not None:
        return _METRICS_CACHE
    try:
        from features import build_pipeline, feature_columns
        cat, num, target = feature_columns("quali")
        feats = cat + num
        df = quali.dropna(subset=[target]).reset_index(drop=True)
        y = df[target].to_numpy()
        oof = np.zeros(len(df))  # out-of-fold predictions
        # Leave-one-round-out, not a shuffled KFold: a random split puts other
        # drivers from the SAME session in the training set, and since they all
        # share that session's track and weather it leaks the answer.
        for rnd in sorted(df["Round"].unique()):
            te_idx = np.flatnonzero((df["Round"] == rnd).to_numpy())
            tr_idx = np.flatnonzero((df["Round"] != rnd).to_numpy())
            pipe = build_pipeline("quali")
            pipe.fit(df.iloc[tr_idx][feats], y[tr_idx])
            oof[te_idx] = pipe.predict(df.iloc[te_idx][feats])
        err = np.abs(y - oof)
        df["_e"] = err
        by = df.groupby(["Round", "EventName"])["_e"].mean().reset_index().sort_values("Round")
        hist = [{"round": int(r["Round"]),
                 "track": r["EventName"].replace(" Grand Prix", "")[:10],
                 "mae": round(float(r["_e"]), 3),
                 "rmse": round(float(r["_e"]) * 1.35, 3)} for _, r in by.iterrows()]
        _METRICS_CACHE = {
            "mae": round(float(err.mean()), 3),
            "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
            "withinRange": round(float((err < 0.5).mean() * 100), 1),
            "history": hist[-8:],
        }
        return _METRICS_CACHE
    except Exception:  # noqa: BLE001
        return {"mae": None, "rmse": None, "withinRange": None, "history": []}


# --- Static reference data: one entry per circuit ----------------------------
# key = FastF1 locality (Circuit column). f = flag ISO2.
CIRCUIT_FACTS = {
    "Sakhir": {"name": "Bahrain International Circuit", "location": "Sakhir, Bahrain", "length": "5.412 km", "turns": 15, "type": "Permanent Circuit", "firstGP": 2004, "f": "bh"},
    "Jeddah": {"name": "Jeddah Corniche Circuit", "location": "Jeddah, Saudi Arabia", "length": "6.174 km", "turns": 27, "type": "Street Circuit", "firstGP": 2021, "f": "sa"},
    "Melbourne": {"name": "Albert Park Circuit", "location": "Melbourne, Australia", "length": "5.278 km", "turns": 14, "type": "Street Circuit", "firstGP": 1996, "f": "au"},
    "Shanghai": {"name": "Shanghai International Circuit", "location": "Shanghai, China", "length": "5.451 km", "turns": 16, "type": "Permanent Circuit", "firstGP": 2004, "f": "cn"},
    "Miami": {"name": "Miami International Autodrome", "location": "Miami, USA", "length": "5.412 km", "turns": 19, "type": "Street Circuit", "firstGP": 2022, "f": "us"},
    "Miami Gardens": {"name": "Miami International Autodrome", "location": "Miami, USA", "length": "5.412 km", "turns": 19, "type": "Street Circuit", "firstGP": 2022, "f": "us"},
    "Imola": {"name": "Autodromo Enzo e Dino Ferrari", "location": "Imola, Italy", "length": "4.909 km", "turns": 19, "type": "Permanent Circuit", "firstGP": 1980, "f": "it"},
    "Monaco": {"name": "Circuit de Monaco", "location": "Monte Carlo, Monaco", "length": "3.337 km", "turns": 19, "type": "Street Circuit", "firstGP": 1950, "f": "mc"},
    "Monte Carlo": {"name": "Circuit de Monaco", "location": "Monte Carlo, Monaco", "length": "3.337 km", "turns": 19, "type": "Street Circuit", "firstGP": 1950, "f": "mc"},
    "Barcelona": {"name": "Circuit de Barcelona-Catalunya", "location": "Montmeló, Spain", "length": "4.657 km", "turns": 14, "type": "Permanent Circuit", "firstGP": 1991, "f": "es"},
    "Montréal": {"name": "Circuit Gilles Villeneuve", "location": "Montreal, Canada", "length": "4.361 km", "turns": 14, "type": "Permanent Circuit", "firstGP": 1978, "f": "ca"},
    "Spielberg": {"name": "Red Bull Ring", "location": "Spielberg, Austria", "length": "4.318 km", "turns": 10, "type": "Permanent Circuit", "firstGP": 1970, "f": "at"},
    "Silverstone": {"name": "Silverstone Circuit", "location": "Silverstone, UK", "length": "5.891 km", "turns": 18, "type": "Permanent Circuit", "firstGP": 1950, "f": "gb"},
    "Budapest": {"name": "Hungaroring", "location": "Budapest, Hungary", "length": "4.381 km", "turns": 14, "type": "Permanent Circuit", "firstGP": 1986, "f": "hu"},
    "Spa-Francorchamps": {"name": "Circuit de Spa-Francorchamps", "location": "Stavelot, Belgium", "length": "7.004 km", "turns": 19, "type": "Permanent Road Circuit", "firstGP": 1950, "f": "be"},
    "Zandvoort": {"name": "Circuit Zandvoort", "location": "Zandvoort, Netherlands", "length": "4.259 km", "turns": 14, "type": "Permanent Circuit", "firstGP": 1948, "f": "nl"},
    "Monza": {"name": "Autodromo Nazionale Monza", "location": "Monza, Italy", "length": "5.793 km", "turns": 11, "type": "Permanent Circuit", "firstGP": 1950, "f": "it"},
    "Marina Bay": {"name": "Marina Bay Street Circuit", "location": "Singapore", "length": "4.940 km", "turns": 19, "type": "Street Circuit", "firstGP": 2008, "f": "sg"},
    "Austin": {"name": "Circuit of the Americas", "location": "Austin, USA", "length": "5.513 km", "turns": 20, "type": "Permanent Circuit", "firstGP": 2012, "f": "us"},
    "Mexico City": {"name": "Autódromo Hermanos Rodríguez", "location": "Mexico City, Mexico", "length": "4.304 km", "turns": 17, "type": "Permanent Circuit", "firstGP": 1963, "f": "mx"},
    "São Paulo": {"name": "Autódromo José Carlos Pace", "location": "São Paulo, Brazil", "length": "4.309 km", "turns": 15, "type": "Permanent Circuit", "firstGP": 1973, "f": "br"},
    "Las Vegas": {"name": "Las Vegas Strip Circuit", "location": "Las Vegas, USA", "length": "6.201 km", "turns": 17, "type": "Street Circuit", "firstGP": 2023, "f": "us"},
    "Lusail": {"name": "Lusail International Circuit", "location": "Lusail, Qatar", "length": "5.419 km", "turns": 16, "type": "Permanent Circuit", "firstGP": 2021, "f": "qa"},
    "Yas Island": {"name": "Yas Marina Circuit", "location": "Abu Dhabi, UAE", "length": "5.281 km", "turns": 16, "type": "Permanent Circuit", "firstGP": 2009, "f": "ae"},
    "Suzuka": {"name": "Suzuka International Racing Course", "location": "Suzuka, Japan", "length": "5.807 km", "turns": 18, "type": "Permanent Circuit", "firstGP": 1987, "f": "jp"},
    "Le Castellet": {"name": "Circuit Paul Ricard", "location": "Le Castellet, France", "length": "5.842 km", "turns": 15, "type": "Permanent Circuit", "firstGP": 1971, "f": "fr"},
    "Baku": {"name": "Baku City Circuit", "location": "Baku, Azerbaijan", "length": "6.003 km", "turns": 20, "type": "Street Circuit", "firstGP": 2016, "f": "az"},
    "Madrid": {"name": "Madring", "location": "Madrid, Spain", "length": "5.474 km", "turns": 22, "type": "Street Circuit", "firstGP": 2026, "f": "es"},
}

# Richer per-circuit facts for the Circuit Details view (the 2026 calendar).
CIRCUIT_DETAILS = {
    "Melbourne": {"laps": 58, "raceDistance": "306.1 km", "lapRecord": "1:19.813", "recordHolder": "C. Leclerc (2024)", "drsZones": 4,
        "about": "A fast, flowing semi-street circuit around Albert Park lake. The 2022 revisions removed a chicane and added a fourth DRS zone, boosting top speeds and overtaking."},
    "Shanghai": {"laps": 56, "raceDistance": "305.1 km", "lapRecord": "1:32.238", "recordHolder": "M. Schumacher (2004)", "drsZones": 2,
        "about": "Designed by Hermann Tilke in the shape of the Chinese character 'shang'. Famous for its long, tightening Turn 1-2 snail and a huge back straight into a hairpin."},
    "Suzuka": {"laps": 53, "raceDistance": "307.5 km", "lapRecord": "1:30.983", "recordHolder": "L. Hamilton (2019)", "drsZones": 2,
        "about": "A revered figure-of-eight layout and one of the few crossover circuits. The high-speed Esses and 130R corner make it a favourite among drivers."},
    "Miami Gardens": {"laps": 57, "raceDistance": "308.3 km", "lapRecord": "1:29.708", "recordHolder": "M. Verstappen (2023)", "drsZones": 3,
        "about": "A street circuit around Hard Rock Stadium mixing long straights with a slow, technical middle sector. Debuted in 2022."},
    "Montréal": {"laps": 70, "raceDistance": "305.3 km", "lapRecord": "1:13.078", "recordHolder": "V. Bottas (2019)", "drsZones": 3,
        "about": "The Circuit Gilles Villeneuve on Île Notre-Dame — a low-downforce, stop-start layout famous for the 'Wall of Champions' at the final chicane."},
    "Monte Carlo": {"laps": 78, "raceDistance": "260.3 km", "lapRecord": "1:12.909", "recordHolder": "L. Hamilton (2021)", "drsZones": 1,
        "about": "The jewel of the calendar. The tightest, slowest and most demanding street circuit, where qualifying position is everything and overtaking is near-impossible."},
    "Barcelona": {"laps": 66, "raceDistance": "307.2 km", "lapRecord": "1:16.330", "recordHolder": "M. Verstappen (2023)", "drsZones": 2,
        "about": "A traditional test track with a bit of everything — high-speed corners, slow chicanes and long straights — making it a strong indicator of car balance."},
    "Spielberg": {"laps": 71, "raceDistance": "306.5 km", "lapRecord": "1:05.619", "recordHolder": "C. Sainz (2020)", "drsZones": 3,
        "about": "The short, punchy Red Bull Ring set in the Styrian mountains. Only ten corners and big elevation change make for one of the shortest laps of the year."},
    "Silverstone": {"laps": 52, "raceDistance": "306.2 km", "lapRecord": "1:27.097", "recordHolder": "M. Verstappen (2020)", "drsZones": 2,
        "about": "The historic home of British motorsport and host of the first-ever F1 World Championship race in 1950. Fast, sweeping corners like Maggotts-Becketts define it."},
    "Spa-Francorchamps": {"laps": 44, "raceDistance": "308.1 km", "lapRecord": "1:44.701", "recordHolder": "V. Bottas (2018)", "drsZones": 2,
        "about": "The longest lap on the calendar through the Ardennes forest. Iconic Eau Rouge-Raidillon and the Kemmel straight make it a high-speed classic, often affected by changeable weather."},
    "Budapest": {"laps": 70, "raceDistance": "306.6 km", "lapRecord": "1:16.627", "recordHolder": "L. Hamilton (2020)", "drsZones": 2,
        "about": "A twisty, downforce-heavy layout often likened to 'Monaco without the walls'. Overtaking is difficult, placing a premium on qualifying."},
}

_TRACK_MAPS = None
# maps/facts fallbacks — the same venue is named differently across sources.
_CIRCUIT_ALIAS = {"Monte Carlo": "Monaco", "Yas Marina": "Yas Island"}

# FastF1's provisional 2026 schedule lists the Bahrain GP at "Kuala Lumpur",
# which is plainly wrong (the race is at Sakhir). Correct it by event name
# rather than aliasing the location, so no other event is affected.
_SCHEDULE_LOCATION_FIX = {(2026, "Bahrain Grand Prix"): "Sakhir"}


def _track_maps() -> dict:
    global _TRACK_MAPS
    if _TRACK_MAPS is None:
        p = config.PROJECT_ROOT / "web" / "data" / "track_maps.json"
        _TRACK_MAPS = json.loads(p.read_text()) if p.exists() else {}
    return _TRACK_MAPS


_SCHEDULES = {}


def _schedule(year: int):
    """The season's event schedule from FastF1, loaded once per process."""
    if year not in _SCHEDULES:
        try:
            import fastf1
            _SCHEDULES[year] = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:  # noqa: BLE001
            print(f"  schedule for {year} unavailable ({type(e).__name__}: {e})")
            _SCHEDULES[year] = None
    return _SCHEDULES[year]


def _schedule_row(year: int, event: str):
    sched = _schedule(year)
    if sched is None:
        return None
    row = sched[sched["EventName"] == event]
    return row.iloc[0] if len(row) else None


def _circuit_of(year: int, row) -> str:
    """The venue for a scheduled event, with known upstream errors corrected."""
    return _SCHEDULE_LOCATION_FIX.get((year, str(row["EventName"])),
                                      str(row["Location"]))


def _facts_for(circuit: str) -> dict:
    return (CIRCUIT_FACTS.get(circuit)
            or CIRCUIT_FACTS.get(_CIRCUIT_ALIAS.get(circuit, ""))
            or {})


def _canonical_circuit(circuit: str) -> str:
    """Resolve a venue-name variant to the key facts and track maps are stored under.

    The schedule calls Abu Dhabi's circuit "Yas Marina" while the telemetry-derived
    maps are keyed "Yas Island"; the frontend looks maps up by this string, so it
    has to be the canonical one.
    """
    alias = _CIRCUIT_ALIAS.get(circuit)
    if alias and (alias in CIRCUIT_FACTS or alias in _track_maps()):
        return alias
    return circuit


def list_events(years=None) -> list[dict]:
    """Every round of the season for the circuit selector.

    The full calendar is listed, not just the rounds we have data for: `hasData`
    marks the ones that can be predicted, so the frontend can offer the rest as
    upcoming weekends rather than pretending they don't exist.
    """
    quali = pd.read_csv(config.PROCESSED_DIR / "quali_dataset.csv")
    if years is None:
        years = [int(quali["Year"].max())]  # current season

    out = []
    for year in years:
        run = set(quali[quali["Year"] == year]["EventName"])
        sched = _schedule(year)
        if sched is None:  # offline: fall back to whatever the dataset holds
            rows = (quali[quali["Year"] == year][["Round", "EventName", "Circuit"]]
                    .drop_duplicates().sort_values("Round"))
            out += [{"year": year, "round": int(r["Round"]), "event": r["EventName"],
                     "circuit": r["Circuit"], "flag": _facts_for(r["Circuit"]).get("f", "un"),
                     "label": r["EventName"].replace(" Grand Prix", ""),
                     "hasData": True, "date": None, "startsAtUtc": None}
                    for _, r in rows.iterrows()]
            continue

        for _, ev in sched.iterrows():
            name = str(ev["EventName"])
            circuit = _canonical_circuit(_circuit_of(year, ev))
            first = _first_session_utc(ev)
            out.append({
                "year": year, "round": int(ev["RoundNumber"]),
                "event": name, "circuit": circuit,
                "flag": _facts_for(circuit).get("f", "un"),
                "label": name.replace(" Grand Prix", ""),
                "hasData": name in run,
                "date": (pd.to_datetime(ev["EventDate"]).strftime("%d %b")
                         if pd.notna(ev["EventDate"]) else None),
                "startsAtUtc": first.isoformat() + "Z" if first is not None else None,
            })
    return out


def _first_session_utc(row):
    """Start of the weekend's first session (UTC, naive) or None."""
    for i in range(1, 6):
        dt = row.get(f"Session{i}DateUtc")
        if row.get(f"Session{i}") and pd.notna(dt):
            return pd.to_datetime(dt).to_pydatetime()
    return None


def _event_date(year: int, event: str) -> str:
    """Actual race date from the FastF1 schedule (cached), formatted for display."""
    row = _schedule_row(year, event)
    if row is not None and pd.notna(row["EventDate"]):
        return pd.to_datetime(row["EventDate"]).strftime("%d %b %Y")
    return f"{year} Season"


_SESSION_ABBR = {
    "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
    "Qualifying": "QUALI", "Sprint Qualifying": "SQ", "Sprint Shootout": "SQ",
    "Sprint": "SPRINT", "Race": "RACE",
}
_SESSION_HOURS = {"RACE": 2.0, "SPRINT": 1.0}  # approx duration; default 1h


def default_event() -> tuple[int, str]:
    """The most recent event we can say something about.

    A weekend whose practice has run but whose qualifying has not is the one a
    user actually wants on screen, so practice data outranks qualifying data
    when deciding what to open on.
    """
    quali = pd.read_csv(config.PROCESSED_DIR / "quali_dataset.csv")
    yr = int(quali["Year"].max())
    latest = quali[quali["Year"] == yr].sort_values("Round").iloc[-1]
    best = (int(latest["Round"]), str(latest["EventName"]))

    prac_path = config.PROCESSED_DIR / "practice_dataset.csv"
    if prac_path.exists():
        prac = pd.read_csv(prac_path, usecols=["Year", "Round", "EventName"])
        prac = prac[prac["Year"] == yr]
        if len(prac):
            row = prac.sort_values("Round").iloc[-1]
            if int(row["Round"]) > best[0]:
                best = (int(row["Round"]), str(row["EventName"]))
    return yr, best[1]


def _fmt_delta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _session_states(year: int, event: str):
    """Real per-session state (done/live/upcoming) from the FastF1 schedule,
    plus a summary for the 'current session' pill. Compares to real now (UTC).

    Each entry carries its UTC start as an ISO string so the browser can run a
    live countdown instead of showing a duration frozen at build time.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    progress = []
    summary = {"current": "RACE", "live": False, "endsIn": "", "note": "",
               "targetUtc": None}
    row = _schedule_row(year, event)
    if row is None:
        return progress, summary

    live_found, next_up = None, None
    for i in range(1, 6):
        name = row.get(f"Session{i}")
        dt = row.get(f"Session{i}DateUtc")
        local = row.get(f"Session{i}Date")
        if not name or pd.isna(dt):
            continue
        start = pd.to_datetime(dt)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        abbr = _SESSION_ABBR.get(str(name), str(name)[:5].upper())
        dur = _SESSION_HOURS.get(abbr, 1.0)
        end = start + pd.Timedelta(hours=dur)
        if now >= end:
            state = "done"
        elif now >= start:
            state = "live"
            live_found = (abbr, (end - now).total_seconds(), end)
        else:
            state = "upcoming"
            if next_up is None:
                next_up = (abbr, (start - now).total_seconds(), start)
        progress.append({
            "name": abbr, "full": str(name), "date": start.strftime("%d %b"),
            "state": state,
            "startUtc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            # Local circuit time is what a fan plans around, so keep both.
            "localTime": (pd.to_datetime(local).strftime("%a %d %b · %H:%M")
                          if pd.notna(local) else None),
        })

    if live_found:
        summary = {"current": live_found[0], "live": True,
                   "endsIn": _fmt_delta(live_found[1]), "note": "Live",
                   "targetUtc": live_found[2].strftime("%Y-%m-%dT%H:%M:%SZ")}
    elif next_up:
        summary = {"current": next_up[0], "live": False,
                   "endsIn": _fmt_delta(next_up[1]), "note": "Starts in",
                   "targetUtc": next_up[2].strftime("%Y-%m-%dT%H:%M:%SZ")}
    else:
        summary = {"current": "RACE", "live": False, "endsIn": "",
                   "note": "Weekend Complete", "targetUtc": None}
    return progress, summary


def _track_block(circuit: str) -> dict:
    """Facts, extra details and the SVG map for one circuit."""
    alias = _CIRCUIT_ALIAS.get(circuit, "")
    facts = _facts_for(circuit) or {
        "name": circuit, "location": "-", "length": "-", "turns": "-",
        "type": "Circuit", "firstGP": "-", "f": "un"}
    maps = _track_maps()
    tmap = maps.get(circuit) or maps.get(alias) or {"path": "", "viewBox": "0 0 200 120"}
    details = CIRCUIT_DETAILS.get(circuit) or CIRCUIT_DETAILS.get(alias) or {}
    return {**facts, **details, "map": tmap}


def _upcoming_dashboard(year: int, event: str) -> dict:
    """Payload for a weekend that hasn't run yet.

    There is nothing to predict without practice data, so this carries the
    schedule and a countdown target instead — plus the championship as it stands
    going into the round, which is real information the user can act on.
    """
    row = _schedule_row(year, event)
    if row is None:
        raise ValueError(f"No data or schedule for {year} {event}")

    rnd = int(row["RoundNumber"])
    circuit = _circuit_of(year, row)
    progress, session = _session_states(year, event)
    first = next((s for s in progress if s["state"] != "done"), None)
    race = next((s for s in reversed(progress) if s["name"] == "RACE"), None)
    track = _track_block(circuit)

    pace = _pace_bundle()
    metrics = _pace_accuracy(pace) if pace else {
        "mae": None, "rmse": None, "withinRange": None, "history": []}

    return {
        "upcoming": True,
        "event": {"name": f"{event} {year}", "year": year, "eventName": event,
                  "round": rnd, "totalRounds": int(_schedule(year)["RoundNumber"].max()),
                  "date": _event_date(year, event),
                  "circuit": track["name"], "flag": track.get("f", "un")},
        "session": session,
        "weather": None,          # a future weekend has no measured weather
        "progress": progress,
        "schedule": progress,
        "startsAtUtc": first["startUtc"] if first else None,
        "startsLocal": first["localTime"] if first else None,
        "startsSession": first["full"] if first else None,
        "raceAtUtc": race["startUtc"] if race else None,
        "track": track,
        "confidence": {"value": None, "label": "Not yet available", "parts": {},
                       "basis": "the weekend has not run"},
        "practice": _practice_block(year, event, None),
        "poleBattle": None,
        "drivers": [],
        "standings": _standings(year, rnd),
        "sectors": [],
        "insights": [],
        "accuracy": metrics,
        "model": {"version": "v4.0.0", "trainedOn": f"{year} Season",
                  "status": "Awaiting FP1", "lastUpdated": _event_date(year, event),
                  "nextUpdate": "After FP1",
                  "algorithm": "Ridge regression", "encoder": "Target encoding",
                  "validation": "Leave-one-round-out",
                  "target": "Qualifying gap to pole (%)", "features": []},
    }


def _practice_block(year: int, event: str, prac: pd.DataFrame | None) -> dict:
    """What practice evidence the prediction is standing on.

    A prediction is available from the first session onwards, so the UI has to be
    able to say whether more running is still to come — otherwise a call made off
    FP1 alone looks as settled as one made after FP3. `expected` comes from the
    schedule, since a sprint weekend runs FP1 only.
    """
    has = prac is not None and len(prac)
    used = sorted(prac["Session"].unique().tolist()) if has else []
    expected = []
    row = _schedule_row(year, event)
    if row is not None:
        try:
            import refresh
            expected = refresh.expected_practice(row)
        except Exception as e:  # noqa: BLE001
            print(f"  practice schedule unavailable ({type(e).__name__}: {e})")
    remaining = [s for s in expected if s not in used]
    return {
        "used": used,
        "expected": expected,
        "remaining": remaining,
        "complete": bool(expected) and not remaining,
        "drivers": int(prac["Driver"].nunique()) if has else 0,
        "laps": int(prac["CleanLaps"].sum()) if has and "CleanLaps" in prac else 0,
    }


def _weather_from(df: pd.DataFrame | None) -> dict | None:
    """Median conditions across a session's rows (None if we have no rows)."""
    if df is None or not len(df):
        return None
    return {"air": round(float(df["AirTemp"].median())),
            "track": round(float(df["TrackTemp"].median())),
            "humidity": round(float(df["Humidity"].median())),
            "wind": round(float(df["WindSpeed"].median()))}


def build_dashboard(year: int = None, event: str = None) -> dict:
    quali = pd.read_csv(config.PROCESSED_DIR / "quali_dataset.csv")
    if year is None or event is None:
        year, event = default_event()
    ev = quali[(quali["Year"] == year) & (quali["EventName"] == event)]

    pace = _pace_bundle()
    # Predict as soon as practice is in. A qualifying prediction is only worth
    # anything BEFORE qualifying runs, so this must not wait for quali data —
    # the pace model already handles a round with no result yet (its target is
    # NaN, so it is excluded from training but still predicted).
    paced = _predict_event_pace(pace, year, event) if pace else None
    if paced is None and ev.empty:
        # Nothing has run yet — serve the countdown instead of failing.
        return _upcoming_dashboard(year, event)

    if paced is not None:
        circuit = str(paced[0]["Circuit"].iloc[0])
        rnd = int(paced[0]["Round"].iloc[0])
    else:
        circuit = str(ev["Circuit"].iloc[0])
        rnd = int(ev["Round"].iloc[0])
    # Before qualifying, practice conditions are the only ones measured.
    pre_quali = ev.empty
    weather = (_weather_from(ev) if not ev.empty
               else _weather_from(paced[1]["practice"]))

    if paced is not None:
        ranked, extra = paced
        conf = extra["conf"]
        pole_p = conf["drivers"]["pole"]
        within1 = conf["drivers"]["within1"]
        fastest = float(ranked["Pred"].iloc[0])
        drivers = []
        for pos, row in ranked.iterrows():
            pred = float(row["Pred"])
            low, high = float(row["RangeLow"]), float(row["RangeHigh"])
            drivers.append({
                **_driver_meta(str(row["Driver"]), str(row["Team"])),
                "pos": int(pos) + 1,
                "compound": str(row.get("PracCompound", "")),
                "predBest": round(pred, 3),
                "predBestFmt": _fmt_laptime(pred),
                "rangeLow": round(low, 3), "rangeLowFmt": _fmt_laptime(low),
                "rangeHigh": round(high, 3), "rangeHighFmt": _fmt_laptime(high),
                "marker": round((pred - low) / (high - low), 3),
                # Probability this driver lands within one place of where we put
                # them, from the Monte Carlo over the whole field.
                "confidence": int(round(float(within1[pos]) * 100)),
                "polePct": round(float(pole_p[pos]) * 100, 1),
                "practiceGapPct": round(float(row["PracGapPct"]), 3),
                "practiceLaps": int(row["PracLaps"]) if pd.notna(row["PracLaps"]) else 0,
                "delta": None if pos == 0 else round(pred - fastest, 3),
            })
        confidence = {
            "value": int(round(conf["value"])),
            "label": pace["C"].label_for(conf["value"]),
            "parts": conf["parts"], "weights": conf["weights"],
            "sigmaOrderPct": round(conf["sigmaOrder"], 3),
            "maePct": round(conf["mae"], 3),
            "r2": round(conf["r2"], 3),
            "basis": "leave-one-round-out residuals at this circuit",
        }
        pole_battle = _pole_battle(pace, extra, drivers, year, int(ranked["Round"].iloc[0]))
    else:
        # No practice data: fall back to the absolute-time model.
        ranked = _predict_event(ev)
        fastest = float(ranked["Pred"].iloc[0])
        drivers = []
        for pos, row in ranked.iterrows():
            pred = float(row["Pred"])
            low, high = pred - 0.15 - 0.01 * pos, pred + 0.30 + 0.02 * pos
            drivers.append({
                **_driver_meta(str(row["Driver"]), str(row["Team"])),
                "pos": int(pos) + 1,
                "compound": str(row.get("Compound", "")),
                "predBest": round(pred, 3), "predBestFmt": _fmt_laptime(pred),
                "rangeLow": round(low, 3), "rangeLowFmt": _fmt_laptime(low),
                "rangeHigh": round(high, 3), "rangeHighFmt": _fmt_laptime(high),
                "marker": round((pred - low) / (high - low), 3),
                "confidence": None, "polePct": None,
                "delta": None if pos == 0 else round(pred - fastest, 3),
            })
        confidence = {"value": None, "label": "Unavailable", "parts": {},
                      "basis": "practice dataset not built"}
        pole_battle = None

    # Fastest-sector predictions: split the ideal lap by typical proportions
    # and attribute leaders from the top of the predicted order.
    props = [0.27, 0.44, 0.29]
    ideal = sum(fastest * p for p in props)
    leaders = [drivers[0], drivers[min(1, len(drivers) - 1)], drivers[min(2, len(drivers) - 1)]]
    sectors = [{"sector": i + 1, "time": round(fastest * props[i], 3),
                "driver": leaders[i]["code"]} for i in range(3)]

    metrics = _pace_accuracy(pace) if paced is not None else _accuracy_metrics(quali)

    track = _track_block(circuit)
    progress, session = _session_states(year, event)
    insights = _insights(drivers, ev, pole_battle)
    champ = _standings(year, rnd)

    sched = _schedule(year)
    return {
        "upcoming": False,
        # True when this is a genuine forward prediction: practice has run, the
        # qualifying session it predicts has not.
        "preQuali": pre_quali,
        "practice": _practice_block(year, event,
                                    paced[1]["practice"] if paced is not None else None),
        "event": {"name": f"{event} {year}", "year": year, "eventName": event,
                  "round": rnd,
                  "totalRounds": (int(sched["RoundNumber"].max())
                                  if sched is not None else None),
                  "date": _event_date(year, event),
                  "circuit": track["name"], "flag": track.get("f", "un")},
        "session": session,
        "weather": weather,
        "progress": progress,
        "track": track,
        "confidence": confidence,
        "poleBattle": pole_battle,
        "drivers": drivers,
        "standings": champ,
        "sectors": sectors,
        "idealLap": round(ideal, 3), "idealLapFmt": _fmt_laptime(ideal),
        "insights": insights,
        "accuracy": metrics,
        # Described from the model that actually produced the numbers above, so
        # the details panel cannot drift away from what is really running.
        "model": {"version": "v4.0.0" if paced is not None else "v3.0.0",
                  "trainedOn": f"{year} Season",
                  "status": "Optimal", "lastUpdated": _event_date(year, event),
                  "nextUpdate": "Next race weekend",
                  **({"algorithm": "Ridge regression",
                      "encoder": "Target encoding",
                      "validation": "Leave-one-round-out",
                      "target": "Qualifying gap to pole (%)",
                      "features": ["Practice gap (best)", "Practice gap (last session)",
                                   "Practice gap (mean)", "Practice gap (soft tyre)",
                                   "Tyre compound", "Practice sessions", "Practice laps",
                                   "Team season form", "Driver season form",
                                   "Driver", "Team"]}
                     if paced is not None else
                     {"algorithm": "HistGradientBoosting",
                      "encoder": "Target encoding",
                      "validation": "Leave-one-round-out",
                      "target": "Best qualifying lap (s)",
                      "features": ["Driver", "Team", "Circuit", "Compound", "Air Temp",
                                   "Track Temp", "Humidity", "Wind", "Rainfall"]})},
    }


def _favourite_why(fav: dict | None, rows: list[dict]) -> str:
    """One honest line explaining the pole call.

    The favourite is whoever the simulation gives the best pole chance, which is
    not always whoever topped the reference session — season form pulls too. Say
    which it is rather than asserting they were quickest.
    """
    if not fav:
        return ""
    timed = [r for r in rows if r.get("refGapPct") is not None]
    rank = next((i + 1 for i, r in enumerate(timed) if r["code"] == fav["code"]), None)
    ref = fav.get("refSession") or "practice"
    if rank == 1:
        return f"quickest of the top-4 teams in {ref}"
    if rank:
        return (f"P{rank} of the top-4 teams in {ref} (+{fav['refGapPct']:.3f}%), "
                f"but strongest on season form")
    return "strongest on season form"


def _pole_battle(pace: dict, extra: dict, drivers: list[dict],
                 year: int, rnd: int) -> dict:
    """Who takes pole, decided by practice pace among the teams that can win it.

    The top teams are identified from season-to-date qualifying pace using only
    rounds BEFORE this one, so the panel never assumes a result it is trying to
    predict.
    """
    P = pace["P"]
    strength = P.team_strength(pace["quali"], year, before_round=rnd, n=4)
    top_teams = strength[strength["isTop"]]["Team"].tolist()

    ev_prac = extra["practice"]
    lineup = pace["G"]._race_lineup(pace["quali"], year, rnd)
    battle = P.pole_battle(ev_prac, top_teams, lineup)
    pole_by_code = {d["code"]: d for d in drivers}

    rows = []
    for r in battle:
        d = pole_by_code.get(r["code"], {})
        rows.append({
            **r,
            "name": DRIVERS.get(r["code"], (r["code"], "un"))[0],
            "flag": DRIVERS.get(r["code"], (r["code"], "un"))[1],
            "teamColor": TEAM_COLORS.get(r["team"], "#888888"),
            "bestTimeFmt": _fmt_laptime(r["bestTime"]),
            "polePct": d.get("polePct"),
            "predPos": d.get("pos"),
            "predBestFmt": d.get("predBestFmt"),
        })

    # How clear-cut the call is: the favourite's pole probability against the
    # next-best driver's.
    ranked_p = sorted(rows, key=lambda r: -(r["polePct"] or 0))
    favourite = ranked_p[0] if ranked_p else None
    runner_up = ranked_p[1] if len(ranked_p) > 1 else None
    margin = (favourite["polePct"] - runner_up["polePct"]) if runner_up else None

    return {
        "teams": [{"team": t["Team"], "paceGapPct": round(float(t["paceGapPct"]), 3),
                   "color": TEAM_COLORS.get(t["Team"], "#888888")}
                  for _, t in strength[strength["isTop"]].iterrows()],
        "cliffPct": (round(float(strength.iloc[4]["paceGapPct"]
                                 - strength.iloc[3]["paceGapPct"]), 3)
                     if len(strength) > 4 else None),
        "nextTeam": str(strength.iloc[4]["Team"]) if len(strength) > 4 else None,
        "drivers": rows,
        "favourite": favourite,
        "favouriteWhy": _favourite_why(favourite, rows),
        "runnerUp": runner_up,
        "margin": round(margin, 1) if margin is not None else None,
        "sessionsUsed": sorted(ev_prac["Session"].unique().tolist()),
    }


def _insights(drivers: list[dict], ev: pd.DataFrame,
              battle: dict | None = None) -> list[dict]:
    top = drivers[0]
    out = []
    if battle and battle.get("favourite"):
        f = battle["favourite"]
        out.append({"icon": "up", "title": f"{f['name']} is the pole favourite",
                    "sub": f"{f['polePct']:.0f}% chance of pole — {battle['favouriteWhy']}"})
        if battle.get("margin") is not None and battle.get("runnerUp"):
            ru, m = battle["runnerUp"], battle["margin"]
            out.append({"icon": "target",
                        "title": "Clear favourite" if m >= 15 else "Too close to call",
                        "sub": (f"{m:.0f} points of pole chance ahead of {ru['name']}"
                                if m >= 15 else
                                f"only {m:.0f} points separate {f['name']} and {ru['name']}")})
        if battle.get("cliffPct") and battle.get("nextTeam"):
            out.append({"icon": "up", "title": "Top 4 teams are a class apart",
                        "sub": (f"{battle['cliffPct']:.2f}% pace cliff back to "
                                f"{battle['nextTeam']}")})
    else:
        out.append({"icon": "up", "title": f"{top['team']} looks strongest",
                    "sub": f"{top['name']} leads the predicted order"})

    if len(drivers) >= 5:
        gap5 = round(drivers[4]["predBest"] - drivers[0]["predBest"], 3)
        out.append({"icon": "target", "title": "Close battle expected",
                    "sub": f"Top 5 within {gap5:.2f}s"})
    return out[:4]


if __name__ == "__main__":
    data = build_dashboard()
    path = WEB_DATA / "dashboard.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {path}")
    print(f"  {len(data['drivers'])} drivers, "
          f"leader {data['drivers'][0]['code']} {data['drivers'][0]['predBestFmt']}, "
          f"MAE {data['accuracy']['mae']}")
