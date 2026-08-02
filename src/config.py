"""Project-wide paths and FastF1 cache configuration.

Import this module before using FastF1 so the on-disk cache is always enabled.
The cache is important: FastF1 downloads a lot of data per session, so caching
turns a ~30s fetch into an instant load on subsequent runs.
"""
from __future__ import annotations

from pathlib import Path

import fastf1

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # existing Ergast JSON (secondary metadata)
CACHE_DIR = DATA_DIR / "cache"      # FastF1 on-disk cache
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned DataFrames we build
MODELS_DIR = PROJECT_ROOT / "models"

for _d in (CACHE_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- FastF1 cache ------------------------------------------------------------
_cache_enabled = False


def enable_cache() -> Path:
    """Turn on the FastF1 on-disk cache (idempotent). Returns the cache path."""
    global _cache_enabled
    if not _cache_enabled:
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        _cache_enabled = True
    return CACHE_DIR


# Enable on import so callers never forget.
enable_cache()
