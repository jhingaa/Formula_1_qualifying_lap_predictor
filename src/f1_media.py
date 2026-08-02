"""Download the official F1 driver portraits and team logos from formula1.com.

Source: the same media.formula1.com assets that back
https://www.formula1.com/en/results/2026/drivers and .../team — current-season,
in-team-kit driver cut-outs and white team marks. Both are transparent WebP and
therefore sit cleanly on the dark dashboard.

Images are saved into web/assets/ and listed in web/data/*.json so the frontend
works offline and never hotlinks F1's CDN on every page load.

Run:  python f1_media.py            # 2026 grid: portraits + logos
      python f1_media.py 2026 --force
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

import config
import fastf1

WEB = config.PROJECT_ROOT / "web"
DRIVER_DIR = WEB / "assets" / "drivers"
TEAM_DIR = WEB / "assets" / "teams"
DRIVER_MANIFEST = WEB / "data" / "driver_photos.json"
TEAM_MANIFEST = WEB / "data" / "team_logos.json"

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# Head-and-shoulders crop of the full-body cut-out. Cloudinary does the framing,
# so we store one small square per driver instead of a 320x920 portrait.
_CROP = "c_thumb,g_face,w_240,h_240,z_0.72"
_DRIVER_CDN = ("https://media.formula1.com/image/upload/{crop}/q_auto/"
               "v1740000001/common/f1/{year}/{team}/{slug}/{year}{team}{slug}right.webp")
# The white team mark, sized to fit rather than fill so wide logos aren't cropped.
_TEAM_CDN = ("https://media.formula1.com/image/upload/c_fit,w_96,h_96/q_auto/"
             "v1740000001/common/f1/{year}/{team}/{year}{team}logowhite.webp")

# The slug lives inside the legacy headshot URL FastF1 already carries, e.g.
# .../drivers/K/ANDANT01_Kimi_Antonelli/andant01.png... -> "andant01". It cannot
# be derived from the display name (Antonelli races as ANDANT01, not KIMANT01).
_SLUG_RE = re.compile(r"/([a-z]{6}\d{2})\.png")


def _team_slug(team: str) -> str:
    """'Red Bull Racing' -> 'redbullracing' (how F1's asset paths are keyed)."""
    return re.sub(r"[^a-z0-9]", "", str(team).lower())


def _grid(year: int, rnd: int = 1):
    """The season's driver list, straight from a race session's results."""
    s = fastf1.get_session(year, rnd, "R")
    s.load(laps=False, telemetry=False, weather=False, messages=False)
    return s.results


def grid_photo_urls(year: int, rnd: int = 1) -> dict[str, dict]:
    """code -> {url, fallback, name, team} for every driver on the grid."""
    out = {}
    for _, r in _grid(year, rnd).iterrows():
        code, head = str(r["Abbreviation"]), str(r.get("HeadshotUrl") or "")
        m = _SLUG_RE.search(head)
        if not m:
            continue
        out[code] = {
            "url": _DRIVER_CDN.format(crop=_CROP, year=year,
                                      team=_team_slug(r["TeamName"]), slug=m.group(1)),
            # Older, white-background portrait — only used if the cut-out 404s.
            "fallback": head.replace("/1col/", "/3col/"),
            "name": str(r["FullName"]),
            "team": str(r["TeamName"]),
        }
    return out


def _download(url: str, min_bytes: int = 2000) -> bytes | None:
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, headers=_UA), timeout=30) as r:
            data = r.read()
        # A CDN miss still returns 200 with F1's fallback image; those come back
        # tiny, so treat an implausibly small file as a miss.
        return data if len(data) > min_bytes else None
    except (urllib.error.URLError, OSError):
        return None


def fetch_driver_photos(year: int, rnd: int = 1, force: bool = False) -> dict[str, str]:
    """Download every driver portrait and write the manifest. Returns code -> path."""
    DRIVER_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for code, info in grid_photo_urls(year, rnd).items():
        for ext, url in (("webp", info["url"]), ("png", info["fallback"])):
            dest = DRIVER_DIR / f"{code}.{ext}"
            rel = f"assets/drivers/{dest.name}"
            if dest.exists() and not force:
                manifest[code] = rel
                break
            blob = _download(url)
            if blob:
                dest.write_bytes(blob)
                manifest[code] = rel
                print(f"  {code}  {len(blob) / 1024:5.1f} kB  {dest.name}")
                break
        else:
            print(f"  {code}  no portrait found")

    DRIVER_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    DRIVER_MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


def fetch_team_logos(year: int, rnd: int = 1, force: bool = False) -> dict[str, str]:
    """Download each constructor's white logo. Returns team name -> path.

    Keyed by FastF1's TeamName (not the URL slug) so the dashboard can look a
    logo up with the same string it already stores against every driver.
    """
    TEAM_DIR.mkdir(parents=True, exist_ok=True)
    teams = sorted({str(t) for t in _grid(year, rnd)["TeamName"]})
    manifest = {}
    for team in teams:
        slug = _team_slug(team)
        dest = TEAM_DIR / f"{slug}.webp"
        rel = f"assets/teams/{dest.name}"
        if dest.exists() and not force:
            manifest[team] = rel
            continue
        # Logos are simple marks — a few hundred bytes is a valid file here.
        blob = _download(_TEAM_CDN.format(year=year, team=slug), min_bytes=200)
        if blob:
            dest.write_bytes(blob)
            manifest[team] = rel
            print(f"  {team:18} {len(blob) / 1024:5.1f} kB  {dest.name}")
        else:
            print(f"  {team:18} no logo found")

    TEAM_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    TEAM_MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    yr = int(args[0]) if args else 2026
    force = "--force" in sys.argv
    print("Driver portraits:")
    drivers = fetch_driver_photos(yr, force=force)
    print("Team logos:")
    logos = fetch_team_logos(yr, force=force)
    print(f"\n{len(drivers)} portraits -> {DRIVER_DIR}")
    print(f"{len(logos)} logos     -> {TEAM_DIR}")
