"""Serve the dashboard: static files from web/ + a live /api/dashboard endpoint.

No web framework needed (stdlib http.server), so it runs anywhere.

    python src/webapp.py            # http://localhost:8000
    python src/webapp.py --port 8080 --year 2024 --event "Belgian Grand Prix"
    python src/webapp.py --refresh-min 0     # disable the auto-refresh thread

While a race weekend is on, a background thread pulls each session's data as
soon as it finishes, so a qualifying prediction appears once practice is over
without anyone running a command.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import config
from dashboard_data import build_dashboard, list_events

WEB_DIR = config.PROJECT_ROOT / "web"


class Handler(SimpleHTTPRequestHandler):
    year = 2024
    event = "Belgian Grand Prix"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB_DIR), **kw)

    def end_headers(self):
        # Applies to the static files too: without this the browser caches
        # app.js/styles.css heuristically and edits don't show up on refresh.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/refresh":
            # Manual trigger behind the Refresh button, for when you don't want
            # to wait for the next scheduled check.
            try:
                self._json(_refresh_once())
            except Exception as e:  # noqa: BLE001
                self.send_error(500, f"{type(e).__name__}: {e}")
            return
        if parsed.path == "/api/events":
            try:
                self._json(list_events())
            except Exception as e:  # noqa: BLE001
                self.send_error(500, f"{type(e).__name__}: {e}")
            return
        if parsed.path == "/api/dashboard":
            qs = parse_qs(parsed.query)
            year = int(qs["year"][0]) if "year" in qs else None
            event = qs["event"][0] if "event" in qs else None
            try:
                self._json(build_dashboard(year, event))
            except Exception as e:  # noqa: BLE001
                self.send_error(500, f"{type(e).__name__}: {e}")
            return
        return super().do_GET()

    def log_message(self, *a):  # keep the console quiet
        pass


_REFRESH_LOCK = threading.Lock()


def _refresh_once(year: int = None) -> dict:
    """Pull any session that has finished since the last check.

    Serialised: the background thread and the Refresh button must never run two
    FastF1 fetches into the same CSV at once.
    """
    import refresh
    with _REFRESH_LOCK:
        return refresh.auto_refresh(year, verbose=False)


def _refresh_loop(year: int, interval_s: int):
    """Keep the datasets current while the server runs.

    dashboard_data keys its model cache on the CSV mtimes, so the next request
    after a successful pull rebuilds automatically — no restart needed.
    """
    while True:
        try:
            r = _refresh_once(year)
            if r["changed"]:
                print(f"[auto-refresh] {r['year']}: {r['summary']}")
        except Exception as e:  # noqa: BLE001
            print(f"[auto-refresh] failed: {type(e).__name__}: {e}")
        time.sleep(interval_s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--event", default="Belgian Grand Prix")
    p.add_argument("--refresh-min", type=int, default=10,
                   help="minutes between data checks; 0 disables auto-refresh")
    args = p.parse_args()

    Handler.year, Handler.event = args.year, args.event
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"F1 Lap Predictor dashboard -> http://localhost:{args.port}")
    print(f"  serving {WEB_DIR}")
    print(f"  API: /api/dashboard  ({args.year} {args.event})")
    if args.refresh_min > 0:
        import refresh
        threading.Thread(target=_refresh_loop,
                         args=(refresh.current_season(), args.refresh_min * 60),
                         daemon=True).start()
        print(f"  auto-refresh: every {args.refresh_min} min")
    else:
        print("  auto-refresh: off")
    print("  Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
