"""The link: the live cameras' anonymized video, for anyone who has the link, through a tunnel to the internet.

    .venv/bin/uvicorn anonymizer.viewer:app --host 127.0.0.1 --port 8431 --no-access-log

A program of its own, on a port of its own, so that a tunnel to it can reach nothing else: not the anonymizer's page
(8430, which shows the cameras' originals) and not the analysis (8420). It serves exactly two things under a secret
key, /v/<key>/ (the live cameras) and /v/<key>/<name>.mjpg (one camera's stick figures on the empty store, view.py,
drawn from the skeleton stream only), and 404 for everything else, a wrong key included. Only cameras running live
now are shown; a finished stream is not. The key is in data/anonymizer/view_token (never committed); delete the file
and restart for a new link. --no-access-log: the key is in every path, and a log would keep it.
"""

import hmac
import os
import secrets
from html import escape
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from . import paths, view

SKELETONS = paths.SKELETONS
TOKEN_FILE = paths.OWN / "view_token"
PORT = 8431
HEADERS = {"Cache-Control": "no-store", "X-Robots-Tag": "noindex, nofollow", "Referrer-Policy": "no-referrer"}

app = FastAPI(title="Hyllora live", docs_url=None, redoc_url=None, openapi_url=None)


def token() -> str:
    """The link's secret key, made the first time (32 random characters, only this user may read the file)."""
    try:
        t = TOKEN_FILE.read_text().strip()
        if len(t) >= 24:
            return t
    except OSError:
        pass
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    t = secrets.token_urlsafe(24)
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(t)
    return t


def _check(key: str) -> None:
    if not hmac.compare_digest(key.encode(), token().encode()):
        raise HTTPException(404)


PAGE = """<!doctype html><html lang="sv"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex">
<title>Hyllora live</title>
<style>
  body {{ margin: 0; background: #111; color: #eee; font: 15px/1.4 system-ui, sans-serif; }}
  header {{ padding: 16px; display: flex; justify-content: space-between; align-items: baseline; gap: 16px; flex-wrap: wrap; }}
  h1 {{ font-size: 18px; margin: 0; }}
  .note {{ color: #999; font-size: 13px; }}
  main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 480px), 1fr)); gap: 16px; padding: 0 16px 16px; }}
  figure {{ margin: 0; background: #000; }}
  figcaption {{ padding: 8px 12px; font-weight: 600; }}
  img {{ display: block; width: 100%; height: auto; }}
</style></head><body>
<header><h1>Hyllora live</h1><span class="note">Anonymiserat: streckgubbar på en bild av den tomma butiken. Ingen kamerabild av en person.</span></header>
<main>{cams}</main>
<script>setTimeout(() => location.reload(), {reload});</script>
</body></html>"""


@app.get("/v/{key}/", response_class=HTMLResponse)
def page(key: str):
    _check(key)
    names = view.live_names(SKELETONS)
    cams = "".join(f'<figure><figcaption>{escape(n.removesuffix(".live"))}</figcaption>'
                   f'<img src="{escape(n)}.mjpg" alt="Kamera {escape(n)}"></figure>' for n in names)
    # with cameras: reload rarely (a reload restarts the pictures); without: look again soon
    return HTMLResponse(PAGE.format(cams=cams or '<p class="note">Ingen kamera sänder just nu.</p>',
                                    reload=600_000 if names else 10_000), headers=HEADERS)


@app.get("/v/{key}/{name}.mjpg")
def stream(key: str, name: str):
    _check(key)
    folder = SKELETONS / name
    if Path(name).name != name or name.startswith(".") or not view.is_live(folder):
        raise HTTPException(404)
    return StreamingResponse(view.mjpeg(folder), media_type=f"multipart/x-mixed-replace; boundary={view.BOUNDARY}",
                             headers=HEADERS)
