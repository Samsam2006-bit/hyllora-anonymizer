"""The box's status to the Hyllora hub (hub/boxes.py, 2026-11-17): what the live cameras are doing, so Hyllora's team
sees it in the hub instead of at the store's computer. Sameh: "bara så man ser status och felkoder och inget om det som
sker i själva footagen", "förutsatt att det är gdpr vänligt".

What goes: the program's version, the operating system, and per live camera its name, step, frame number, how long it
has stood still, error and last log lines. Every text goes through scrub() first (no login, no path or query of an
address); the hub cleans it again. Nothing of what the cameras see is sent: no picture, no skeleton, no people, no
counts. The box connects outwards, every EVERY_S, so the store needs no open port.

The hub's address and the box's key are set on the anonymizer's page (PUT /api/hub) and kept in
data/anonymizer/hub.json, readable by this user only; HYLLORA_HUB_URL and HYLLORA_HUB_KEY override them."""

import json
import os
import platform
import re
import threading
import time
import urllib.error
import urllib.request

from . import paths

FILE = paths.DATA / "anonymizer" / "hub.json"
EVERY_S = 15.0
TIMEOUT_S = 10.0
_URL = re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.-]*://)(?:[^\s/@]*@)?([^\s/:?#@]+)(:\d+)?[^\s]*")


def scrub(s) -> str | None:
    """A line as it may leave the store: an address keeps its scheme, host and port only (as hub.boxes.scrub)."""
    return None if s is None else _URL.sub(lambda m: m.group(1) + m.group(2) + (m.group(3) or ""), str(s))[:300]


def settings() -> dict:
    url, key = os.environ.get("HYLLORA_HUB_URL", "").strip(), os.environ.get("HYLLORA_HUB_KEY", "").strip()
    if not (url and key):
        try:
            saved = json.loads(FILE.read_text(encoding="utf-8"))
            url, key = url or saved.get("url", ""), key or saved.get("key", "")
        except (OSError, ValueError):
            pass
    return {"url": url.rstrip("/"), "key": key}


def save(url: str, key: str) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"url": url.rstrip("/"), "key": key}, f)
    os.replace(tmp, FILE)


def version() -> str:
    """The testrepo's commit (packaging/testrepo/run.ps1 writes .version), else "utveckling"."""
    try:
        return (paths.ROOT / ".version").read_text(encoding="utf-8").strip()[:7] or "utveckling"
    except OSError:
        return "utveckling"


def payload(cameras: list[dict], started: float) -> dict:
    """The status as it is sent: only these fields, every text scrubbed."""
    return {"version": version(), "os": f"{platform.system()} {platform.release()}", "uptime_s": round(time.monotonic() - started),
            "cameras": [{"name": c["name"], "running": c["running"], "phase": scrub(c.get("phase")), "frame": c.get("frame"),
                         "still_s": c.get("still_s"), "elapsed_s": c.get("elapsed_s"), "error": scrub(c.get("error")),
                         "gaps": len((c.get("info") or {}).get("gaps") or []),
                         "fps": (c.get("stats") or {}).get("fps"), "model_ms": (c.get("stats") or {}).get("model_ms"),
                         "mode": scrub(_mode(c)),
                         "log": [scrub(x) for x in (c.get("log") or [])[-30:]]} for c in cameras]}


def _mode(c: dict) -> str | None:
    """The camera's settings in a few words (settings.PRESETS), for the hub: which mode works on this computer."""
    o = c.get("opts") or {}
    if not o.get("model"):
        return None
    return " · ".join(x for x in ("n" if "26n" in o["model"] else "s", f"{o.get('imgsz')} px",
                                  f"{o['fps']:g}/s" if o.get("fps") else "alla bilder", "rörelsekontroll" if o.get("motion") else "",
                                  "en process" if c.get("fleet") else "") if x)


class Reporter(threading.Thread):
    """Sends the status every EVERY_S while the hub is set; `last` says how the latest try went (the page shows it)."""

    def __init__(self, cameras, every_s: float = EVERY_S):
        super().__init__(daemon=True, name="hub-status")
        self.cameras, self.every_s = cameras, every_s
        self.started = time.monotonic()
        self.last: dict = {"ok": None, "at": None, "error": None}
        self.wake = threading.Event()

    def send_once(self) -> None:
        cfg = settings()
        if not (cfg["url"] and cfg["key"]):
            self.last = {"ok": None, "at": None, "error": None}
            return
        body = json.dumps(payload(self.cameras(), self.started)).encode()
        req = urllib.request.Request(cfg["url"] + "/api/box/status", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg['key']}"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                self.every_s = float(json.loads(r.read() or b"{}").get("every_s") or self.every_s)
            self.last = {"ok": True, "at": time.time(), "error": None}
        except urllib.error.HTTPError as e:
            why = "fel nyckel" if e.code == 401 else f"portalen svarade {e.code}"
            self.last = {"ok": False, "at": time.time(), "error": why}
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.last = {"ok": False, "at": time.time(), "error": scrub(f"når inte portalen ({getattr(e, 'reason', e)})")}

    def run(self) -> None:
        while True:
            try:
                self.send_once()
            except Exception as e:  # never let the status stop the cameras' page
                self.last = {"ok": False, "at": time.time(), "error": scrub(f"{type(e).__name__}: {e}")}
            self.wake.wait(max(5.0, min(self.every_s, 300.0)))
            self.wake.clear()
