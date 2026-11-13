"""Sends a skeleton stream on to the analysis on another computer, as it is written (J3 (c), ARCHITECTURE §13):

    .venv/bin/python -m anonymizer.send <stream folder> https://analysis.example [--name Kam1.live]
    .venv/bin/python -m anonymizer.main rtsp://… <out folder> --send https://analysis.example

The anonymizer always connects outwards, so the computer in the store needs no open port. The stream file on this
computer is the queue: the sender keeps how far the other side has it (the answer to each batch) and sends what
comes after, about once a second, gzip, whole lines only. When the network or the analysis is gone, the file goes
on growing here and is sent in order when it is back (the waits between tries double up to RETRY_MAX_S). The empty
store is sent whenever it is made anew. Only the stream's own lines and the empty store leave: nothing of a picture.

The key is the box's own (inferero/ingest.py, scripts/ingest_key.py on the analysis' computer): from
HYLLORA_INGEST_KEY, or the file data/anonymizer/ingest_key (owner-only). It is never written to a log or a stream.
"""

import argparse
import gzip
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

from . import paths, skeleton

KEY_FILE = paths.OWN / "ingest_key"
EVERY_S = 1.0  # a batch about once a second: not a request per frame
BATCH_MAX = 1024 * 1024  # bytes of stream per batch (a backlog goes in several)
RETRY_MAX_S = 30.0
TIMEOUT_S = 20.0
DRAIN_S = 30.0  # after stop: how long to go on sending what is left before giving up


def key() -> str:
    k = os.environ.get("HYLLORA_INGEST_KEY", "").strip()
    if not k:
        try:
            k = KEY_FILE.read_text().strip()
        except OSError:
            k = ""
    if "." not in k:
        raise ValueError(f"Ingen nyckel: sätt HYLLORA_INGEST_KEY eller lägg den i {KEY_FILE}")
    return k


def _post(url: str, data: bytes, token: str, method: str = "POST", headers: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": f"Bearer {token}", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


class Sender(threading.Thread):
    """Sends `folder`'s stream to `base` (the analysis' address) under `name`, until its end line is through, or
    stop() and what is left has gone (DRAIN_S at most). `status()` for the page and the tests."""

    def __init__(self, folder: Path, base: str, token: str, name: str | None = None, every_s: float = EVERY_S,
                 post=_post, drain_s: float = DRAIN_S):
        super().__init__(daemon=True, name="send")
        self.folder, self.path = Path(folder), Path(folder) / skeleton.FILE
        self.name = name or self.folder.name
        self.base, self.token, self.every_s, self.post, self.drain_s = base.rstrip("/"), token, every_s, post, drain_s
        self.offset = 0  # how far the other side has the stream (bytes): 0 starts it over there
        self.ino = None
        self.bg_sent = None  # the background file's time when last sent
        self.done = False  # the end line is through
        self.error: str | None = None
        self.last_ok: float | None = None
        self._stop = threading.Event()
        self._stop_at: float | None = None

    def stop(self) -> None:
        self._stop_at = time.monotonic()
        self._stop.set()

    def status(self) -> dict:
        try:
            size = self.path.stat().st_size
        except OSError:
            size = 0
        return {"to": self.base, "name": self.name, "sent": self.offset, "behind": max(0, size - self.offset),
                "done": self.done, "error": self.error,
                "since_ok_s": round(time.monotonic() - self.last_ok, 1) if self.last_ok else None}

    def _url(self, what: str) -> str:
        return f"{self.base}/api/ingest/{quote(self.name, safe='')}/{what}"

    def run(self) -> None:
        tries = 0
        while not self.done:
            if self._stop.is_set() and time.monotonic() - self._stop_at > self.drain_s:
                return
            try:
                more = self._once()
                self.error, tries = None, 0
                self.last_ok = time.monotonic()
            except (OSError, ValueError) as e:  # the network, the other side, or what it said
                self.error = f"{type(e).__name__}: {e}"
                tries += 1
                if self._stop.wait(min(RETRY_MAX_S, 2 ** (tries - 1))) and time.monotonic() - self._stop_at > self.drain_s:
                    return
                continue
            if self.done:
                return
            if not more:
                if self._stop.is_set() and self._caught_up():
                    return
                self._stop.wait(self.every_s)

    def _caught_up(self) -> bool:
        try:
            return self.path.stat().st_size <= self.offset
        except OSError:
            return True

    def _once(self) -> bool:
        """One batch (and the empty store when it is new). True when there was more than one batch to send."""
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return False
        if self.ino is not None and (st.st_ino != self.ino or st.st_size < self.offset):
            self.offset = 0  # the anonymizer started the file over: so does the other side
            self.bg_sent = None
        self.ino = st.st_ino
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read(BATCH_MAX)
        chunk = chunk[:chunk.rfind(b"\n") + 1]  # whole lines only: the rest is still being written
        more = False
        if chunk:
            code, ans = self.post(self._url(f"lines?offset={self.offset}"), gzip.compress(chunk), self.token,
                                  headers={"Content-Encoding": "gzip", "Content-Type": "application/x-ndjson"})
            if code == 409 and isinstance(ans.get("offset"), int):  # the other side has it to here: go on from there
                self.offset = ans["offset"]
                return True
            if code != 200 or not isinstance(ans.get("offset"), int):
                raise ValueError(f"{code} {ans.get('detail', '')}".strip())
            self.offset = ans["offset"]
            last = chunk.rstrip(b"\n").rsplit(b"\n", 1)[-1]
            if b'"end":true' in last and self.offset >= st.st_size:
                self._background()  # the final empty store goes before the stream is called done
                self.done = True
                return False
            more = len(chunk) >= BATCH_MAX // 2
        if self.offset > 0:
            self._background()
        return more

    def _background(self) -> None:
        bg = self.folder / skeleton.BACKGROUND
        try:
            at = bg.stat().st_mtime
        except OSError:
            return
        if at == self.bg_sent:
            return
        code, ans = self.post(self._url("background"), bg.read_bytes(), self.token, method="PUT",
                              headers={"Content-Type": "image/jpeg"})
        if code != 200:
            raise ValueError(f"bakgrunden: {code} {ans.get('detail', '')}".strip())
        self.bg_sent = at


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="the stream's folder (skeleton.jsonl and background.jpg)")
    ap.add_argument("to", help="the analysis' address, e.g. http://192.168.1.20:8420")
    ap.add_argument("--name", help="the stream's name there (default: the folder's name)")
    a = ap.parse_args()
    s = Sender(Path(a.folder), a.to, key(), a.name)
    s.start()
    try:
        while s.is_alive():
            s.join(10)
            print(json.dumps(s.status(), ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        s.stop()
        s.join()
    print(json.dumps(s.status(), ensure_ascii=False))
    sys.exit(0 if s.done or s.status()["behind"] == 0 else 1)


if __name__ == "__main__":
    main()
