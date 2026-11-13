"""The anonymized video of a live camera, as it is made: the stick figures on the empty store, drawn from the skeleton
stream only (skeleton.draw), as MJPEG. Nothing here ever touches a camera picture: it reads the stream's file as the
anonymizer writes it, so what it can show is exactly what leaves the anonymizer. Used by the anonymizer's own page
(server.py) and by the link (viewer.py)."""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import skeleton

FPS = 10  # pictures per second sent at most (the stream has ~25: the link need not carry them all)
WIDTH = 960  # sent this wide at most over the link (viewer.py): little to carry over the internet
QUALITY = 70
LOCAL_WIDTH, LOCAL_QUALITY = 1920, 90  # on this computer's own page (server.py) nothing is carried: sharp
                                       # (Sameh 2026-11-13: "varför är footagen blurrig")
IDLE_END_S = 30.0  # a stream not written to for this long has stopped (the anonymizer died without its end line)
BOUNDARY = "frame"


def _has_end(path: Path) -> bool:
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - 4096))
        return b'"end":true' in f.read()


def is_live(folder: Path) -> bool:
    """A live camera's stream being written now: its header says live, it has no end line, and it grew lately."""
    p = Path(folder) / skeleton.FILE
    try:
        if time.time() - p.stat().st_mtime > IDLE_END_S or not skeleton.header(folder).get("live"):
            return False
        return not _has_end(p)
    except (OSError, ValueError):
        return False


def live_names(root: Path) -> list[str]:
    return sorted(p.parent.name for p in Path(root).glob("*/" + skeleton.FILE) if is_live(p.parent)) if Path(root).is_dir() else []


def picture(rows, size, bg: np.ndarray | None, width: int = WIDTH, quality: int = QUALITY) -> bytes:
    """One frame's people on the empty store (or on grey until it is made), as a JPEG at most `width` wide."""
    W, H = size
    img = bg if bg is not None and bg.shape[1] == W and bg.shape[0] == H else np.full((H, W, 3), 60, np.uint8)
    out = skeleton.draw(img, rows)
    if W > width:
        out = cv2.resize(out, (width, round(H * width / W)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def mjpeg(folder: Path, stop=lambda: False, fps: float = FPS, poll_s: float = 0.03, width: int = WIDTH,
          quality: int = QUALITY):
    """multipart/x-mixed-replace parts of the stream from now on, at most `fps` a second, until it ends, stops
    growing for IDLE_END_S, or stop() is true."""
    folder = Path(folder)
    path = folder / skeleton.FILE
    with open(path) as f:
        head = json.loads(f.readline())
        size = tuple(head["size"])
        f.seek(0, 2)  # from now on: the link shows the camera as it is, not its past
        bg, bg_mtime, sent, idle, latest = None, None, 0.0, time.monotonic(), None
        while not stop():
            pos = f.tell()
            line = f.readline()
            if not line.endswith("\n"):  # nothing new, or half a line: wait for the rest
                f.seek(pos)
                if time.monotonic() - idle > IDLE_END_S:
                    return
                time.sleep(poll_s)
                continue
            idle = time.monotonic()
            rec = json.loads(line)
            if rec.get("end"):
                return
            latest = rec["p"]
            if idle - sent < 1.0 / fps:
                continue
            sent = idle
            try:
                m = (folder / skeleton.BACKGROUND).stat().st_mtime
                if m != bg_mtime:
                    bg, bg_mtime = cv2.imread(str(folder / skeleton.BACKGROUND)), m
            except OSError:
                pass
            jpg = picture(latest, size, bg, width, quality)
            yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n").encode() + jpg + b"\r\n"
