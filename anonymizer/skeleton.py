"""The skeleton stream: everything that leaves the anonymizer, as JSON lines in a folder of its own.

    skeleton.jsonl   line 1, the header: {"format": FORMAT, "version": 1, "source" (the video's file name), "size": [W, H], "roi": [x0, y0, x1, y1] | null,
                       "src_fps", "rate" (analysed frames per second), "stride", "frames", "model", "imgsz", "conf",
                       "tracker", "background"}
                     then one line per analysed frame: {"i": frame number (1 = the first), "t": video time in s,
                       "p": [[x0, y0, x1, y1, conf, track id, x, y, c, ... (17 keypoints)], ...]}, full-frame pixels,
                       and "l": {row: body vector} for the people described in that frame (looks.py), base64 of
                       float32, when the header's "look" names the model (None: no vectors at all),
                       and "o": [[x0, y0, x1, y1, conf, class], ...] on the frames where the things the store showed
                       a photo of were looked for (objects.py; the header's "objects" names the classes, null: never)
                     last, {"end": true, "frames", "analysed", "wall_s", "background": {...}, "stage_ms": {stage: mean ms per analysed frame}} once the whole video is through. Without it the stream stopped short.
    background.jpg   the store with no one in it (background.py), for calibration and for drawing the people over
    skeleton.v3.mp4  made from the two above when first asked for (video): the stick figures on the empty store, the
                     anonymized camera's video for the Analys view after a run, smoothed and at twice the stream's rate

The analysis reads it with `read` and never needs anything else from the anonymizer. Only what the analysis uses
is here: no picture. The body vector describes a body's look without being one; it can be switched off (--no-look).
"""

import base64
import json
import os
import re
import threading
import time
from pathlib import Path

import cv2

import numpy as np

FORMAT = "hyllora-skeleton"
VERSION = 1
FILE = "skeleton.jsonl"
BACKGROUND = "background.jpg"
N_KP = 17
ROW = 6 + 3 * N_KP  # x0 y0 x1 y1 conf tid + (x y c) per keypoint
VIDEO = "skeleton.v3.mp4"  # v3 (2026-11-13): the store not darkened; v2 (2026-11-03): smoothed, twice the rate.
# An older file is not read, so each is made again once
# draw() as the anonymizer's own page draws a frame (static/index.html): the empty store as it is (not darkened,
# Sameh 2026-11-13: "det ska vara samma"), per track a colour,
# a dashed box, the limbs and face (COCO keypoints) and a dot per keypoint from SHOW_C
EDGES = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 1), (0, 2), (1, 3), (2, 4))
COLORS = ("#2F6BFF", "#F76707", "#12A150", "#A61E4D", "#7048E8", "#0B7285", "#E67700")
SHOW_C = 0.5
_video_lock = threading.Lock()


def redact(url: str) -> str:
    """A camera URL without its login, for a header, a list or a message: those leave the anonymizer (S5)."""
    return re.sub(r"//[^/@]*@", "//", str(url))


class Writer:
    """`flush`: every line reaches the file at once (a live camera: the view and the analysis read it as it grows)."""

    def __init__(self, folder: Path, header: dict, flush: bool = False):
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / FILE
        self.flush = flush
        self.f = open(self.path, "w")
        self._line({"format": FORMAT, "version": VERSION, **header})

    def _line(self, obj: dict) -> None:
        self.f.write(json.dumps(obj, separators=(",", ":")) + "\n")
        if self.flush:
            self.f.flush()

    def frame(self, i: int, t: float, xyxy, conf, tid, kxy, kc, looks: dict | None = None, objects=None) -> None:
        rows = []
        for n in range(len(xyxy)):
            row = [round(float(v), 2) for v in xyxy[n]] + [round(float(conf[n]), 3), int(tid[n])]
            for k in range(N_KP):
                row += [round(float(kxy[n, k, 0]), 2), round(float(kxy[n, k, 1]), 2), round(float(kc[n, k]), 3)]
            rows.append(row)
        rec = {"i": i, "t": float(t), "p": rows}  # t unrounded: the analysis matches frames by it
        if looks:
            rec["l"] = {str(k): base64.b64encode(np.asarray(v, np.float32).tobytes()).decode() for k, v in looks.items()}
        if objects is not None:  # looked for on this frame, even when nothing was found
            rec["o"] = [[round(float(v), 2) for v in o[:4]] + [round(float(o[4]), 3), int(o[5])] for o in objects]
        self._line(rec)

    def end(self, **info) -> None:
        self._line({"end": True, **info})
        self.f.close()

    def abort(self) -> None:
        """Stopped short: the file stays without its end line, so a reader knows (fail closed, J3b)."""
        self.f.close()


def header(folder: Path) -> dict:
    with open(Path(folder) / FILE) as f:
        h = json.loads(f.readline())
    if h.get("format") != FORMAT or h.get("version") != VERSION:
        raise ValueError(f"{folder}: inte en skelettström ({FORMAT} {VERSION})")
    return h


def is_live(folder: Path) -> bool:
    """A live camera's stream not yet ended: read as a stream (follow), not as a file. False when unreadable."""
    p = Path(folder) / FILE
    try:
        if not header(folder).get("live"):
            return False
        with open(p, "rb") as f:
            f.seek(max(0, p.stat().st_size - 4096))
            return b'"end":true' not in f.read()
    except (OSError, ValueError):
        return False


def _record(rec: dict):
    if rec.get("end"):
        return "end", rec
    looks = {int(k): np.frombuffer(base64.b64decode(v), np.float32) for k, v in rec.get("l", {}).items()}
    return rec["i"], rec["t"], rec["p"], looks, rec.get("o")


def read(folder: Path):
    """The header, then (i, t, rows, {row: body vector}, objects or None) per analysed frame; the end line is yielded as ("end", info). A stream that
    stopped short simply ends without it."""
    with open(Path(folder) / FILE) as f:
        h = json.loads(f.readline())
        if h.get("format") != FORMAT or h.get("version") != VERSION:
            raise ValueError(f"{folder}: inte en skelettström ({FORMAT} {VERSION})")
        yield h
        for line in f:
            if not line.endswith("\n"):  # half written: the anonymizer is still at it, or died mid-line
                return
            rec = _record(json.loads(line))
            yield rec
            if rec[0] == "end":
                return


def follow(folder: Path, stop=lambda: False, poll_s: float = 0.05, idle_s: float = 1.0):
    """A live stream as it is written (by the anonymizer, or by the analysis' /ingest from another computer): the
    header, then records as `read` gives them, from now on (what was written before is the past). None after each
    `idle_s` with nothing new, so the reader can tell a stream gone quiet and look at `stop`; ("restart", header)
    when the file was started over (a new run under the same name). Ends after the end line or when stop()."""
    path = Path(folder) / FILE

    def opened():
        f = open(path, "rb")
        line = f.readline()
        if not line.endswith(b"\n"):
            f.close()
            return None, None
        h = json.loads(line)
        if h.get("format") != FORMAT or h.get("version") != VERSION:
            f.close()
            raise ValueError(f"{folder}: inte en skelettström ({FORMAT} {VERSION})")
        return f, h

    f, h = opened()
    if f is None:
        raise ValueError(f"{folder}: strömmen har inget huvud än")
    yield h
    start = f.tell()
    f.seek(start + f.read().rfind(b"\n") + 1)  # from now on: past the last whole line written so far
    last = time.monotonic()
    try:
        while not stop():
            pos = f.tell()
            line = f.readline()
            if line.endswith(b"\n"):
                last = time.monotonic()
                rec = _record(json.loads(line))
                yield rec
                if rec[0] == "end":
                    return
                continue
            f.seek(pos)
            try:
                st = path.stat()
                started_over = st.st_ino != os.fstat(f.fileno()).st_ino or st.st_size < pos
            except OSError:
                started_over = False
            if started_over:
                g, h2 = opened()
                if g is not None:
                    f.close()
                    f = g
                    last = time.monotonic()
                    yield "restart", h2
                    continue
            if time.monotonic() - last >= idle_s:
                last = time.monotonic()
                yield None
            time.sleep(poll_s)
    finally:
        f.close()


# What may cross from the anonymizer to the analysis (/ingest, J3 (c)): the stream's own lines and nothing else, so a
# picture can never be passed off as one. The header's and the end line's keys as Writer and main.run write them.
HEADER_KEYS = {"format", "version", "source", "size", "roi", "src_fps", "rate", "stride", "frames", "model", "imgsz",
               "conf", "tracker", "background", "look", "objects", "objects_model", "objects_every_s", "live", "t0"}
END_KEYS = {"end", "frames", "analysed", "wall_s", "background", "stage_ms", "gaps"}
FRAME_KEYS = {"i", "t", "p", "l", "o"}
LINE_MAX = 256 * 1024  # bytes: a frame with 50 people and their vectors is ~60 KB
LOOK_MAX = 8192  # base64 characters for one body vector (a 512-float32 vector is 2732)
PEOPLE_MAX = 200


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _small(v, depth: int = 0) -> bool:
    """Plain values of a header or end line: numbers, short words, short lists and small objects, never a blob."""
    if v is None or isinstance(v, bool) or _num(v):
        return True
    if isinstance(v, str):
        return len(v) <= 200
    if depth >= 3:
        return False
    if isinstance(v, list):
        return len(v) <= 256 and all(_small(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return len(v) <= 64 and all(isinstance(k, str) and len(k) <= 64 and _small(x, depth + 1) for k, x in v.items())
    return False


def check_line(line: bytes, first: bool) -> dict:
    """One line of a stream as it may cross to the analysis, parsed; ValueError for anything else. `first`: the
    header (the stream's first line). A frame line holds numbers only, and body vectors of a vector's size."""
    if len(line) > LINE_MAX:
        raise ValueError("raden är för lång")
    try:
        obj = json.loads(line)
    except ValueError:
        raise ValueError("raden är inte JSON")
    if not isinstance(obj, dict):
        raise ValueError("raden är inte ett objekt")
    if first:
        if obj.get("format") != FORMAT or obj.get("version") != VERSION:
            raise ValueError(f"första raden är inte ett huvud ({FORMAT} {VERSION})")
        if set(obj) - HEADER_KEYS or not all(_small(v) for v in obj.values()):
            raise ValueError("huvudet har fält som inte hör dit")
        if not (isinstance(obj.get("size"), list) and len(obj["size"]) == 2 and all(_num(v) for v in obj["size"])):
            raise ValueError("huvudet saknar storlek")
        return obj
    if obj.get("end") is True:
        if set(obj) - END_KEYS or not all(_small(v) for v in obj.values()):
            raise ValueError("slutraden har fält som inte hör dit")
        return obj
    if set(obj) - FRAME_KEYS or not (_num(obj.get("i")) and _num(obj.get("t"))):
        raise ValueError("bildrutans rad har fält som inte hör dit")
    p = obj.get("p")
    if not (isinstance(p, list) and len(p) <= PEOPLE_MAX
            and all(isinstance(r, list) and len(r) == ROW and all(_num(v) for v in r) for r in p)):
        raise ValueError("personerna är inte rader av tal")
    looks = obj.get("l", {})
    if not (isinstance(looks, dict) and all(
            isinstance(k, str) and k.isdigit() and int(k) < len(p) and isinstance(v, str) and len(v) <= LOOK_MAX
            for k, v in looks.items())):
        raise ValueError("kroppsvektorerna har fel form")
    for v in looks.values():
        # a body vector is unit length (reid.describe_many normalises it): finite floats of norm 1, which a picture's
        # bytes read as float32 never are
        try:
            b = base64.b64decode(v, validate=True)
            if not b or len(b) % 4:
                raise ValueError
            x = np.frombuffer(b, np.float32).astype(np.float64)
            if not (np.isfinite(x).all() and abs(np.linalg.norm(x) - 1.0) < 0.01):
                raise ValueError
        except ValueError:
            raise ValueError("en kroppsvektor är inte en vektor av längd 1")
    o = obj.get("o")
    if o is not None and not (isinstance(o, list) and len(o) <= PEOPLE_MAX
                              and all(isinstance(r, list) and len(r) == 6 and all(_num(v) for v in r) for r in o)):
        raise ValueError("sakerna är inte rader av tal")
    return obj


def draw(img: np.ndarray, rows) -> np.ndarray:
    """One frame's people on the empty store, as the anonymizer's page shows them: all an anonymized camera has to
    show. Made from the stream's numbers only, so nothing of anyone's picture is in it."""
    out = img.copy()
    k = out.shape[1] / 1280
    for r in np.asarray(rows, np.float64).reshape(-1, ROW):
        c = COLORS[int(r[5]) % len(COLORS)]
        col = tuple(int(c[i:i + 2], 16) for i in (5, 3, 1))  # BGR
        _dashed_box(out, r[:4], col, max(1, round(1.5 * k)), 6 * k, 5 * k)
        kp = r[6:].reshape(N_KP, 3)
        seen = kp[:, 2] >= SHOW_C
        pts = [(int(round(x)), int(round(y))) for x, y in kp[:, :2]]
        for i, j in EDGES:
            if seen[i] and seen[j]:
                cv2.line(out, pts[i], pts[j], col, max(2, round(4 * k)), cv2.LINE_AA)
        for i in np.flatnonzero(seen):
            cv2.circle(out, pts[i], max(2, round(4.5 * k)), col, -1, cv2.LINE_AA)
    return out


def _dashed_box(img, box, col, w, dash, gap) -> None:
    x0, y0, x1, y1 = box
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        a, b = np.array(a), np.array(b)
        n = np.linalg.norm(b - a)
        for s in np.arange(0, n, dash + gap):
            p, q = a + (b - a) * s / max(n, 1), a + (b - a) * min(s + dash, n) / max(n, 1)
            cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), col, w, cv2.LINE_AA)


def video(folder: Path) -> Path:
    """The stream as a video (H.264 in MP4, which the browser plays): the stick figures on the empty store, at twice
    the stream's rate (`smooth`), so its time is the video's. Made once and kept beside the stream."""
    folder = Path(folder)
    out = folder / VIDEO
    with _video_lock:
        if out.is_file() and out.stat().st_mtime >= (folder / FILE).stat().st_mtime:
            return out
        stream = read(folder)
        h = next(stream)
        W, H = h["size"]
        bg = cv2.imread(str(folder / BACKGROUND))
        bg = cv2.resize(bg, (W, H)) if bg is not None else np.full((H, W, 3), 128, np.uint8)
        frames = [rec[2] for rec in stream if rec[0] != "end"]
        tmp = folder / f".{VIDEO}.{os.getpid()}.mp4"
        w = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"avc1"), 2 * float(h["rate"]), (W, H))
        if not w.isOpened():
            raise OSError("Kan inte skriva H.264-video (avc1) med OpenCV")
        for rows in smooth(frames):
            w.write(draw(bg, rows))
        w.release()
        os.replace(tmp, out)
        return out


SMOOTH_SIGMA = 1.0  # analysed frames: the keypoints jump between frames about as much as people move (medurummet
# 2026-11-03: 4.4 px against 4.3 px a frame), so each is averaged over its neighbours of the same track
SMOOTH_REACH = 2


def smooth(frames):
    """Per analysed frame, two frames to draw: the people with their keypoints and boxes averaged over time per track
    (Gaussian over ±SMOOTH_REACH frames, weighted by keypoint confidence), and halfway to the next frame, for whoever
    is in both. For showing only: the analysis reads the stream itself."""
    rows = [np.asarray(f, np.float64).reshape(-1, ROW) for f in frames]
    out = [r.copy() for r in rows]
    where = {}  # track -> {frame: row index}
    for k, r in enumerate(rows):
        for n, tid in enumerate(r[:, 5].astype(int)):
            where.setdefault(tid, {})[k] = n
    g = {d: np.exp(-0.5 * (d / SMOOTH_SIGMA) ** 2) for d in range(-SMOOTH_REACH, SMOOTH_REACH + 1)}
    for at in where.values():
        for k, n in at.items():
            near = [(rows[k + d][at[k + d]], g[d]) for d in g if k + d in at]
            box = sum(w * r[:4] for r, w in near) / sum(w for _, w in near)
            kp = np.stack([r[6:].reshape(N_KP, 3) for r, _ in near])
            wk = np.array([w for _, w in near])[:, None] * np.where(kp[:, :, 2] >= SHOW_C, kp[:, :, 2], 0)
            own = out[k][n, 6:].reshape(N_KP, 3)
            ok = wk.sum(axis=0) > 0
            own[ok, :2] = (wk[:, ok, None] * kp[:, ok, :2]).sum(axis=0) / wk[:, ok].sum(axis=0)[:, None]
            out[k][n, :4] = box
    for k, r in enumerate(out):
        yield r
        nxt = out[k + 1] if k + 1 < len(out) else None
        if nxt is None:
            yield r
            continue
        ids = {int(t): n for n, t in enumerate(nxt[:, 5])}
        mid = []
        for row in r:
            m = ids.get(int(row[5]))
            if m is None:
                continue
            o = row.copy()
            o[:4] = (row[:4] + nxt[m, :4]) / 2
            a, b = row[6:].reshape(N_KP, 3), nxt[m, 6:].reshape(N_KP, 3)
            kp = o[6:].reshape(N_KP, 3)
            both = (a[:, 2] >= SHOW_C) & (b[:, 2] >= SHOW_C)
            kp[both, :2] = (a[both, :2] + b[both, :2]) / 2
            kp[:, 2] = np.minimum(a[:, 2], b[:, 2])
            mid.append(o)
        yield np.array(mid).reshape(-1, ROW)
