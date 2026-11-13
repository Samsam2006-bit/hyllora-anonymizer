"""The anonymizer: pictures in, a skeleton stream out (skeleton.py). The picture is only held in memory, one frame at
a time; nothing of it is written but the empty store (background.py).

    .venv/bin/python -m anonymizer.main <video> <out folder> [--stride 1] [--roi x0,y0,x1,y1]
    .venv/bin/python -m anonymizer.main rtsp://… <out folder> [--seconds 60]     (a live camera: Ctrl-C ends it)
    .venv/bin/python -m anonymizer.main rtsp://… <out folder> --send http://<analysis>:8420   (and sent on, send.py)
    echo rtsp://… | .venv/bin/python -m anonymizer.main - <out folder>           (the URL kept out of the process list)

Three parts, so a file and a live camera share everything but where frames come from and where rows go (CURRENT_STATE
J3, plan 2026-11-09):

    source  FileSource: every stride-th frame of a video, in video time (the night's mode, J2).
            StreamSource: a live camera (RTSP, HTTP MJPEG), always its newest frame, timed by when it arrived;
            a dropped camera is opened again, and the time it gave nothing is a gap (J3 (b), 2026-11-11).
    core    Core.step(frame, t): detector, pose and tracker, the body vector, the store's things, a sample for the
            empty store. The frame is dropped as soon as step returns.
    output  skeleton.Writer: the stream as a file, and with --send the same file sent on to the analysis on another
            computer as it grows (send.Sender, POST /api/ingest, J3 (c), 2026-11-12).

Frames are read as the analysis read them (OpenCV, the frame's own timestamp), so the times are the analysis' own.
The detector, pose and tracker are the analysis' defaults (inferero.pipeline.config.DEFAULT_SETTINGS), repeated
here rather than imported: the anonymizer never imports the pipeline. Each run times its stages (Stages) and puts
the mean per analysed frame in the stream's end line, so we know where the time goes before making it faster.
"""

import argparse
import contextlib
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inferero  # noqa: E402,F401  (offline switches before ultralytics is imported)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from inferero import reid, weights  # noqa: E402

from . import background, looks, objects, send as sending, skeleton  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BOTSORT = ROOT / "inferero" / "botsort.yaml"
MODEL, IMGSZ, CONF = "yolo26s-pose.pt", 960, 0.3  # = DEFAULT_SETTINGS
TRACK_BUFFER_S = 1.5  # = pipeline.config.TRACK_BUFFER_S["botsort"]
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
HALF = DEVICE != "cpu"
SAMPLES = 30  # frames spread over the video that the empty store is made from (background.py)
STREAM_RATE_GUESS = 10.0  # = pipeline.config.REALTIME_FPS_GUESS: analysed frames per second assumed from a camera
RETRY_MAX_S = 30.0  # = pipeline.config.STREAM_RETRY_MAX_S: a dropped camera is tried after 1, 2, 4 … s, never further
NO_FRAME_S = 5.0  # a camera that sends nothing this long has dropped (as LatestFrame.get's timeout in the analysis)
LIVE_EVERY = 10  # a live camera: a sample for the empty store every this many analysed frames, so it is ready in
                 # ~SAMPLES * LIVE_EVERY frames (~12 s at 25/s) and the anonymized view has the store to draw on


def tracker_cfg(out: Path, rate: float) -> str:
    cfg = yaml.safe_load(BOTSORT.read_text())
    cfg["track_buffer"] = max(1, round(TRACK_BUFFER_S * rate))
    cfg["gmc_method"] = "none"
    path = out / "tracker.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def infer(model, frame, roi, tracker: str | None, imgsz: int, conf: float):
    """Boxes, confidences, track ids and keypoints in full-frame pixels, as CameraWorker._infer had them.
    Without a tracker (the background's final check): plain detection, ids -1."""
    x0, y0 = (roi[0], roi[1]) if roi else (0, 0)
    img = frame[roi[1]:roi[3], roi[0]:roi[2]] if roi else frame
    kw = dict(classes=[0], conf=conf, imgsz=imgsz, device=DEVICE, verbose=False, quantize=16 if HALF else None)
    r = (model.track(img, persist=True, tracker=tracker, **kw) if tracker else model.predict(img, **kw))[0]
    b = r.boxes
    if b is None or not len(b) or (tracker and b.id is None):
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros((0, 17, 2)), np.zeros((0, 17))
    xyxy = b.xyxy.cpu().numpy() + np.array([x0, y0, x0, y0])
    n = len(xyxy)
    kxy, kc = np.zeros((n, 17, 2)), np.zeros((n, 17))
    if r.keypoints is not None and r.keypoints.conf is not None:
        kxy, kc = r.keypoints.xy.cpu().numpy() + np.array([x0, y0]), r.keypoints.conf.cpu().numpy()
    tid = b.id.cpu().numpy().astype(int) if tracker else np.full(n, -1)
    return xyxy, b.conf.cpu().numpy(), tid, kxy, kc


class Stages:
    """Wall time per stage, summed over a run: `with stages("model"): ...` (nesting is fine: each `with` keeps its
    own start). `mean_ms(n)` per analysed frame."""

    def __init__(self):
        self.total: dict[str, float] = {}

    @contextlib.contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def add(self, name: str, seconds: float) -> None:
        self.total[name] = self.total.get(name, 0.0) + seconds

    def mean_ms(self, n: int) -> dict[str, float]:
        return {k: round(1000.0 * v / max(n, 1), 2) for k, v in self.total.items()}


class FileSource:
    """Every stride-th frame of a video file: (frame number, video time, frame). 1 is the first frame."""

    def __init__(self, src: str, stride: int = 1):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise ValueError(f"Kan inte öppna {src}")
        self.name, self.stride = Path(src).name, max(1, stride)
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.rate = self.src_fps / self.stride
        self.idx = 0  # frames read, analysed or not

    def frames(self, stages: Stages):
        prev_t = None
        while True:
            t0 = time.perf_counter()
            ok, frame = self.cap.read()
            if ok:
                self.idx += 1
            stages.add("read", time.perf_counter() - t0)  # skipped frames are decoded too: they count here
            if not ok:
                return
            if self.idx % self.stride:
                continue
            t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if prev_t is not None and not t > prev_t:  # backend without timestamps (as CameraWorker._run)
                t = prev_t + self.stride / self.src_fps
            prev_t = t
            yield self.idx, t, frame

    def close(self) -> None:
        self.cap.release()


def is_stream(src: str) -> bool:
    return "://" in str(src)


redact = skeleton.redact


class Latest(threading.Thread):
    """Reads a camera as fast as it sends and keeps only the newest frame, its number (counted from 1 over the
    whole run, so skipped frames show as jumps in "i") and when it arrived (as the analysis' LatestFrame, not
    imported: the anonymizer never loads the pipeline). The thread owns the capture and releases it itself."""

    def __init__(self, cap, count: int):
        super().__init__(daemon=True, name="camera")
        self.cap, self.lock, self.new = cap, threading.Lock(), threading.Event()
        self.frame, self.at, self.ok, self.n, self.stopped = None, 0.0, True, count, False

    def run(self) -> None:
        try:
            while not self.stopped:
                ok, frame = self.cap.read()
                with self.lock:
                    self.ok, self.frame, self.at = ok, frame, time.monotonic()
                    if ok:
                        self.n += 1
                self.new.set()
                if not ok:
                    break
        finally:
            self.cap.release()

    def get(self, timeout: float):
        """(ok, newest frame, its number, arrival); waits for one newer than the last handed out."""
        if not self.new.wait(timeout):
            return False, None, self.n, time.monotonic()
        with self.lock:
            self.new.clear()
            return self.ok, self.frame, self.n, self.at

    def stop(self) -> None:
        self.stopped = True


class StreamSource:
    """A live camera: (frame number, seconds since the run started, frame), always the newest frame, so a core slower
    than the camera skips frames instead of falling behind. `t0` is the wall clock at t = 0. A camera that drops is
    opened again after 1, 2, 4 … s; the time it gave nothing is kept in `gaps` ([from, to] in t). Ends on `stop()`."""

    live = True

    def __init__(self, url: str, opener=None, no_frame_s: float = NO_FRAME_S, retry_max_s: float = RETRY_MAX_S):
        # a URL goes straight to FFmpeg: OpenCV's other backends write the whole URL, login included, to the log (S5)
        self.opener = opener or (lambda u: cv2.VideoCapture(u, cv2.CAP_FFMPEG))
        self.url, self.name = url, redact(url)
        self.no_frame_s, self.retry_max_s = no_frame_s, retry_max_s
        self.stride, self.total, self.idx = 1, 0, 0
        self.gaps: list[list[float]] = []
        self._stop = threading.Event()
        cap = self.opener(url)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Kan inte öppna kameran {self.name}")
        self.src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.rate = min(self.src_fps or STREAM_RATE_GUESS, STREAM_RATE_GUESS)
        self.t0_mono, self.t0 = time.monotonic(), time.time()
        self.reader = Latest(cap, 0)
        self.reader.start()
        ok, frame, _, _ = self.reader.get(self.no_frame_s)  # the size from the first frame: MJPEG has none to ask
        if not ok:
            self.close()
            raise ValueError(f"Kameran {self.name} skickar inga bilder")
        self.size = frame.shape[1], frame.shape[0]
        self._first = frame

    def frames(self, stages: Stages):
        if self._first is not None:
            frame, self._first = self._first, None
            self.idx = self.reader.n
            yield self.idx, self.reader.at - self.t0_mono, frame
        tries = 0
        while not self._stop.is_set():
            t0 = time.perf_counter()
            ok, frame, n, at = self.reader.get(self.no_frame_s)
            stages.add("read", time.perf_counter() - t0)  # mostly waiting for the camera's next frame
            if self._stop.is_set():
                return
            if ok and frame is not None:
                tries = 0
                if self.gaps and self.gaps[-1][1] is None:
                    self.gaps[-1][1] = round(at - self.t0_mono, 3)
                self.idx = n
                yield n, at - self.t0_mono, frame
                continue
            # dropped (closed, or nothing for no_frame_s): open it again, waiting longer each time
            if not self.gaps or self.gaps[-1][1] is not None:
                self.gaps.append([round(time.monotonic() - self.t0_mono, 3), None])
            self.reader.stop()
            if self._stop.wait(min(self.retry_max_s, 2 ** tries)):
                return
            tries += 1
            cap = self.opener(self.url)
            if not cap.isOpened():
                cap.release()
                continue
            self.reader = Latest(cap, self.reader.n)
            self.reader.start()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.reader.stop()
        if self.gaps and self.gaps[-1][1] is None:
            self.gaps[-1][1] = round(time.monotonic() - self.t0_mono, 3)


class Core:
    """What sees the picture: one frame in, its skeleton rows out (step), and the empty store at the end (background).
    `every`: a sample for the empty store every this many analysed frames, SAMPLES at most."""

    def __init__(self, size, roi, rate: float, out: Path, look: bool = True, things: list[dict] | None = None,
                 every: int = 50, samples: int = SAMPLES, pose: str | None = None):
        from ultralytics import YOLO

        W, H = size
        self.roi = tuple(int(v) for v in roi) if roi and tuple(roi) != (0, 0, W, H) else None
        self.model = YOLO(weights.path(MODEL))
        # its own model for the background: a predict on the tracking model would run through its tracker's callbacks
        self.check = YOLO(weights.path(MODEL))
        self.trk = tracker_cfg(out, rate)
        self.pose = pose  # None: the analysis' default (MODEL, BoT-SORT); else a backends.names() entry
        self.backend = None
        if pose:
            from . import backends
            self.backend = backends.Backend(pose, CONF, IMGSZ, DEVICE, HALF, rate, TRACK_BUFFER_S, weights.path)
        self.describer = looks.Looks(W, self.roi) if look else None
        self.things = things or None
        self.finder = objects.Finder(things, DEVICE, HALF) if things else None
        self.next_find = -1.0
        self.plate = background.Plate((W, H), samples)
        self.every, self.samples = max(1, every), samples
        self.analysed = 0
        self.stages = Stages()

    def header(self) -> dict:
        """The core's part of the stream's first line."""
        return {"roi": list(self.roi) if self.roi else None, "model": self.pose or MODEL, "imgsz": IMGSZ, "conf": CONF,
                "tracker": "bytetrack" if self.pose else "botsort", "background": skeleton.BACKGROUND,
                "look": reid.REID_MODEL if self.describer else None,
                "objects": [o["name"] for o in self.things] if self.things else None,
                "objects_model": objects.MODEL if self.things else None, "objects_every_s": objects.EVERY_S}

    def step(self, frame: np.ndarray, t: float) -> tuple:
        """(xyxy, conf, tid, kxy, kc, looks or None, things found or None) for one frame at time t."""
        s = self.stages
        with s("model"):
            if self.backend is None:
                xyxy, conf, tid, kxy, kc = infer(self.model, frame, self.roi, self.trk, IMGSZ, CONF)
            else:
                xyxy, conf, tid, kxy, kc = self._other(frame)
        found = None
        if self.finder is not None and t >= self.next_find:
            with s("objects"):
                found, self.next_find = self.finder.find(frame, self.roi), t + objects.EVERY_S
        described = None
        if self.describer is not None:
            with s("look"):
                described = self.describer.describe(frame, xyxy, conf, tid, t)
        if self.analysed % self.every == 0 and len(self.plate.frames) < self.samples:
            with s("background"):
                # every person the detector sees, tracked or not yet, is left out of the empty store
                self.plate.add(frame, infer(self.check, frame, self.roi, None, IMGSZ, 0.15)[0])
        self.analysed += 1
        return xyxy, conf, tid, kxy, kc, described, found

    def _other(self, frame):
        """The same as infer with a tracker, from another model (backends.py)."""
        x0, y0, x1, y1 = self.roi or (0, 0, frame.shape[1], frame.shape[0])
        xyxy, conf, tid, kxy, kc = self.backend.track(frame[y0:y1, x0:x1])
        if len(xyxy):
            xyxy, kxy = xyxy + np.array([x0, y0, x0, y0]), kxy + np.array([x0, y0])
        return xyxy, conf, tid, kxy, kc

    def background_ready(self) -> bool:
        """All the samples for the empty store are in."""
        return len(self.plate.frames) >= self.samples

    def background(self):
        """The empty store (an image or None) and what it says about itself."""
        with self.stages("background"):
            return self.plate.build(self.check, lambda m, img: infer(m, img, None, None, IMGSZ, background.CHECK_CONF)[0])


def run(src: str, out: Path, stride: int = 1, roi=None, samples: int = SAMPLES, look: bool = True, progress=None,
        stop=None, things: list[dict] | None = None, pose: str | None = None, seconds: float | None = None,
        send: str | None = None, send_key: str | None = None) -> dict:
    """A video file or a live camera (a URL) through the core into a skeleton file. `progress(frame, total)` now and
    then; `things`: what the store showed a photo of (objects.load), looked for every objects.EVERY_S; None or []:
    nothing. A file: `stop()` true ends it short (no end line: the analysis sees it stopped). A camera has no end of
    its own: `stop()`, `seconds` or Ctrl-C ends it whole, with the empty store and the end line. `send`: the analysis'
    address on another computer; the stream goes there as it is written (send.Sender, with `send_key` or send.key())."""
    progress = progress or (lambda i, n: print(f"  {i}/{n} bildrutor" if n else f"  bildruta {i}", flush=True))
    send_key = (send_key or sending.key()) if send else None  # no key: stop before the camera is opened
    source = StreamSource(src) if is_stream(src) else FileSource(src, stride)
    live = getattr(source, "live", False)
    try:
        out.mkdir(parents=True, exist_ok=True)
        every = max(1, source.total // source.stride // samples) if source.total else LIVE_EVERY if live else 50
        core = Core(source.size, roi, source.rate, out, look, things, every, samples, **({"pose": pose} if pose else {}))
        head = {"source": source.name, "size": list(source.size), "src_fps": source.src_fps, "rate": source.rate,
                "stride": source.stride, "frames": source.total, **core.header()}
        if live:  # t is seconds since t0 (the wall clock, unix seconds): a camera has no video time of its own
            head.update(live=True, t0=round(source.t0, 3))
        w = skeleton.Writer(out, head, flush=live or bool(send))
        sender = sending.Sender(out, send, send_key) if send else None
        if sender:
            sender.start()
        t0 = time.monotonic()
        bg = bg_info = None
        try:
            try:
                for idx, t, frame in source.frames(core.stages):
                    rows = core.step(frame, t)
                    with core.stages("write"):
                        w.frame(idx, t, *rows)
                    if live and bg_info is None and core.background_ready():
                        # the empty store once its samples are in, not at the end: the view draws on it meanwhile
                        bg, bg_info = _write_background(core, out)
                    if core.analysed % 100 == 0:
                        progress(idx, source.total)
                    if live and ((stop is not None and stop()) or (seconds and time.monotonic() - t0 >= seconds)):
                        source.stop()
                    elif not live and stop is not None and stop():
                        raise InterruptedError("stoppad")
            except KeyboardInterrupt:
                if not live:
                    raise
            source.close()  # through (or a camera stopped): let go of it before the empty store is made
            if bg_info is None:
                bg, bg_info = _write_background(core, out)
        except BaseException:
            w.abort()  # no end line, also when the empty store failed: the analysis never takes it for whole
            if sender:
                sender.stop()
            raise
    finally:
        source.close()
    info = {"frames": source.idx, "analysed": core.analysed, "wall_s": round(time.monotonic() - t0, 1),
            "background": bg_info, "stage_ms": core.stages.mean_ms(core.analysed)}
    if live:
        info["gaps"] = source.gaps
    w.end(**info)
    if sender:  # what is left goes on, the end line last (DRAIN_S at most)
        sender.stop()
        sender.join(sending.DRAIN_S + 5)
        info["sent"] = sender.status()
    return info


def _write_background(core, out: Path):
    """The empty store from the core's samples, written next to the stream (via a temporary file: a reader never
    gets half a picture)."""
    bg, info = core.background()
    if bg is not None:
        tmp = out / ("." + skeleton.BACKGROUND)
        cv2.imwrite(str(tmp) + ".jpg", bg, [cv2.IMWRITE_JPEG_QUALITY, 90])
        Path(str(tmp) + ".jpg").replace(out / skeleton.BACKGROUND)
    return bg, info


def _stop_on_stdin() -> None:
    """After the address, the page says "stop" on stdin (server.LiveJob.stop), not with a signal: a Windows
    service has no console to send one through. Stdin closing (the page's process ended) stops too, so a camera never
    runs on without the page. Either is a Ctrl-C in the main thread, so the stream ends whole, as before."""
    import _thread

    def watch():
        for line in sys.stdin:
            if line.strip() == "stop":
                break
        _thread.interrupt_main()

    threading.Thread(target=watch, daemon=True, name="stdin-stop").start()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="a video file, a camera's URL, or - for a URL on stdin")
    ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=1, help="every n-th frame (the analysis' stride)")
    ap.add_argument("--roi", help="x0,y0,x1,y1: the part of the frame that is camera picture (lens.py)")
    ap.add_argument("--no-look", action="store_true", help="no body vector (looks.py): more ID swaps, nothing of the look")
    ap.add_argument("--no-objects", action="store_true", help="do not look for what data/objects has photos of (objects.py)")
    ap.add_argument("--pose", help="another person and pose model to compare (backends.names()); default the analysis' own")
    ap.add_argument("--seconds", type=float, help="a live camera: stop after this many seconds (else Ctrl-C)")
    ap.add_argument("--send", help="the analysis' address on another computer (send.py): the stream goes there as it grows")
    a = ap.parse_args()
    if a.src == "-":  # the camera's address on stdin: with its login it must not be in the process list (server.py)
        a.src = sys.stdin.readline().strip()
        _stop_on_stdin()
    roi = [int(v) for v in a.roi.split(",")] if a.roi else None
    info = run(a.src, Path(a.out), max(1, a.stride), roi, look=not a.no_look, things=None if a.no_objects else objects.load(), pose=a.pose, seconds=a.seconds,
               send=a.send)
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
