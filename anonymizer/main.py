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

from inferero import gpu, reid, weights  # noqa: E402

from . import background, looks, objects, paths, send as sending, settings, skeleton  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BOTSORT = ROOT / "inferero" / "botsort.yaml"
MODEL, IMGSZ, CONF = settings.MODEL, settings.IMGSZ, 0.3  # = DEFAULT_SETTINGS
TRACK_BUFFER_S = 1.5  # = pipeline.config.TRACK_BUFFER_S["botsort"]
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
HALF = gpu.half_ok(DEVICE)  # not on a GTX 10xx: its 16-bit floats are many times slower than 32-bit
SAMPLES = 30  # frames spread over the video that the empty store is made from (background.py)
STREAM_RATE_GUESS = 10.0  # = pipeline.config.REALTIME_FPS_GUESS: analysed frames per second assumed from a camera
RETRY_MAX_S = 30.0  # = pipeline.config.STREAM_RETRY_MAX_S: a dropped camera is tried after 1, 2, 4 … s, never further
NO_FRAME_S = 5.0  # a camera that sends nothing this long has dropped (as LatestFrame.get's timeout in the analysis)
OPEN_TIMEOUT_MS = 10_000  # a camera that does not answer is an error after this, not a page that says nothing (OpenCV's
                          # own default is 30 s, and the store's computer showed "bildruta 0" without a word, 2026-11-13)
READ_TIMEOUT_MS = 10_000  # a read that hangs gives up after this, so a dropped camera is opened again
# Settings a live camera can be started with (the page's Läge, 2026-11-17, Sameh: "gör så man kan toggla så man kan
# se vilken som funkar bäst"). Full is the analysis' own; Lätt is what 20 cameras on one GTX 1080 may need: the small
# model at 640 px, 5 frames a second, no body vector, and no model on a camera where nothing moves.
PRESETS, MODELS_OK, IMGSZ_OK = settings.PRESETS, settings.MODELS_OK, settings.IMGSZ_OK
CHECK_IMGSZ = 960  # the empty store's check and its samples' masks at full size whatever the camera runs at
MOTION_W = 160  # the motion check compares frames this wide, blurred
MOTION_DIFF = 18  # grey levels for a pixel to have changed
MOTION_SHARE = 0.002  # of the pixels: more changed than this and the camera has motion
MOTION_EVERY_S = 2.0  # the model still runs this often on a still camera (someone may stand without moving)
STATS_EVERY_S = 5.0  # a stats line ("~ {...}") this often: the page shows analysed frames a second per camera
LIVE_EVERY = 10  # a live camera: a sample for the empty store every this many analysed frames, so it is ready in
                 # ~SAMPLES * LIVE_EVERY frames (~12 s at 25/s) and the anonymized view has the store to draw on


def tracker_cfg(out: Path, rate: float) -> str:
    cfg = yaml.safe_load(BOTSORT.read_text())
    cfg["track_buffer"] = max(1, round(TRACK_BUFFER_S * rate))
    cfg["gmc_method"] = "none"
    path = out / "tracker.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def make_tracker(rate: float):
    """A camera's own BoT-SORT, as model.track makes it from the same settings (tracker_cfg): for cameras whose
    frames go through the model together (fleet.py), where one model has no tracker per camera."""
    from ultralytics.trackers.bot_sort import BOTSORT as BoTSORT
    from ultralytics.utils import IterableSimpleNamespace

    cfg = yaml.safe_load(BOTSORT.read_text())
    cfg["track_buffer"] = max(1, round(TRACK_BUFFER_S * rate))
    cfg["gmc_method"] = "none"
    return BoTSORT(IterableSimpleNamespace(**cfg))


def detect(model, imgs: list, imgsz: int, conf: float):
    """People and keypoints in several frames in one call (a batch): the results, untracked."""
    with gpu.GPU_LOCK:
        return model.predict(imgs, classes=[0], conf=conf, imgsz=imgsz, device=DEVICE, verbose=False,
                             quantize=16 if HALF else None)


def track(tracker, r, offset=(0, 0)):
    """One camera's detections (a result of detect) through its tracker, as model.track does it
    (ultralytics.trackers.track.on_predict_postprocess_end): every frame updates the tracker, also one with nobody."""
    x0, y0 = offset
    tracks = tracker.update(r.boxes.cpu().numpy(), r.orig_img)
    if not len(tracks):
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros((0, 17, 2)), np.zeros((0, 17))
    idx = tracks[:, -1].astype(int)
    n = len(idx)
    kxy, kc = np.zeros((n, 17, 2)), np.zeros((n, 17))
    if r.keypoints is not None and r.keypoints.conf is not None:
        kxy, kc = r.keypoints.xy.cpu().numpy()[idx] + np.array([x0, y0]), r.keypoints.conf.cpu().numpy()[idx]
    h, w = r.orig_shape[:2]  # as Results.update does: the tracker's boxes inside the picture
    xyxy = np.clip(tracks[:, :4], 0, [w, h, w, h])
    return xyxy + np.array([x0, y0, x0, y0]), tracks[:, 5], tracks[:, 4].astype(int), kxy, kc


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


def open_camera(url: str, hwdec: bool = False):
    """A camera's URL straight to FFmpeg (OpenCV's other backends write the whole URL, login included, to the log),
    with a time limit for opening and for each read: without them a camera that does not answer leaves the run
    waiting without a word. `hwdec`: the video decoded on the graphics card where FFmpeg can (d3d11va on Windows),
    else on the processor as before; every frame of every camera is decoded, also those the model skips, so twenty
    cameras' decoding is many cores."""
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS]
    if hwdec:
        params += [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)


_OUT = threading.Lock()


def emit(line: str) -> None:
    """One whole line on stdout: threads printing at once (fleet.py's cameras) otherwise run their lines together,
    and the page then reads one camera's step as another's (2026-11-17)."""
    with _OUT:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def say(msg: str) -> None:
    """What the run is doing now, one line on stdout: the page shows a live camera's latest line (server.LiveJob),
    so a camera that gives nothing can be told from models still loading or a view that does not show."""
    emit(msg)


def device_name() -> str:
    if DEVICE == "cuda":
        return f"grafikkortet {torch.cuda.get_device_name(0)}"
    return "Apples grafikkort" if DEVICE == "mps" else "processorn (inget grafikkort hittat, långsamt)"


class StreamSource:
    """A live camera: (frame number, seconds since the run started, frame), always the newest frame, so a core slower
    than the camera skips frames instead of falling behind. `t0` is the wall clock at t = 0. A camera that drops is
    opened again after 1, 2, 4 … s; the time it gave nothing is kept in `gaps` ([from, to] in t). Ends on `stop()`."""

    live = True

    def __init__(self, url: str, opener=None, no_frame_s: float = NO_FRAME_S, retry_max_s: float = RETRY_MAX_S,
                 max_fps: float = 0.0, hwdec: bool = False, say=None):
        """`max_fps`: hand out at most this many frames a second (0: as fast as they are taken), the newest each time.
        `hwdec`: open_camera's."""
        self.opener = opener or (lambda u: open_camera(u, hwdec))
        self.say = say or globals()["say"]  # fleet.py: the camera's own lines carry its name
        self.max_fps, self._next = float(max_fps or 0), 0.0
        self.url, self.name = url, redact(url)
        self.no_frame_s, self.retry_max_s = no_frame_s, retry_max_s
        self.stride, self.total, self.idx = 1, 0, 0
        self.gaps: list[list[float]] = []
        self._stop = threading.Event()
        cap = self.opener(url)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Kan inte öppna kameran {self.name} (inget svar på {OPEN_TIMEOUT_MS // 1000} s: fel adress, "
                             "inloggning, eller så tar kameran inte fler anslutningar; prova Testa kameran)")
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
            self._next = time.monotonic() + (1.0 / self.max_fps if self.max_fps else 0.0)  # the first one counts too
            yield self.idx, self.reader.at - self.t0_mono, frame
        tries = 0
        while not self._stop.is_set():
            if self.max_fps:  # wait for the next one due, then take the newest the camera has sent
                wait = self._next - time.monotonic()
                if wait > 0 and self._stop.wait(wait):
                    return
                self._next = time.monotonic() + 1.0 / self.max_fps
            t0 = time.perf_counter()
            ok, frame, n, at = self.reader.get(self.no_frame_s)
            stages.add("read", time.perf_counter() - t0)  # mostly waiting for the camera's next frame
            if self._stop.is_set():
                return
            if ok and frame is not None:
                tries = 0
                if self.gaps and self.gaps[-1][1] is None:
                    self.gaps[-1][1] = round(at - self.t0_mono, 3)
                    self.say(f"Kameran är tillbaka efter {self.gaps[-1][1] - self.gaps[-1][0]:.0f} s")
                self.idx = n
                yield n, at - self.t0_mono, frame
                continue
            # dropped (closed, or nothing for no_frame_s): open it again, waiting longer each time
            if not self.gaps or self.gaps[-1][1] is not None:
                self.gaps.append([round(time.monotonic() - self.t0_mono, 3), None])
            self.reader.stop()
            self.say(f"Ingen bild från kameran, försöker igen om {min(self.retry_max_s, 2 ** tries):.0f} s")
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
                 every: int = 50, samples: int = SAMPLES, pose: str | None = None, still: Path | None = None,
                 model: str = MODEL, imgsz: int = IMGSZ, motion: bool = False, shared=None):
        """`still`: a live camera's kept still (background.Still), so its empty store survives someone standing in
        the way and the camera starting again; None for a file. `model`, `imgsz`, `motion`: PRESETS. `shared`: the
        models of a process that runs many cameras (fleet.Models); the frames then come with their detections."""
        from ultralytics import YOLO

        W, H = size
        self.roi = tuple(int(v) for v in roi) if roi and tuple(roi) != (0, 0, W, H) else None
        self.model_name, self.imgsz, self.motion = model, int(imgsz), bool(motion)
        if shared is not None:
            self.model, self.check = None, shared.check(model)
            self.trk, self.tracker = None, make_tracker(rate)
        else:
            self.model = YOLO(weights.path(model))
            # its own model for the background: a predict on the tracking model would run through its tracker's callbacks
            self.check = YOLO(weights.path(model))
            self.trk, self.tracker = tracker_cfg(out, rate), None
        self.prev_small, self.people, self.last_run_t = None, False, -1e9
        self.ran = self.skipped = 0
        self.pose = pose  # None: the analysis' default (MODEL, BoT-SORT); else a backends.names() entry
        self.backend = None
        if pose:
            from . import backends
            self.backend = backends.Backend(pose, CONF, IMGSZ, DEVICE, HALF, rate, TRACK_BUFFER_S, weights.path)
        self.describer = looks.Looks(W, self.roi) if look else None
        self.things = things or None
        self.finder = objects.Finder(things, DEVICE, HALF) if things else None
        self.next_find = -1.0
        self.plate = background.Plate((W, H), samples, keep=still is not None)
        self.still = background.Still(still, self.plate.small) if still is not None else None
        self.every, self.samples = max(1, every), samples
        self.analysed = 0
        self.stages = Stages()

    def header(self) -> dict:
        """The core's part of the stream's first line."""
        return {"roi": list(self.roi) if self.roi else None, "model": self.pose or self.model_name, "imgsz": self.imgsz,
                "conf": CONF, "motion": self.motion,
                "tracker": "bytetrack" if self.pose else "botsort", "background": skeleton.BACKGROUND,
                "look": reid.REID_MODEL if self.describer else None,
                "objects": [o["name"] for o in self.things] if self.things else None,
                "objects_model": objects.MODEL if self.things else None, "objects_every_s": objects.EVERY_S}

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]] if self.roi else frame

    def gate(self, frame: np.ndarray, t: float) -> bool:
        """Whether the model runs on this frame. Without `motion`: always. With it: when the picture moved since the
        last frame, when someone was there last time, or every MOTION_EVERY_S; a camera where nothing happens costs
        a small diff and no model."""
        if not self.motion:
            return True
        with self.stages("motion"):
            h, w = frame.shape[:2]
            g = cv2.cvtColor(cv2.resize(frame, (MOTION_W, max(1, round(h * MOTION_W / w))), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (5, 5), 0).astype(np.int16)
            moved = self.prev_small is None or float((np.abs(g - self.prev_small) > MOTION_DIFF).mean()) > MOTION_SHARE
            self.prev_small = g
        return moved or self.people or t - self.last_run_t >= MOTION_EVERY_S

    def step(self, frame: np.ndarray, t: float, det=None, run: bool | None = None) -> tuple:
        """(xyxy, conf, tid, kxy, kc, looks or None, things found or None) for one frame at time t. `det`: its
        detections from a batch (fleet.py), else the core runs its model. `run` False: the gate said no, nobody."""
        s = self.stages
        if run is None:
            run = det is not None or self.gate(frame, t)
        if not run:
            self.analysed += 1
            self.skipped += 1
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros((0, 17, 2)), np.zeros((0, 17)), None, None
        with s("model"):
            if det is not None:
                xyxy, conf, tid, kxy, kc = track(self.tracker, det, self.roi[:2] if self.roi else (0, 0))
            elif self.backend is None:
                xyxy, conf, tid, kxy, kc = infer(self.model, frame, self.roi, self.trk, self.imgsz, CONF)
            else:
                xyxy, conf, tid, kxy, kc = self._other(frame)
        self.ran += 1
        self.people, self.last_run_t = len(xyxy) > 0, t
        found = None
        if self.finder is not None and t >= self.next_find:
            with s("objects"):
                found, self.next_find = self.finder.find(frame, self.roi), t + objects.EVERY_S
        described = None
        if self.describer is not None:
            with s("look"):
                described = self.describer.describe(frame, xyxy, conf, tid, t)
        if (self.ran - 1) % self.every == 0 and self.plate.wants():
            with s("background"), gpu.GPU_LOCK:
                # every person the detector sees, tracked or not yet, is left out of the empty store
                self.plate.add(frame, infer(self.check, frame, self.roi, None, CHECK_IMGSZ, 0.15)[0])
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
        """Time to make the empty store: its samples are in (live: and background.REBUILD new ones since)."""
        return self.plate.due()

    def _check(self, m, img):
        with gpu.GPU_LOCK:
            return infer(m, img, None, None, CHECK_IMGSZ, background.CHECK_CONF)[0]

    def background(self):
        """The empty store (an image or None) and what it says about itself."""
        with self.stages("background"):
            return self.plate.build(self.check, self._check, self.still)

    def background_from_still(self):
        """A live camera's kept still as the empty store before its samples are in, or (None, None)."""
        if self.still is None:
            return None, None
        with self.stages("background"):
            return self.still.picture(self.plate.size, self.plate.k, self.check, self._check)


def run(src: str, out: Path, stride: int = 1, roi=None, samples: int = SAMPLES, look: bool = True, progress=None,
        stop=None, things: list[dict] | None = None, pose: str | None = None, seconds: float | None = None,
        send: str | None = None, send_key: str | None = None, model: str = MODEL, imgsz: int = IMGSZ,
        fps: float = 0.0, motion: bool = False, hwdec: bool = False) -> dict:
    """A video file or a live camera (a URL) through the core into a skeleton file. `progress(frame, total)` now and
    then; `things`: what the store showed a photo of (objects.load), looked for every objects.EVERY_S; None or []:
    nothing. A file: `stop()` true ends it short (no end line: the analysis sees it stopped). A camera has no end of
    its own: `stop()`, `seconds` or Ctrl-C ends it whole, with the empty store and the end line. `send`: the analysis'
    address on another computer; the stream goes there as it is written (send.Sender, with `send_key` or send.key())."""
    progress = progress or (lambda i, n: print(f"  {i}/{n} bildrutor" if n else f"  bildruta {i}", flush=True))
    send_key = (send_key or sending.key()) if send else None  # no key: stop before the camera is opened
    if is_stream(src):
        say(f"Öppnar kameran {redact(src)}")
    source = StreamSource(src, max_fps=fps, hwdec=hwdec) if is_stream(src) else FileSource(src, stride)
    live = getattr(source, "live", False)
    try:
        out.mkdir(parents=True, exist_ok=True)
        every = max(1, source.total // source.stride // samples) if source.total else LIVE_EVERY if live else 50
        if live:
            say(f"Kameran svarar ({source.size[0]}×{source.size[1]}, {source.src_fps:.0f} bilder/s). "
                f"Laddar modellerna på {device_name()}")
        extra = {"pose": pose} if pose else {}
        extra.update(model=model, imgsz=imgsz, motion=motion)
        if live:  # the camera's still, by the stream's name: the same camera started again finds it (background.Still)
            extra["still"] = paths.OWN / "backgrounds" / f"{out.name}.png"
        core = Core(source.size, roi, source.rate, out, look, things, every, samples, **extra)
        head = {"source": source.name, "size": list(source.size), "src_fps": source.src_fps, "rate": source.rate,
                "stride": source.stride, "frames": source.total, **core.header()}
        if live:  # t is seconds since t0 (the wall clock, unix seconds): a camera has no video time of its own
            head.update(live=True, t0=round(source.t0, 3), max_fps=fps or None, hwdec=hwdec)
        w = skeleton.Writer(out, head, flush=live or bool(send))
        sender = sending.Sender(out, send, send_key) if send else None
        if sender:
            sender.start()
        t0 = time.monotonic()
        if live:
            say(f"Kör på {device_name()}")
        bg = bg_info = None
        if live:
            bg, bg_info = _write_background(core, out, core.background_from_still())
        try:
            try:
                stats = Stats(core) if live else None
                for idx, t, frame in source.frames(core.stages):
                    rows = core.step(frame, t)
                    if stats:
                        stats.tick()
                    with core.stages("write"):
                        w.frame(idx, t, *rows)
                    if live and core.background_ready():
                        # the empty store once its samples are in, and again as they roll: the view draws on it
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
            if bg_info is None or not bg_info.get("samples"):  # none yet, or only the still's
                bg, bg_info = _write_background(core, out)
        except BaseException:
            w.abort()  # no end line, also when the empty store failed: the analysis never takes it for whole
            if sender:
                sender.stop()
            raise
    finally:
        source.close()
    info = {"frames": source.idx, "analysed": core.analysed, "model_runs": getattr(core, "ran", core.analysed),
            "wall_s": round(time.monotonic() - t0, 1),
            "background": bg_info, "stage_ms": core.stages.mean_ms(core.analysed)}
    if live:
        info["gaps"] = source.gaps
    w.end(**info)
    if sender:  # what is left goes on, the end line last (DRAIN_S at most)
        sender.stop()
        sender.join(sending.DRAIN_S + 5)
        info["sent"] = sender.status()
    return info


class Stats:
    """A line "~ {json}" on stdout every STATS_EVERY_S: frames a second through the core, through the model, and the
    model's time per frame. The page reads it (server.LiveJob.stats); nothing in it is about the picture."""

    def __init__(self, core, prefix: str = ""):
        self.core, self.prefix = core, prefix
        self.t0, self.a0, self.r0, self.m0 = time.monotonic(), 0, 0, 0.0

    def tick(self) -> None:
        now = time.monotonic()
        if now - self.t0 < STATS_EVERY_S:
            return
        c, dt = self.core, now - self.t0
        model_s = c.stages.total.get("model", 0.0)
        ran = getattr(c, "ran", c.analysed) - self.r0
        line = {"fps": round((c.analysed - self.a0) / dt, 1), "model_fps": round(ran / dt, 1),
                "model_ms": round(1000 * (model_s - self.m0) / ran, 1) if ran else None}
        emit(f"{self.prefix}~ {json.dumps(line)}")
        self.t0, self.a0, self.r0, self.m0 = now, c.analysed, getattr(c, "ran", c.analysed), model_s


def _write_background(core, out: Path, made=None):
    """The empty store from the core's samples (or `made`), written next to the stream (via a temporary file: a
    reader never gets half a picture)."""
    bg, info = made if made is not None else core.background()
    if bg is not None:
        tmp = out / ("." + skeleton.BACKGROUND)
        cv2.imwrite(str(tmp) + ".jpg", bg, [cv2.IMWRITE_JPEG_QUALITY, 90])
        replace_retrying(Path(str(tmp) + ".jpg"), out / skeleton.BACKGROUND)
    return bg, info


def replace_retrying(src: Path, dst: Path, tries: int = 40, wait_s: float = 0.05) -> None:
    """On Windows a file cannot be replaced while another process has it open (the page reading the empty store to
    draw on): wait a moment and try again, rather than end the camera over a picture."""
    for k in range(tries):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(wait_s)


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
    ap.add_argument("--model", default=MODEL, choices=MODELS_OK, help="the person and pose model (PRESETS)")
    ap.add_argument("--imgsz", type=int, default=IMGSZ, choices=IMGSZ_OK, help="the model's picture size")
    ap.add_argument("--fps", type=float, default=0.0, help="a live camera: at most this many frames a second (0: all)")
    ap.add_argument("--motion", action="store_true", help="no model on frames where nothing moved and nobody was there")
    ap.add_argument("--hwdec", action="store_true", help="a live camera: decode its video on the graphics card where FFmpeg can")
    a = ap.parse_args()
    if a.src == "-":  # the camera's address on stdin: with its login it must not be in the process list (server.py)
        a.src = sys.stdin.readline().strip()
        _stop_on_stdin()
    roi = [int(v) for v in a.roi.split(",")] if a.roi else None
    info = run(a.src, Path(a.out), max(1, a.stride), roi, look=not a.no_look, things=None if a.no_objects else objects.load(), pose=a.pose, seconds=a.seconds,
               send=a.send, model=a.model, imgsz=a.imgsz, fps=a.fps, motion=a.motion, hwdec=a.hwdec)
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
