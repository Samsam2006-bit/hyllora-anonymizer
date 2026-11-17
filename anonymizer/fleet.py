"""Many live cameras in one process (the page's "Alla kameror i en process", 2026-11-17): the models are loaded once,
and the newest frame of every camera that has one goes through the model in the same call (a batch), which is where a
graphics card is fast. Twenty processes, one per camera, would each hold torch and the models (1.2-1.8 GB each, more
than the store's computer has) and take turns on the card one frame at a time.

Each camera keeps what main.run gives it alone: its own reader (main.StreamSource, the newest frame, reconnects and
gaps), its own tracker (main.make_tracker, BoT-SORT as model.track makes it), its own stream file, sender, still and
empty store, and its own motion check. Only the model call is shared. The store's things (objects.py) are not looked
for here.

    .venv/bin/python -m anonymizer.fleet <skeletons folder> [--send <analysis>]

Commands come on stdin, one JSON object a line, so no camera's address (with its login) is ever in the process list:
{"cmd": "add", "name": "kassa", "url": "rtsp://…", "opts": {"model", "imgsz", "fps", "look", "motion"}} and
{"cmd": "stop", "name": "kassa"}. Stdin closing stops every camera. Each camera's lines on stdout start with
"@<name> " (server.py routes them): its steps (main.say), "~ {stats}", "!end {info}" and "!error <message>"; the
process' own stats line is "~ {…}" without a name."""

import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import main, paths, send as sending, skeleton

BATCH_MAX = 32  # frames through the model in one call at most
BATCH_WAIT_S = 0.03  # after the first new frame, wait this long for others: cameras send out of step, and one frame
                     # a call (1.1 on average with three cameras, 2026-11-17) leaves a graphics card mostly idle
STATS_EVERY_S = main.STATS_EVERY_S


def out(name: str | None, msg: str) -> None:
    main.emit(f"@{name} {msg}" if name else msg)


class Models:
    """Each model loaded once for every camera. The empty store's check uses the same model: in this process it is
    only ever called with predict, so no tracker's callbacks run on it."""

    def __init__(self):
        self._m, self._lock = {}, threading.Lock()

    def get(self, name: str):
        with self._lock:
            if name not in self._m:
                from ultralytics import YOLO

                self._m[name] = YOLO(main.weights.path(name))
            return self._m[name]

    check = get


class Cam:
    def __init__(self, fleet: "Fleet", name: str, url: str, opts: dict):
        self.fleet, self.name, self.url = fleet, name, url
        self.opts = {**main.PRESETS["full"], **(opts or {})}
        self.folder = fleet.root / f"{name}.live"
        self.slot, self.lock = None, threading.Lock()
        self.source = self.core = self.writer = self.sender = self.stats = None
        self.ready = self.done = self.ending = self.finished = False
        self.bg = self.bg_info = None
        self.bg_pending = False
        self.wanted_stop = threading.Event()
        self.t_start = time.monotonic()
        threading.Thread(target=self._run, daemon=True, name=f"cam-{name}").start()

    def say(self, msg: str) -> None:
        out(self.name, msg)

    def _run(self) -> None:
        """Open the camera and set it up (both may take seconds, so not in the model loop), then read its frames into
        `slot`, always the newest, until it stops."""
        try:
            o = self.opts
            self.say(f"Öppnar kameran {main.redact(self.url)}")
            self.source = main.StreamSource(self.url, max_fps=o["fps"], hwdec=bool(o.get("hwdec")), say=self.say)
            if self.wanted_stop.is_set():
                self.source.stop()
            src = self.source
            self.say(f"Kameran svarar ({src.size[0]}×{src.size[1]}, {src.src_fps:.0f} bilder/s). "
                     f"Laddar modellerna på {main.device_name()}")
            self.folder.mkdir(parents=True, exist_ok=True)
            self.fleet.models.get(o["model"])
            self.core = main.Core(src.size, None, src.rate, self.folder, o["look"], None, main.LIVE_EVERY, main.SAMPLES,
                                  still=paths.OWN / "backgrounds" / f"{self.folder.name}.png", model=o["model"],
                                  imgsz=o["imgsz"], motion=o["motion"], shared=self.fleet.models)
            head = {"source": src.name, "size": list(src.size), "src_fps": src.src_fps, "rate": src.rate,
                    "stride": src.stride, "frames": src.total, **self.core.header(), "live": True,
                    "t0": round(src.t0, 3), "max_fps": o["fps"] or None, "fleet": True,
                    "hwdec": bool(o.get("hwdec"))}
            self.writer = skeleton.Writer(self.folder, head, flush=True)
            if self.fleet.send:
                self.sender = sending.Sender(self.folder, self.fleet.send, sending.key())
                self.sender.start()
            self.bg, self.bg_info = main._write_background(self.core, self.folder, self.core.background_from_still())
            self.stats = main.Stats(self.core, prefix=f"@{self.name} ")
            self.say(f"Kör på {main.device_name()} (samlad process)")
            self.ready = True
            for item in src.frames(self.core.stages):
                with self.lock:
                    self.slot = item
                self.fleet.wake.set()
        except Exception as e:
            self.fail(e)
            return
        self.done = True
        self.fleet.wake.set()

    def take(self):
        with self.lock:
            item, self.slot = self.slot, None
        return item

    def stop(self) -> None:
        self.wanted_stop.set()
        if self.source is not None:
            self.source.stop()

    def fail(self, e: BaseException) -> None:
        if self.finished:
            return
        self.finished = True
        if self.source is not None:
            self.source.close()
        if self.writer is not None:
            self.writer.abort()
        if self.sender is not None:
            self.sender.stop()
        msg = e if isinstance(e, str) else f"{type(e).__name__}: {e}"
        self.say(f"!error {main.redact(msg)}")

    def finish(self) -> None:
        """A stopped camera ends whole: the empty store, the end line, the rest sent (as main.run)."""
        try:
            while self.bg_pending:
                time.sleep(0.05)
            self.source.close()
            if self.bg_info is None or not self.bg_info.get("samples"):
                self.bg, self.bg_info = main._write_background(self.core, self.folder)
            c = self.core
            info = {"frames": self.source.idx, "analysed": c.analysed, "model_runs": c.ran,
                    "wall_s": round(time.monotonic() - self.t_start, 1), "background": self.bg_info,
                    "stage_ms": c.stages.mean_ms(c.analysed), "gaps": self.source.gaps}
            self.writer.end(**info)
            if self.sender is not None:
                self.sender.stop()
                self.sender.join(sending.DRAIN_S + 5)
                info["sent"] = self.sender.status()
            self.finished = True
            self.say(f"!end {json.dumps(info, ensure_ascii=False)}")
        except Exception as e:
            self.fail(e)


class Fleet:
    def __init__(self, root: Path, send: str | None = None):
        self.root, self.send = Path(root), send
        self.models = Models()
        self.cams: dict[str, Cam] = {}
        self.wake = threading.Event()
        self.closing = False
        # apart, so an ending that waits for its camera's empty store never holds the thread that would make it
        self.bgwork = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fleet-bg")
        self.endwork = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fleet-end")
        self.t0, self.batches, self.frames, self.model_s = time.monotonic(), 0, 0, 0.0

    def command(self, line: str) -> None:
        try:
            c = json.loads(line)
        except ValueError:
            return
        name = str(c.get("name") or "")
        if c.get("cmd") == "add" and name and c.get("url"):
            old = self.cams.get(name)
            if old is not None and not old.finished:
                out(name, "!error körs redan")
                return
            self.cams[name] = Cam(self, name, c["url"], c.get("opts") or {})
        elif c.get("cmd") == "stop" and name in self.cams:
            self.cams[name].stop()

    def read_stdin(self) -> None:
        for line in sys.stdin:
            if line.strip():
                self.command(line)
        self.closing = True  # the page's process ended: stop every camera, end them whole
        for cam in list(self.cams.values()):
            cam.stop()
        self.wake.set()

    def _background(self, cam: Cam) -> None:
        try:
            cam.bg, cam.bg_info = main._write_background(cam.core, cam.folder)
        except Exception:
            traceback.print_exc()
        finally:
            cam.bg_pending = False

    def step(self) -> None:
        """Every camera with a new frame: its motion check, then the model once per (model, size) for all of them,
        then each camera's tracker, stream line and empty store."""
        items = []
        for cam in list(self.cams.values()):
            if cam.ready and not cam.ending and not cam.finished:
                item = cam.take()
                if item is not None:
                    idx, t, frame = item
                    items.append((cam, idx, t, frame, cam.core.gate(frame, t)))
            if cam.done and not cam.ending and not cam.finished:
                cam.ending = True  # no frame comes any more: end it off this loop (finish waits for its empty store)
                self.endwork.submit(cam.finish)
        groups: dict[tuple, list] = {}
        for it in items:
            if it[4]:
                c = it[0].core
                groups.setdefault((c.model_name, c.imgsz), []).append(it)
        dets = {}
        for (model, imgsz), its in groups.items():
            for k in range(0, len(its), BATCH_MAX):
                part = its[k:k + BATCH_MAX]
                t0 = time.perf_counter()
                rs = main.detect(self.models.get(model), [it[0].core.crop(it[3]) for it in part], imgsz, main.CONF)
                dt = time.perf_counter() - t0
                self.model_s += dt
                for it in part:  # each camera's share of the call, so its model time compares with a camera alone
                    it[0].core.stages.add("model", dt / len(part))
                self.batches += 1
                self.frames += len(part)
                for it, r in zip(part, rs):
                    dets[id(it)] = r
        for it in items:
            cam, idx, t, frame, run = it
            try:
                rows = cam.core.step(frame, t, det=dets.get(id(it)), run=run)
                cam.writer.frame(idx, t, *rows)
                cam.stats.tick()
                if cam.core.background_ready() and not cam.bg_pending:
                    cam.bg_pending = True
                    self.bgwork.submit(self._background, cam)
            except Exception as e:
                traceback.print_exc()
                cam.stop()
                cam.fail(e)

    def stats(self) -> None:
        now = time.monotonic()
        if now - self.t0 < STATS_EVERY_S:
            return
        dt = now - self.t0
        line = {"cameras": sum(1 for c in self.cams.values() if not c.finished), "frames_per_s": round(self.frames / dt, 1),
                "batch": round(self.frames / self.batches, 1) if self.batches else None,
                "batch_ms": round(1000 * self.model_s / self.batches, 1) if self.batches else None,
                "busy": round(self.model_s / dt, 2)}
        if main.DEVICE == "cuda":
            line["gpu_mb"] = round(main.torch.cuda.memory_reserved(0) / 2 ** 20)
        out(None, f"~ {json.dumps(line)}")
        self.t0, self.batches, self.frames, self.model_s = now, 0, 0, 0.0

    def run(self, stdin: bool = True) -> None:
        if stdin:
            threading.Thread(target=self.read_stdin, daemon=True, name="stdin").start()
        out(None, f"Samlad process igång på {main.device_name()}")
        while True:
            if self.wake.wait(0.05) and len(self.cams) > 1:
                time.sleep(BATCH_WAIT_S)
            self.wake.clear()
            self.step()
            self.stats()
            if self.closing and all(c.finished for c in self.cams.values()):
                break
        self.bgwork.shutdown(wait=True)
        self.endwork.shutdown(wait=True)


def run_main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="the skeletons folder: each camera writes <root>/<name>.live")
    ap.add_argument("--send", help="the analysis' address on another computer (send.py)")
    a = ap.parse_args()
    Fleet(Path(a.root), a.send).run()


if __name__ == "__main__":
    run_main()
