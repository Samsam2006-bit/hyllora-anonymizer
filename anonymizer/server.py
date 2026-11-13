"""The anonymizer's own page, a program of its own beside the analysis (CURRENT_STATE J): on http://127.0.0.1:8430,
never on the analysis' 8420. It is the only one allowed to show the camera's picture, so the original is shown here
and nowhere in the analysis app.

    .venv/bin/uvicorn anonymizer.server:app --host 127.0.0.1 --port 8430     (or scripts/anonymizer.sh install)

Only from this computer: a request from another address is refused, whatever host it was started on (there is no
login here yet). In the test phase the uploaded videos (data/uploads) stand in for the cameras; out go the skeleton
streams in data/skeletons, which the analysis reads as cameras `skeletons/<name>`.
"""

import base64
import json
import re
import secrets
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import objects, paths, skeleton, view

ROOT = paths.ROOT
DATA = paths.DATA
UPLOADS = paths.UPLOADS  # stands in for the cameras in the test phase
SKELETONS = paths.SKELETONS
OBJECTS = paths.OBJECTS  # the store's photos of what to look for (objects.py): stay here, never sent on
MAX_PHOTO = 12 * 1024 * 1024
HERE = Path(__file__).resolve().parent
BRAND = ROOT / "static"  # Hyllora's look, shared with the analysis app: only these files are served from it
BRAND_FILES = {"tokens.css", "style.css", "hyllora.svg"}
VIDEO = (".mov", ".mp4", ".m4v", ".avi", ".mkv")
LOCAL = {"127.0.0.1", "::1", "localhost", "testclient"}

app = FastAPI(title="Hyllora anonymiserare", docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def only_this_computer(request: Request, call_next):
    if (request.client.host if request.client else "") not in LOCAL:
        return JSONResponse({"detail": "Anonymiseraren nås bara från den här datorn"}, status_code=403)
    return await call_next(request)


def _plain(name: str) -> bool:
    return bool(name) and Path(name).name == name and not name.startswith(".") and "\\" not in name


@app.get("/")
def page():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/static/{name}")
def brand(name: str):
    if name not in BRAND_FILES:
        raise HTTPException(404)
    return FileResponse(BRAND / name)


app.mount("/static/vendor/fonts", StaticFiles(directory=BRAND / "vendor" / "fonts"), name="fonts")


@app.get("/api/sources")
def sources():
    files = sorted((p for p in UPLOADS.glob("*") if p.suffix.lower() in VIDEO), key=lambda p: -p.stat().st_mtime) \
        if UPLOADS.is_dir() else []
    return [{"file": p.name, "name": re.sub(r"^[0-9a-z]{8}_", "", p.name), "size": p.stat().st_size} for p in files]


@app.post("/api/sources")
async def upload_source(file: UploadFile = File(...)):
    """A video uploaded on this page: on a computer with only the anonymizer (packaging/) there is no analysis app
    to upload it in. Named as the analysis names uploads (an 8-character id, then the file's own name)."""
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename or "video.mp4").name).lstrip(".") or "video.mp4"
    if Path(stem).suffix.lower() not in VIDEO:
        raise HTTPException(400, "Bara video: " + ", ".join(VIDEO))
    UPLOADS.mkdir(parents=True, exist_ok=True)
    name = f"{secrets.token_hex(4)}_{stem}"
    part = UPLOADS / ("." + name + ".part")  # a broken upload is never listed as a video
    try:
        with open(part, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        part.replace(UPLOADS / name)
    finally:
        part.unlink(missing_ok=True)
    return {"file": name, "name": stem}


@app.get("/api/sources/{name}")
def source_video(name: str):
    """The original, for the side by side view: only the anonymizer shows it."""
    p = UPLOADS / name
    if not _plain(name) or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="video/mp4" if p.suffix.lower() in (".mov", ".mp4", ".m4v") else None)


@app.get("/api/outputs")
def outputs():
    out = []
    for p in sorted(SKELETONS.glob("*/" + skeleton.FILE)) if SKELETONS.is_dir() else []:
        try:
            h = skeleton.header(p.parent)
        except (OSError, ValueError):
            continue
        with open(p, "rb") as f:  # the end line says the whole video went through (it stopped short otherwise)
            f.seek(max(0, p.stat().st_size - 4096))
            ended = b'"end":true' in f.read()
        src = h.get("source")
        out.append({"name": p.parent.name, "camera_source": f"skeletons/{p.parent.name}", "ended": ended,
                    "frames": h.get("frames"), "rate": h.get("rate"), "look": bool(h.get("look")), "size": h.get("size"),
                    "running": (JOB.running and JOB.out == p.parent.name) or p.parent.name in _live_running(),
                    "live": bool(h.get("live")),
                    "original": f"/api/sources/{quote(src)}" if src and _plain(src) and (UPLOADS / src).is_file() else None})
    return out


@app.get("/api/outputs/{name}/{file}")
def output_file(name: str, file: str):
    folder = SKELETONS / name
    if not _plain(name) or file not in (skeleton.FILE, skeleton.BACKGROUND) or not (folder / file).is_file():
        raise HTTPException(404)
    return FileResponse(folder / file, media_type="image/jpeg" if file.endswith(".jpg") else "application/x-ndjson",
                        headers={"Cache-Control": "no-store"})


class Job:
    """One anonymization at a time (the GPU is shared with the analysis)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running, self.stopping = False, False
        self.out = self.source = self.error = None
        self.frame, self.total, self.started, self.info = 0, 0, 0.0, None

    def status(self) -> dict:
        return {"running": self.running, "source": self.source, "out": self.out, "error": self.error, "info": self.info,
                "sends": bool(_ingest_url()),
                "progress": round(self.frame / self.total, 3) if self.total else 0.0,
                "elapsed_s": round(time.monotonic() - self.started, 1) if self.started else 0.0}

    def start(self, source: str, stride: int, look: bool, things: list[dict] | None = None) -> dict:
        with self.lock:
            if self.running:
                raise HTTPException(409, "En anonymisering körs redan")
            if not _plain(source) or not (UPLOADS / source).is_file():
                raise HTTPException(404, "Hittar inte videon")
            out = f"{Path(source).stem}.s{stride}"
            self.running, self.stopping, self.error, self.info = True, False, None, None
            self.source, self.out, self.frame, self.total, self.started = source, out, 0, 0, time.monotonic()
        threading.Thread(target=self._run, args=(source, out, stride, look, things), daemon=True, name="anonymize").start()
        return self.status()

    def _run(self, source, out, stride, look, things):
        from . import main  # loads torch and the models: only when a job is started

        def progress(i, n):
            self.frame, self.total = i, n

        try:
            self.info = main.run(str(UPLOADS / source), SKELETONS / out, stride, look=look, progress=progress,
                                 stop=lambda: self.stopping, things=things, send=_ingest_url() or None)
            self.frame = self.total
        except InterruptedError:
            self.error = "Stoppad: strömmen saknar slutrad, så analysen räknar den inte som hel"
        except Exception as e:
            traceback.print_exc()
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.running = False


JOB = Job()


def _ingest_url() -> str:
    """The analysis on another computer (send.py, J3 (c)): set, a video's stream and each camera's go there."""
    import os as _os

    return _os.environ.get("HYLLORA_INGEST_URL", "").strip()


class RunIn(BaseModel):
    source: str
    stride: int = 1
    look: bool = True
    objects: bool = True  # look for what data/objects has photos of


@app.post("/api/run")
def run(body: RunIn):
    return JOB.start(body.source, max(1, min(10, body.stride)), body.look,
                     objects.load(OBJECTS) if body.objects else None)


# what to look for (objects.py): the store uploads a photo and draws a box round the thing in it
def _all_objects() -> list[dict]:
    try:
        return json.loads((OBJECTS / objects.FILE).read_text()).get("objects", [])
    except (OSError, ValueError):
        return []


@app.get("/api/objects")
def list_objects():
    return _all_objects()


@app.get("/api/objects/photo/{file}")
def object_photo(file: str):
    p = OBJECTS / file
    if not _plain(file) or p.suffix != ".jpg" or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


class PhotoIn(BaseModel):
    name: str
    image: str  # data URL or base64 of a JPEG or PNG
    box: list[float]  # x0, y0, x1, y1 in the photo's own pixels


@app.post("/api/objects")
def add_object(body: PhotoIn):
    """A photo of a thing, added to the thing of that name (a new one if there is none): several photos of one
    basket, from different sides, make a steadier class."""
    import cv2
    import numpy as np

    name = body.name.strip()[:40]
    raw = body.image.split(",", 1)[-1]
    if not name or len(raw) > MAX_PHOTO * 4 // 3 + 4:
        raise HTTPException(400, "Namn och en bild under 12 MB behövs")
    try:
        img = cv2.imdecode(np.frombuffer(base64.b64decode(raw), np.uint8), cv2.IMREAD_COLOR)
    except ValueError:
        img = None
    if img is None:
        raise HTTPException(400, "Kunde inte läsa bilden")
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in (body.box + [0, 0, 0, 0])[:4])
    x0, x1 = sorted((max(0.0, min(w, x0)), max(0.0, min(w, x1))))
    y0, y1 = sorted((max(0.0, min(h, y0)), max(0.0, min(h, y1))))
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise HTTPException(400, "Dra en ruta runt saken i bilden")
    OBJECTS.mkdir(parents=True, exist_ok=True)
    file = f"{secrets.token_hex(4)}.jpg"
    cv2.imwrite(str(OBJECTS / file), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    objs = _all_objects()
    o = next((o for o in objs if o["name"].lower() == name.lower()), None)
    if o is None:
        o = {"id": secrets.token_hex(4), "name": name, "refs": []}
        objs.append(o)
    o["refs"].append({"image": file, "box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]})
    objects.save(objs, OBJECTS)
    return o


@app.delete("/api/objects/{oid}/{file}")
def remove_photo(oid: str, file: str):
    """One photo off a thing; the thing goes with its last photo."""
    objs = _all_objects()
    o = next((o for o in objs if o["id"] == oid), None)
    if o is None or not any(r["image"] == file for r in o["refs"]):
        raise HTTPException(404)
    o["refs"] = [r for r in o["refs"] if r["image"] != file]
    if _plain(file):
        (OBJECTS / file).unlink(missing_ok=True)
    objects.save([q for q in objs if q["refs"]], OBJECTS)
    return {"ok": True}


@app.post("/api/stop")
def stop():
    JOB.stopping = True
    return JOB.status()


@app.get("/api/job")
def job():
    return JOB.status()


# ---------- live cameras (J3 (b), 2026-11-11): several at once, each into skeletons/<name>.live ----------
# Each camera is a process of its own (anonymizer.main): three cameras as threads of this process stuck loading
# their models, and opening a camera holds the process it is opened in. The camera's URL (login included) goes to
# it on stdin and is held in memory only, never written, listed or put in the process list: a list shows it redacted.
LIVE_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://")


def live_cmd(out: Path, look: bool, things: bool) -> list[str]:
    """With HYLLORA_INGEST_URL set (the analysis on another computer), each camera's stream is sent there as it grows
    (send.py, J3 (c)); the key is the box's (send.key)."""
    import sys as _sys

    to = _ingest_url()
    return [_sys.executable, "-m", "anonymizer.main", "-", str(out)] + ([] if look else ["--no-look"]) \
        + ([] if things else ["--no-objects"]) + (["--send", to] if to else [])


class LiveJob:
    def __init__(self, name: str, url: str):
        self.name, self.url, self.out = name, url, f"{name}.live"
        self.running, self.error, self.info, self.proc = True, None, None, None
        self.started = time.monotonic()

    def frame(self) -> int:
        """The camera's frame number in the stream's last line (the stream is the only thing the process writes)."""
        p = SKELETONS / self.out / skeleton.FILE
        try:
            with open(p, "rb") as f:
                f.seek(max(0, p.stat().st_size - 65536))
                for line in reversed(f.read().splitlines()):
                    m = re.match(rb'\{"i":(\d+)', line)
                    if m:
                        return int(m.group(1))
        except OSError:
            pass
        return 0

    def status(self) -> dict:
        return {"name": self.name, "out": self.out, "source": skeleton.redact(self.url), "running": self.running,
                "error": self.error, "info": self.info, "frame": self.frame(),
                "elapsed_s": round(time.monotonic() - self.started, 1)}

    def stop(self) -> None:
        # a camera stopped is a whole stream: background and end line. Said on its stdin, not with a signal: a
        # Windows service has no console to send Ctrl-C or Ctrl-Break through (packaging/windows)
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write("stop\n")
                self.proc.stdin.flush()
            except (OSError, ValueError):
                pass


LIVE: dict[str, LiveJob] = {}
LIVE_LOCK = threading.Lock()


def _live_running() -> set[str]:
    return {j.out for j in LIVE.values() if j.running}


class LiveIn(BaseModel):
    name: str
    url: str
    look: bool = True
    objects: bool = False


def _run_live(job: LiveJob, look: bool, things: bool):
    import subprocess

    try:
        proc = subprocess.Popen(live_cmd(SKELETONS / job.out, look, things), cwd=ROOT, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        proc.stdin.write(job.url + "\n")  # stdin stays open: "stop" ends the camera, and so does this process ending
        proc.stdin.flush()
        job.proc = proc
        got = {}
        readers = [threading.Thread(target=lambda k, s: got.__setitem__(k, s.read()), args=(k, s), daemon=True)
                   for k, s in (("out", proc.stdout), ("err", proc.stderr))]
        for t in readers:
            t.start()
        proc.wait()
        for t in readers:
            t.join()
        out, err = got.get("out", ""), got.get("err", "")
        lines = [ln for ln in out.splitlines() if ln.startswith("{")]
        if job.proc.returncode == 0 and lines:
            job.info = json.loads(lines[-1])
        else:
            last = [ln for ln in err.splitlines() if ln.strip()][-1:] or [f"avslutades med {job.proc.returncode}"]
            job.error = skeleton.redact(last[0])[:300]  # OpenCV and FFmpeg may name the URL: never its login
    except Exception as e:
        job.error = skeleton.redact(f"{type(e).__name__}: {e}")
    finally:
        job.running = False


@app.post("/api/live")
def live_start(body: LiveIn):
    name = re.sub(r"[^0-9A-Za-z_-]", "", body.name)[:40]
    url = body.url.strip()
    if url.startswith("test:"):
        file = url[len("test:"):]
        if not _plain(file) or not (UPLOADS / file).is_file():
            raise HTTPException(404, "Hittar inte videon")
        url = _testcam_url(file)
    if not name or not url.lower().startswith(LIVE_SCHEMES):
        raise HTTPException(400, "Ett namn (bokstäver, siffror, - och _) och en kameraadress (rtsp:// eller http://) behövs")
    with LIVE_LOCK:
        if name in LIVE and LIVE[name].running:
            raise HTTPException(409, f"{name} körs redan")
        job = LIVE[name] = LiveJob(name, url)
    threading.Thread(target=_run_live, args=(job, body.look, body.objects), daemon=True, name=f"live-{name}").start()
    return job.status()


@app.get("/api/live")
def live_list():
    return [j.status() for j in LIVE.values()]


@app.post("/api/live/{name}/stop")
def live_stop(name: str):
    job = LIVE.get(name)
    if job is None:
        raise HTTPException(404)
    job.stop()
    return job.status()


@app.get("/api/live/{name}/view.mjpg")
def live_view(name: str):
    """The anonymized video as it is made, on this computer (the link is viewer.py)."""
    folder = SKELETONS / f"{name}.live"
    if not _plain(name) or not view.is_live(folder):
        raise HTTPException(404)
    return StreamingResponse(view.mjpeg(folder, width=view.LOCAL_WIDTH, quality=view.LOCAL_QUALITY), media_type=f"multipart/x-mixed-replace; boundary={view.BOUNDARY}",
                             headers={"Cache-Control": "no-store"})


@app.get("/api/link")
def link():
    """Where the link's server (viewer.py) shows the live cameras; the address outside comes from the tunnel."""
    from . import viewer

    return {"local": f"http://127.0.0.1:{viewer.PORT}/v/{viewer.token()}/", "port": viewer.PORT}


# a test camera ("test:<upload>"): an uploaded video played as a live camera by testcam.py, a process of its own
TESTCAM: dict = {}


def _testcam_url(name: str) -> str:
    import socket
    import subprocess
    import sys as _sys

    from . import testcam

    with socket.socket() as so:
        up = so.connect_ex(("127.0.0.1", testcam.PORT)) == 0
    if not up:
        TESTCAM["proc"] = subprocess.Popen([_sys.executable, "-m", "anonymizer.testcam", str(testcam.PORT)], cwd=ROOT,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(50):
            time.sleep(0.1)
            with socket.socket() as so:
                if so.connect_ex(("127.0.0.1", testcam.PORT)) == 0:
                    break
    return f"http://127.0.0.1:{testcam.PORT}/{quote(name)}"



# "Testa kameran": which address gives a camera's video (probe.py), only for cameras on the local network
class ProbeIn(BaseModel):
    host: str
    user: str = ""
    password: str = ""


@app.post("/api/probe")
def probe_camera(body: ProbeIn):
    from . import probe

    try:
        return probe.probe(body.host, body.user.strip(), body.password)
    except probe.NotLocal as e:
        raise HTTPException(400, str(e)) from None
