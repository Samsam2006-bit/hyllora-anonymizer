"""Test cameras: the uploaded videos played as live cameras (MJPEG over HTTP), each from its start in real time, so a
live run can be tried without a camera (as scripts/fake_cameras.py). A process of its own, started by the anonymizer's
page when a test camera is chosen (server.py): opening a camera holds the anonymizer's process while it waits for the
first frame, so a test camera served from inside it never sent one (2026-11-11).

    .venv/bin/python -m anonymizer.testcam [port]     (only 127.0.0.1; http://127.0.0.1:<port>/<upload's file name>)
"""

import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from anonymizer import paths  # noqa: E402  (run as python -m anonymizer.testcam)

UPLOADS = paths.UPLOADS
PORT = 8432


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        name = self.path.lstrip("/")
        p = UPLOADS / name
        if not name or Path(name).name != name or name.startswith(".") or not p.is_file():
            self.send_error(404)
            return
        cap = cv2.VideoCapture(str(p))
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        t0 = time.monotonic()
        try:
            while True:
                vt, new = time.monotonic() - t0, False
                while cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 < vt:
                    if not cap.grab():
                        return
                    new = True
                if not new:
                    time.sleep(0.005)
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    return
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(buf) + buf.tobytes() + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            cap.release()


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else PORT), Handler).serve_forever()
