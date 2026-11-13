"""The body vector (inferero/reid.py) worked out in the anonymizer, where the picture is, and passed on with the
skeleton (decision 2026-10-13: with it "larm" solves to 4 people, without it 5). It is not a picture, but it
describes how a body looks: it stays on the box and is never written by the analysis (reid.py).

When to describe a person, as CameraWorker._looks decided it before (repeated here: the anonymizer never imports the
pipeline): someone seen whole and alone, every REID_EVERY_S while new, near someone else or at the picture's edge,
otherwise every REID_IDLE_S. The analysis' own skip (someone this camera leaves to another, Runtime.coverage) is
applied on the analysis' side, as it needs the floor plan.
"""

import numpy as np

from inferero import reid

# = inferero.pipeline.config (REID_*)
REID_WINDOW_S = 5.0
REID_EVERY_S = 1.0
REID_IDLE_S = 2.0
REID_EDGE = 0.05
REID_IOU = 0.2
REID_MIN_H = 80


def _iou(a: np.ndarray) -> np.ndarray:
    x0 = np.maximum(a[:, None, 0], a[None, :, 0])
    y0 = np.maximum(a[:, None, 1], a[None, :, 1])
    x1 = np.minimum(a[:, None, 2], a[None, :, 2])
    y1 = np.minimum(a[:, None, 3], a[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    return inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)


class Looks:
    def __init__(self, width: int, roi=None):
        self.x0r, self.x1r = (roi[0], roi[2]) if roi else (0, width)
        self.first: dict[int, float] = {}
        self.last: dict[int, float] = {}

    def describe(self, frame: np.ndarray, xyxy, conf, tid, t: float) -> dict[int, np.ndarray]:
        """{row: unit body vector} for the people due a description in this frame."""
        n = len(xyxy)
        if not n:
            return {}
        iou = _iou(np.asarray(xyxy, dtype=np.float64))
        np.fill_diagonal(iou, 0)
        edge = REID_EDGE * (self.x1r - self.x0r)
        foot = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]])
        todo = []
        for i in range(n):
            x0, y0, x1, y1 = xyxy[i]
            k = int(tid[i])
            first = self.first.setdefault(k, t)
            if y1 - y0 < REID_MIN_H or conf[i] < 0.5 or iou[i].max(initial=0) >= REID_IOU:
                continue
            others = np.delete(foot, i, axis=0)
            busy = (t - first <= REID_WINDOW_S
                    or (len(others) and np.hypot(*(others - foot[i]).T).min() < y1 - y0)
                    or x0 - self.x0r < edge or self.x1r - x1 < edge)
            if t - self.last.get(k, -1e9) >= (REID_EVERY_S if busy else REID_IDLE_S):
                todo.append(i)
        out = {}
        if todo:
            for i, desc in zip(todo, reid.describe_many(frame, xyxy[todo])):
                if desc:
                    out[i] = desc["body"]
                    self.last[int(tid[i])] = t
        return out
