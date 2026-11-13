"""What a person looks like, to know them again after they were out of sight (Runtime._reid, identity.py).

The tracker keeps an ID while it can follow a person from frame to frame. Someone who walks behind a pillar, out
of the picture and back, or stands hidden behind others for longer than the tracker's buffer comes back with a new
ID, and without this would be counted as a new visitor.

The description is a vector of the whole body's appearance from a general image model, Meta's Perception Encoder
(PE-core base, Apache 2.0, REID_MODEL), on the person's box letterboxed to a tall picture. Until 2026-10-01 it was
the colours of the clothes, which could not tell apart two people in black t-shirts and jeans (on "larm" it was
wrong more often than right, ARCHITECTURE §6). The vector is anonymous: no name, no face template, and it lives only
in memory during the run (it is not written to the report or to disk).

Averaged over a track's frames (`Look`), two people are compared with the cosine distance of their mean vectors.
"""

import threading

import cv2
import numpy as np

from . import weights
from .gpu import GPU_LOCK

REID_MODEL = "vit_pe_core_base_patch16_224.fb"  # chosen 2026-10-01 among models with a commercial licence (WORKLOG)
REID_SIZE = (256, 128)  # (h, w) the model sees: tall, like a person

_model = None
_load_lock = threading.Lock()


def _load():
    """The model, loaded once on first use from models/ (weights.py)."""
    global _model
    with _load_lock:
        if _model is None:
            import timm
            import torch

            dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            m = timm.create_model(REID_MODEL, pretrained=True, num_classes=0, img_size=REID_SIZE,
                                  pretrained_cfg_overlay={"file": weights.path(REID_MODEL + ".safetensors")})
            m = m.eval().to(dev)
            if dev != "cpu":  # 16-bit on the GPU: 8 % faster, the same vectors (cosine 0.9999, 2026-10-02)
                m = m.half()
            cfg = timm.data.resolve_data_config({}, model=m)
            _model = (m, dev, np.array(cfg["mean"], np.float32), np.array(cfg["std"], np.float32))
    return _model


def _letterbox(crop: np.ndarray) -> np.ndarray:
    """The crop scaled to fit REID_SIZE (h, w), centred on grey: a person keeps their proportions."""
    h, w = REID_SIZE
    s = min(h / crop.shape[0], w / crop.shape[1])
    r = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))))
    out = np.full((h, w, 3), 124, np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def describe_many(frame: np.ndarray, boxes) -> list[dict]:
    """{"body": unit vector} per box in one frame (BGR, unpixelated), in one model call; {} for a box too small."""
    H, W = frame.shape[:2]
    crops, idx = [], []
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 >= 8 and y1 - y0 >= 16:
            crops.append(_letterbox(frame[y0:y1, x0:x1])[:, :, ::-1])
            idx.append(i)
    out: list[dict] = [{} for _ in boxes]
    if not crops:
        return out
    import torch

    m, dev, mean, std = _load()
    x = (np.stack(crops).astype(np.float32) / 255.0 - mean) / std
    with GPU_LOCK, torch.no_grad():  # gpu.GPU_LOCK: MPS crashes on concurrent inference
        e = m(torch.from_numpy(x.transpose(0, 3, 1, 2)).to(dev).to(next(m.parameters()).dtype)).float().cpu().numpy()
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    for i, v in zip(idx, e):
        out[i] = {"body": v}
    return out


def distance(a: dict, b: dict) -> float | None:
    """Cosine distance (0 the same, 1 unrelated) between two descriptions, or None when one has no body vector."""
    if "body" not in a or "body" not in b:
        return None
    va, vb = a["body"], b["body"]
    return float(1.0 - va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


class Look:
    """A person's appearance, averaged over the frames where they were seen whole and alone."""

    def __init__(self):
        self.sum: dict[str, np.ndarray] = {}
        self.n: dict[str, int] = {}

    def add(self, desc: dict) -> None:
        for k, h in desc.items():
            self.sum[k] = self.sum.get(k, 0) + h
            self.n[k] = self.n.get(k, 0) + 1

    def merge(self, other: "Look") -> None:
        for k, s in other.sum.items():
            self.sum[k] = self.sum.get(k, 0) + s
            self.n[k] = self.n.get(k, 0) + other.n[k]

    @property
    def frames(self) -> int:
        return max(self.n.values(), default=0)

    def mean(self) -> dict[str, np.ndarray]:
        return {k: s / self.n[k] for k, s in self.sum.items()}
