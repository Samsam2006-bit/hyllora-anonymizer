"""A picture of the store with no one in it, for calibration and for drawing the people over (skeleton.py).

The median of frames spread over the video, each with its people's boxes left out. A pixel covered by a person in
every frame (someone standing still the whole time) gets no value and is painted grey. Then the detector looks at
the result once more, and anything it still takes for a person is painted grey too: rather a grey patch than a
person in the picture that leaves the anonymizer (fail closed).
"""

import warnings

import cv2
import numpy as np

MAX_W = 1920  # samples kept at most this wide (memory: 30 4K frames would be ~750 MB)
PAD = 0.12  # a person's box grown by this share on each side (arms, shadow, the tracker's box lagging)
GREY = 128
CHECK_CONF = 0.2  # the final check is more suspicious than the analysis (conf 0.3)


class Plate:
    def __init__(self, size, n: int = 30):
        W, H = size
        self.size, self.n = (W, H), n
        self.k = min(1.0, MAX_W / W)
        self.small = (max(1, round(W * self.k)), max(1, round(H * self.k)))
        self.frames: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []

    def add(self, frame: np.ndarray, boxes) -> None:
        img = cv2.resize(frame, self.small, interpolation=cv2.INTER_AREA) if self.k < 1 else frame.copy()
        mask = np.zeros(img.shape[:2], bool)
        for x0, y0, x1, y1 in np.asarray(boxes, dtype=np.float64).reshape(-1, 4) * self.k:
            px, py = PAD * (x1 - x0), PAD * (y1 - y0)
            mask[max(0, int(y0 - py)):int(y1 + py) + 1, max(0, int(x0 - px)):int(x1 + px) + 1] = True
        self.frames.append(img)
        self.masks.append(mask)

    def build(self, model=None, infer=None) -> tuple[np.ndarray | None, dict]:
        """The empty store at full size, and how it went: share of pixels never free of people, and how many
        people the final check still found. `infer(model, img)` -> person boxes, for that check."""
        if not self.frames:
            return None, {"samples": 0}
        h, w = self.frames[0].shape[:2]
        out = np.full((h, w, 3), GREY, np.uint8)
        never = np.zeros((h, w), bool)
        band = max(1, 4_000_000 // (w * 3 * len(self.frames)))  # rows per step, ~16 MB of float32 at a time
        for y in range(0, h, band):
            stack = np.stack([f[y:y + band] for f in self.frames]).astype(np.float32)
            m = np.stack([mk[y:y + band] for mk in self.masks])
            stack[m] = np.nan
            with warnings.catch_warnings():  # "All-NaN slice": a pixel never free of people, handled below
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(stack, axis=0)
            hole = np.isnan(med).any(axis=2)
            med[hole] = GREY
            out[y:y + band] = np.clip(med, 0, 255).astype(np.uint8)
            never[y:y + band] = hole
        if self.k < 1:
            out = cv2.resize(out, self.size, interpolation=cv2.INTER_LINEAR)
        left = 0
        if model is not None and infer is not None:
            for x0, y0, x1, y1 in infer(model, out):
                px, py = PAD * (x1 - x0), PAD * (y1 - y0)
                out[max(0, int(y0 - py)):int(y1 + py) + 1, max(0, int(x0 - px)):int(x1 + px) + 1] = GREY
                left += 1
        return out, {"samples": len(self.frames), "never_free": round(float(never.mean()), 4), "greyed_after_check": left}
