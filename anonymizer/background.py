"""A picture of the store with no one in it, for calibration and for drawing the people over (skeleton.py).

The median of frames spread over the video, each with its people's boxes left out. Then the detector looks at the
result once more, and anything it still takes for a person is painted grey: rather a grey patch than a person in the
picture that leaves the anonymizer (fail closed).

**A live camera keeps a still of its own (Still, 2026-11-17).** Sameh: "står någon i vägen blir det vitt bakom … en
stillbild utgås från så länge kameran inte flyttas". Every pixel the camera has once seen free of people is kept in
data/anonymizer/backgrounds/<camera>.png (alpha: seen or not), and used where someone stands now: the empty store is
there from the camera's first second, and someone standing still no longer leaves a hole. The samples roll (the last
`n`), and the picture is made again every REBUILD new samples, so light that changes over the day follows. When the
camera's view has changed (MOVED_SHARE of the pixels both saw differ) twice in a row, it has been moved: the still is
dropped and learnt again. Once is not enough, so a passing change does not throw away a good still.

What no sample and no still has seen free is filled from around it (inpainting, FILL_W wide), not painted grey: it
was a white patch where someone stood. The filling is made from pixels next to the hole, which were free of people.
"""

import collections
import os
import warnings
from pathlib import Path

import cv2
import numpy as np

MAX_W = 1920  # samples kept at most this wide (memory: 30 4K frames would be ~750 MB)
PAD = 0.12  # a person's box grown by this share on each side (arms, shadow, the tracker's box lagging)
GREY = 128
CHECK_CONF = 0.2  # the final check is more suspicious than the analysis (conf 0.3)
REBUILD = 30  # a live camera: the empty store made again every this many new samples
FILL_W = 480  # holes filled from around them at this width (cv2.inpaint is slow on big holes at full size)
MOVED_DIFF = 40  # grey levels apart (blurred) for a pixel to have changed
MOVED_SHARE = 0.35  # of the pixels both the still and now saw free: more changed than this, the camera moved
MOVED_OVERLAP = 0.2  # compared only when at least this share of the picture was seen free by both


class Plate:
    """`keep`: a live camera's rolling samples (the last `n`), made again every REBUILD; else the first `n` only."""

    def __init__(self, size, n: int = 30, keep: bool = False):
        W, H = size
        self.size, self.n, self.keep = (W, H), n, keep
        self.k = min(1.0, MAX_W / W)
        self.small = (max(1, round(W * self.k)), max(1, round(H * self.k)))
        self.frames = collections.deque(maxlen=n) if keep else []
        self.masks = collections.deque(maxlen=n) if keep else []
        self.added, self.built_at = 0, None

    def wants(self) -> bool:
        return self.keep or len(self.frames) < self.n

    def due(self) -> bool:
        """Time to make the empty store: its samples are in, and (live) REBUILD new ones since the last time."""
        return len(self.frames) >= self.n and (self.built_at is None or (self.keep and self.added - self.built_at >= REBUILD))

    def add(self, frame: np.ndarray, boxes) -> None:
        img = cv2.resize(frame, self.small, interpolation=cv2.INTER_AREA) if self.k < 1 else frame.copy()
        self.frames.append(img)
        self.masks.append(self._mask(img.shape[:2], boxes))
        self.added += 1

    def _mask(self, shape, boxes) -> np.ndarray:
        mask = np.zeros(shape, bool)
        for x0, y0, x1, y1 in np.asarray(boxes, dtype=np.float64).reshape(-1, 4) * self.k:
            px, py = PAD * (x1 - x0), PAD * (y1 - y0)
            mask[max(0, int(y0 - py)):int(y1 + py) + 1, max(0, int(x0 - px)):int(x1 + px) + 1] = True
        return mask

    def median(self) -> tuple[np.ndarray, np.ndarray]:
        """(the samples' median at the small size, pixels never free of people)."""
        h, w = self.frames[0].shape[:2]
        out = np.full((h, w, 3), GREY, np.uint8)
        never = np.zeros((h, w), bool)
        frames, masks = list(self.frames), list(self.masks)
        band = max(1, 4_000_000 // (w * 3 * len(frames)))  # rows per step, ~16 MB of float32 at a time
        for y in range(0, h, band):
            stack = np.stack([f[y:y + band] for f in frames]).astype(np.float32)
            m = np.stack([mk[y:y + band] for mk in masks])
            stack[m] = np.nan
            with warnings.catch_warnings():  # "All-NaN slice": a pixel never free of people, handled below
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(stack, axis=0)
            hole = np.isnan(med).any(axis=2)
            med[hole] = GREY
            out[y:y + band] = np.clip(med, 0, 255).astype(np.uint8)
            never[y:y + band] = hole
        return out, never

    def build(self, model=None, infer=None, still: "Still | None" = None) -> tuple[np.ndarray | None, dict]:
        """The empty store at full size, and how it went. `infer(model, img)` -> person boxes, for the final check.
        With a still: its pixels fill where no sample was free, and what this build saw free is kept in it."""
        if not self.frames:
            return None, {"samples": 0}
        self.built_at = self.added
        img, hole = self.median()
        info = {"samples": len(self.frames), "never_free": round(float(hole.mean()), 4)}
        moved = False
        if still is not None and still.img is not None:
            share = still.changed(img, ~hole)
            moved = share is not None and share > MOVED_SHARE
            info["changed"] = None if share is None else round(share, 3)
            if not moved:
                use = hole & still.known
                if use.any():
                    img = _blend(img, still, use, ~hole & still.known)
                hole = hole & ~still.known
                info["from_still"] = round(float(use.mean()), 4)
        out, known, left = render(img, ~hole, self.size, self.k, model, infer)
        info.update(filled=round(float(hole.mean()), 4), greyed_after_check=left)
        if still is not None:
            info["moved"] = still.offer(img, known, moved)
        return out, info


FEATHER = 12.0  # px (sigma at the samples' size): the still's patches fade into the samples, no seam


def _blend(img: np.ndarray, still: "Still", use: np.ndarray, both: np.ndarray) -> np.ndarray:
    """The still where `use`, matched to the light now (a gain per colour from the pixels both saw free: the camera's
    exposure moves over the day) and faded in at the edges, so its patches do not show as boxes."""
    src = still.img.astype(np.float32)
    if both.mean() > 0.05:
        now, then = img[both].astype(np.float32), src[both]
        gain = np.clip(np.median(now, axis=0) / np.maximum(np.median(then, axis=0), 1.0), 0.6, 1.6)
        src = src * gain
    a = cv2.GaussianBlur(use.astype(np.float32), (0, 0), FEATHER)
    a = np.where(use, 1.0, a * still.known)[..., None]  # all still inside, a fade only where the still knows
    return np.clip(img.astype(np.float32) * (1 - a) + src * a, 0, 255).astype(np.uint8)


def render(img: np.ndarray, known: np.ndarray, size, k: float, model=None, infer=None):
    """A small picture with the pixels it knows to the full picture: the rest filled from around it, then the
    detector's check (anything it takes for a person grey, and no longer known). -> (picture, known, greyed)."""
    img, known = img.copy(), known.copy()
    hole = ~known
    if hole.any() and known.any():
        h, w = img.shape[:2]
        f = min(1.0, FILL_W / w)
        sm = cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA) if f < 1 else img
        sh = cv2.resize(hole.astype(np.uint8) * 255, (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST) if f < 1 \
            else hole.astype(np.uint8) * 255
        sh = cv2.dilate(sh, np.ones((3, 3), np.uint8))  # the hole's edge too: it can carry a little of the person
        filled = cv2.inpaint(sm, sh, 5, cv2.INPAINT_TELEA)
        if f < 1:
            filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
        img[hole] = filled[hole]
    out = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR) if k < 1 else img
    left = 0
    if model is not None and infer is not None:
        for x0, y0, x1, y1 in infer(model, out):
            px, py = PAD * (x1 - x0), PAD * (y1 - y0)
            X0, Y0, X1, Y1 = max(0, int(x0 - px)), max(0, int(y0 - py)), int(x1 + px) + 1, int(y1 + py) + 1
            out[Y0:Y1, X0:X1] = GREY
            known[int(Y0 * k):int(Y1 * k) + 1, int(X0 * k):int(X1 * k) + 1] = False
            left += 1
    return out, known, left


class Still:
    """A live camera's still: every pixel it has seen free of people, at the samples' size (Plate.small), as a PNG
    with alpha 255 where seen. Only what passed the final check is kept, so it holds no more than the empty store
    that already leaves the anonymizer."""

    def __init__(self, path: Path, small):
        self.path, self.small = Path(path), tuple(small)
        self.img, self.known, self.moved_once = None, None, False
        try:
            data = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
        except cv2.error:
            data = None
        if data is not None and data.ndim == 3 and data.shape[2] == 4 and (data.shape[1], data.shape[0]) == self.small:
            self.img, self.known = np.ascontiguousarray(data[..., :3]), data[..., 3] > 127

    def changed(self, img: np.ndarray, free: np.ndarray) -> float | None:
        """Share of the pixels both saw free that differ, or None when too few were seen by both."""
        both = free & self.known
        if both.mean() < MOVED_OVERLAP:
            return None
        g = lambda x: cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (9, 9), 0).astype(np.int16)  # noqa: E731
        return float((np.abs(g(img) - g(self.img)) > MOVED_DIFF)[both].mean())

    def offer(self, img: np.ndarray, known: np.ndarray, moved: bool) -> bool:
        """Keep what a build saw (merged with the still unless the camera moved). The first build that says moved is
        not kept; a second in a row replaces the still. -> whether the still was replaced for a move."""
        if moved and not self.moved_once:
            self.moved_once = True
            return False
        replaced = moved
        self.moved_once = False
        if not moved and self.img is not None:
            img, known = img.copy(), known.copy()
            keep = ~known & self.known
            img[keep], known[keep] = self.img[keep], True
        self.img, self.known = img, known
        self.save()
        return replaced

    def picture(self, size, k: float, model=None, infer=None):
        """The still as the empty store, before any sample is in: the camera's first second has its store."""
        if self.img is None:
            return None, None
        out, known, left = render(self.img, self.known, size, k, model, infer)
        return out, {"samples": 0, "from_still": round(float(self.known.mean()), 4), "greyed_after_check": left}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.stem}.{os.getpid()}.png")
        cv2.imwrite(str(tmp), np.dstack([self.img, np.where(self.known, 255, 0).astype(np.uint8)]))
        for k in range(40):  # Windows: the page may hold it a moment
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                if k == 39:
                    raise
                import time
                time.sleep(0.05)
