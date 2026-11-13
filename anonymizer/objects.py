"""Things the store shows the anonymizer a picture of (a shopping basket, a cart), found in the camera's picture and
sent on as boxes only (skeleton.py, "o"). The analysis counts the ones left where they do not belong
(inferero/pipeline/objects.py).

    data/objects/objects.json   {"objects": [{"id", "name", "refs": [{"image": <file in data/objects>, "box": [x0, y0, x1, y1]}]}]}

The store uploads a photo of its basket and draws a box round it (server.py). YOLOE takes the boxed part as a
visual prompt: its embedding, averaged over every photo of the same thing, becomes that thing's class. No text model
and nothing fetched: only the YOLOE weights in models/. The photos stay in data/objects, on this computer, and are
never sent on: only the boxes found in the camera's picture are.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from . import paths

DIR = paths.OBJECTS
FILE = "objects.json"
MODEL = "yoloe-26m-seg.pt"  # the small one scored the same basket 0.25 against 0.58
IMGSZ, CONF = 1280, 0.2  # baskets are small in a ceiling camera's picture: a larger input than for people
EVERY_S = 1.0  # looked for once per this much video time: a left basket stands still, so every frame is no help


def load(folder: Path | None = None) -> list[dict]:
    """The things to look for, each with at least one photo that is there; [] when none are set up."""
    folder = folder or DIR
    try:
        objs = json.loads((folder / FILE).read_text()).get("objects", [])
    except (OSError, ValueError):
        return []
    out = []
    for o in objs:
        refs = [r for r in o.get("refs", []) if (folder / r["image"]).is_file() and len(r.get("box", [])) == 4]
        if o.get("name") and refs:
            out.append({**o, "refs": refs})
    return out


def save(objs: list[dict], folder: Path | None = None) -> None:
    folder = folder or DIR
    folder.mkdir(parents=True, exist_ok=True)
    (folder / FILE).write_text(json.dumps({"objects": objs}, indent=2, ensure_ascii=False))


class Finder:
    """YOLOE with one class per thing, from the store's photos. `find(frame, roi)` -> rows [x0, y0, x1, y1, conf,
    class] in full-frame pixels."""

    def __init__(self, objs: list[dict], device: str, half: bool, folder: Path | None = None):
        import torch
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        from inferero import weights

        folder = folder or DIR
        self.names = [o["name"] for o in objs]
        self.device, self.half = device, half
        self.model = YOLOE(weights.path(MODEL))
        per_class = []
        for o in objs:
            vecs = []
            for r in o["refs"]:
                img = cv2.imread(str(folder / r["image"]))
                if img is None:
                    continue
                # one photo as its own visual prompt: the model sets its class embedding (pe) from the boxed part
                self.model.predict(img, refer_image=img, predictor=YOLOEVPSegPredictor, device=device, verbose=False,
                                   visual_prompts={"bboxes": np.array([r["box"]], np.float64), "cls": np.array([0])})
                vecs.append(self.model.model.pe[0, 0].detach().float().cpu())
            if not vecs:
                raise ValueError(f"Ingen bild att läsa för {o['name']}")
            v = torch.stack(vecs).mean(0)
            per_class.append(v / v.norm())
        self.model.model.set_classes(self.names, torch.stack(per_class)[None].to(self.model.model.pe.dtype))
        self.model.predictor = None  # the next predict is a plain one, with these classes

    def find(self, frame: np.ndarray, roi=None) -> np.ndarray:
        x0, y0 = (roi[0], roi[1]) if roi else (0, 0)
        img = frame[roi[1]:roi[3], roi[0]:roi[2]] if roi else frame
        r = self.model.predict(img, conf=CONF, imgsz=IMGSZ, device=self.device, verbose=False,
                               quantize=16 if self.half else None)[0]
        b = r.boxes
        if b is None or not len(b):
            return np.zeros((0, 6))
        xyxy = b.xyxy.cpu().numpy() + np.array([x0, y0, x0, y0])
        return np.column_stack([xyxy, b.conf.cpu().numpy(), b.cls.cpu().numpy()])
