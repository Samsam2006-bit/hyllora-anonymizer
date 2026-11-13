"""Other person and pose models than Ultralytics, for comparing them (REVIEW_BACKLOG LIC1, WORKLOG 2026-11-10).

    python -m anonymizer.main <video> <out> --pose rtmo-s
    python -m anonymizer.main <video> <out> --pose rfdetr-nano+rtmpose-s

Each backend gives what main.infer gives: boxes, confidences, track ids and COCO-17 keypoints in full-frame pixels.
The models with an Apache-2.0 licence are tracked with supervision's ByteTrack (MIT), not Ultralytics' BoT-SORT;
`yolo26s-pose+bytetrack` is the control that tells the model's part from the tracker's.

Not in requirements.txt: rtmlib and onnxruntime (RTMO, RTMPose, YOLOX) and rfdetr are installed only where the
comparison runs. Their weights are fetched to ~/.cache/rtmlib and ~/.roboflow on first use, not through
inferero.weights (S3): fine for a comparison, not for a store.
"""

import os

import numpy as np
import supervision as sv
import torch

RTMO = {"s": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip",
        "m": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",
        "l": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip"}
RTMPOSE = {"t": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip",
           "s": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
           "m": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"}
YOLOX = {"tiny": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip", (416, 416)),
         "m": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip", (640, 640))}
DETR_CONF = 0.6  # RF-DETR boxes: its many small unsure ones broke tracks (medurummet, 2026-11-10)
BATCH = 4  # RTMPose crops per call, a fixed shape (see _RtmPose)
ORT = "mps"  # rtmlib's name for onnxruntime's CoreML provider; YOLOX fails on it (its NMS) and runs on the CPU


def names() -> list[str]:
    return (["yolo26s-pose+bytetrack", "yolo26n-pose+bytetrack", "yolo26m-pose+bytetrack", "rfdetr-kp"]
            + [f"rtmo-{s}" for s in RTMO]
            + [f"{d}+rtmpose-{p}" for d in ("rfdetr-nano", "rfdetr-small", "rfdetr-medium", "yolox-tiny", "yolox-m")
               for p in RTMPOSE])


def _kbox(k, c, thr=0.3):
    """A person's box from its keypoints (RTMO gives none back through rtmlib)."""
    v = k[c >= thr] if (c >= thr).sum() >= 2 else k
    (x0, y0), (x1, y1) = v.min(0), v.max(0)
    h = y1 - y0
    return np.array([x0 - 0.1 * h, y0 - 0.12 * h, x1 + 0.1 * h, y1 + 0.05 * h])


class _Rfdetr:
    """RF-DETR person boxes (COCO class 1), half precision on the GPU."""

    def __init__(self, cls, device, conf):
        m = cls(device=device)
        self.m = m.inference(compile=False, dtype=torch.float16) if device != "cpu" else m
        self.conf = conf

    def boxes(self, img):
        d = self.m.predict(np.ascontiguousarray(img[:, :, ::-1]), threshold=max(self.conf, DETR_CONF))
        d = d[d.class_id == 1]
        if len(d):
            d = d.with_nms(threshold=0.5)  # RF-DETR gives the odd second box on one person
        return d.xyxy, d.confidence


class _Yolox:
    def __init__(self, size, conf):
        from rtmlib import YOLOX as Y
        url, sz = YOLOX[size]
        self.m = Y(url, model_input_size=sz, backend="onnxruntime", device="cpu", score_thr=conf)

    def boxes(self, img):
        b = np.asarray(self.m(img), dtype=float).reshape(-1, 4)
        return b, np.ones(len(b))  # rtmlib keeps only boxes above score_thr and drops the score


class _RtmPose:
    """RTMPose on every person's crop, all of a frame's people in one call."""

    def __init__(self, size):
        from rtmlib import RTMPose
        self.m = RTMPose(RTMPOSE[size], model_input_size=(192, 256), backend="onnxruntime", device=ORT)
        self.inp = self.m.session.get_inputs()[0].name
        self.outs = [o.name for o in self.m.session.get_outputs()]

    def __call__(self, img, boxes):
        n = len(boxes)
        if not n:
            return np.zeros((0, 17, 2)), np.zeros((0, 17))
        crops, cs = [], []
        for b in boxes:
            c, center, scale = self.m.preprocess(img, b)
            crops.append(np.ascontiguousarray(c.transpose(2, 0, 1), dtype=np.float32))
            cs.append((center, scale))
        # always BATCH crops per call (the rest zeros): CoreML rebuilds the model for every new input shape (~1 s)
        x = np.stack(crops)
        x = np.concatenate([x, np.zeros((-n % BATCH, *x.shape[1:]), dtype=x.dtype)])
        parts = [self.m.session.run(self.outs, {self.inp: x[i:i + BATCH]}) for i in range(0, len(x), BATCH)]
        out = [np.concatenate([p[j] for p in parts]) for j in range(len(self.outs))]
        k, s = zip(*(self.m.postprocess([o[i:i + 1] for o in out], *cs[i]) for i in range(n)))
        return np.concatenate(k), np.concatenate(s)


class Backend:
    """`name` from names(); `rate` the analysed frames per second (the tracker's clock)."""

    def __init__(self, name: str, conf: float, imgsz: int, device: str, half: bool, rate: float, buffer_s: float,
                 weights_path):
        self.name, self.conf, self.imgsz, self.device, self.half = name, conf, imgsz, device, half
        self.det = self.pose = self.rtmo = self.yolo = self.kp = None
        if name.endswith("+bytetrack"):
            from ultralytics import YOLO
            self.yolo = YOLO(weights_path(name.split("+")[0] + ".pt"))
        elif name.startswith("rtmo-"):
            from rtmlib import RTMO as R
            self.rtmo = R(RTMO[name.split("-")[1]], model_input_size=(640, 640), backend="onnxruntime", device=ORT,
                          score_thr=conf)
        elif name == "rfdetr-kp":
            import rfdetr
            m = rfdetr.RFDETRKeypointPreview(device=device)
            self.kp = m.inference(compile=False, dtype=torch.float16) if device != "cpu" else m
        else:
            det, pose = name.split("+")
            if det.startswith("rfdetr-"):
                import rfdetr
                cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}
                self.det = _Rfdetr(cls[det.split("-")[1]], device, conf)
            else:
                self.det = _Yolox(det.split("-")[1], conf)
            self.pose = _RtmPose(pose.split("-")[1])
        # supervision counts lost_track_buffer in frames at 30 frames/s (max_time_lost = frame_rate / 30 * buffer)
        self.tracker = sv.ByteTrack(lost_track_buffer=max(1, round(buffer_s * 30)), frame_rate=max(1, round(rate)),
                                    minimum_consecutive_frames=1)

    def detect(self, img):
        """(xyxy, conf, kxy, kc) in the picture's pixels, not tracked."""
        if self.yolo is not None:
            r = self.yolo.predict(img, classes=[0], conf=self.conf, imgsz=self.imgsz, device=self.device,
                                  verbose=False, quantize=16 if self.half else None)[0]
            if r.boxes is None or not len(r.boxes):
                return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 2)), np.zeros((0, 17))
            return (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.keypoints.xy.cpu().numpy(),
                    r.keypoints.conf.cpu().numpy())
        if self.rtmo is not None:
            k, c = self.rtmo(img)
            if k is None or not len(k):
                return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 2)), np.zeros((0, 17))
            box = np.array([_kbox(k[i], c[i]) for i in range(len(k))])
            return box, np.array([np.sort(ci)[-5:].mean() for ci in c]), k, c
        if self.kp is not None:
            d = self.kp.predict(np.ascontiguousarray(img[:, :, ::-1]), threshold=self.conf)
            if not len(d):
                return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 2)), np.zeros((0, 17))
            kc = d.keypoint_confidence
            return np.asarray(d.data["xyxy"]), np.asarray(d.data.get("confidence", kc.mean(1))), d.xy, kc
        b, s = self.det.boxes(img)
        k, c = self.pose(img, b)
        return b, s, k, c

    def track(self, img):
        xyxy, conf, kxy, kc = self.detect(img)
        n = len(xyxy)
        det = sv.Detections(xyxy=np.asarray(xyxy, dtype=np.float32).reshape(-1, 4),
                            confidence=np.asarray(conf, dtype=np.float32).reshape(-1),
                            class_id=np.zeros(n, dtype=int),
                            data={"kxy": np.asarray(kxy, dtype=float).reshape(n, 17, 2),
                                  "kc": np.asarray(kc, dtype=float).reshape(n, 17)})
        t = self.tracker.update_with_detections(det)
        if not len(t):
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros((0, 17, 2)), np.zeros((0, 17))
        return t.xyxy, t.confidence, t.tracker_id.astype(int), t.data["kxy"], t.data["kc"]
