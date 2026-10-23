"""Пре- и постобработка YOLO (v8/11): одна и та же для CPU и LinQ."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .runtime import Runtime

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# индексы ключевых точек COCO
KP = dict(nose=0, l_eye=1, r_eye=2, l_ear=3, r_ear=4, l_sh=5, r_sh=6, l_el=7, r_el=8, l_wr=9, r_wr=10,
          l_hip=11, r_hip=12, l_knee=13, r_knee=14, l_ank=15, r_ank=16)
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 5), (0, 6)]


@dataclass
class Det:
    box: np.ndarray            # x1,y1,x2,y2 в пикселях исходного кадра
    score: float
    cls: int
    label: str
    kpts: Optional[np.ndarray] = None   # [17,3] x,y,conf
    track_id: int = -1
    extra: dict = field(default_factory=dict)

    @property
    def foot(self):
        x1, y1, x2, y2 = self.box
        return float((x1 + x2) / 2), float(y2)

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return float((x1 + x2) / 2), float((y1 + y2) / 2)

    @property
    def h(self):
        return float(self.box[3] - self.box[1])

    @property
    def w(self):
        return float(self.box[2] - self.box[0])


def letterbox(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, NCHW
    return np.ascontiguousarray(x), r, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, int)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ovr <= iou]
    return np.array(keep, int)


class YoloModel:
    """task='pose' (1 класс + 17 точек) или 'detect' (80 классов COCO или свои)."""

    def __init__(self, runtime: Runtime, task: str, names: Sequence[str] = COCO_NAMES,
                 conf: float = 0.35, iou: float = 0.5, classes: Optional[Sequence[int]] = None):
        self.rt = runtime
        self.task = task
        self.names = list(names) if task == "detect" else ["person"]
        self.conf = conf
        self.iou = iou
        self.classes = set(classes) if classes is not None else None
        self.last_timing = {}

    def __call__(self, frame: np.ndarray) -> List[Det]:
        import time
        t0 = time.perf_counter()
        x, r, dx, dy = letterbox(frame, self.rt.imgsz)
        t1 = time.perf_counter()
        out = self.rt(x)[0]           # [C, N]
        t2 = time.perf_counter()
        dets = self._decode(out.T, r, dx, dy, frame.shape)
        t3 = time.perf_counter()
        self.last_timing = {"pre_ms": (t1 - t0) * 1000, "post_ms": (t3 - t2) * 1000, **self.rt.last_timing,
                            "total_ms": (t3 - t0) * 1000}
        return dets

    def _decode(self, p: np.ndarray, r, dx, dy, shape) -> List[Det]:
        if self.task == "pose":
            scores = p[:, 4]
            cls = np.zeros(len(p), int)
        else:
            cs = p[:, 4:4 + len(self.names)]
            cls = cs.argmax(1)
            scores = cs[np.arange(len(cs)), cls]
        m = scores >= self.conf
        if self.classes is not None:
            m &= np.isin(cls, list(self.classes))
        p, scores, cls = p[m], scores[m], cls[m]
        if not len(p):
            return []
        cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        # NMS по классам: сдвигаем рамки разных классов
        keep = nms(boxes + cls[:, None] * 4096.0, scores, self.iou)
        H, W = shape[:2]
        res = []
        for i in keep:
            b = boxes[i].copy()
            b[[0, 2]] = (b[[0, 2]] - dx) / r
            b[[1, 3]] = (b[[1, 3]] - dy) / r
            b[[0, 2]] = b[[0, 2]].clip(0, W - 1)
            b[[1, 3]] = b[[1, 3]].clip(0, H - 1)
            k = None
            if self.task == "pose":
                k = p[i, 5:5 + 51].reshape(17, 3).copy()
                k[:, 0] = (k[:, 0] - dx) / r
                k[:, 1] = (k[:, 1] - dy) / r
            c = int(cls[i])
            res.append(Det(b, float(scores[i]), c, self.names[c] if c < len(self.names) else str(c), k))
        return res
