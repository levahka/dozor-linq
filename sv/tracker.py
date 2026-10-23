"""Лёгкий трекер в духе ByteTrack: двухэтапное сопоставление по IoU + постоянный предсказатель скорости.

Свой, без зависимостей (лицензия проекта не тянет AGPL).
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from .infer.yolo import Det


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


class Track:
    _next = 1

    def __init__(self, det: Det, t: float):
        self.id = Track._next
        Track._next += 1
        self.box = det.box.astype(float).copy()
        self.vel = np.zeros(4)
        self.det = det
        self.hits = 1
        self.last_t = t
        self.first_t = t
        self.missed = 0

    def predict(self, t: float) -> np.ndarray:
        dt = max(0.0, t - self.last_t)
        return self.box + self.vel * min(dt, 1.0)

    def update(self, det: Det, t: float):
        dt = max(1e-3, t - self.last_t)
        new = det.box.astype(float)
        v = (new - self.box) / dt
        self.vel = 0.6 * self.vel + 0.4 * v
        self.box = new
        self.det = det
        self.hits += 1
        self.last_t = t
        self.missed = 0


class ByteTrackLite:
    def __init__(self, high: float = 0.5, low: float = 0.2, match_iou: float = 0.25,
                 max_age_s: float = 2.0, min_hits: int = 2):
        self.high, self.low = high, low
        self.match_iou = match_iou
        self.max_age_s = max_age_s
        self.min_hits = min_hits
        self.tracks: List[Track] = []

    def _match(self, tracks, dets, t):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        pb = np.array([tr.predict(t) for tr in tracks])
        db = np.array([d.box for d in dets])
        iou = iou_matrix(pb, db)
        # при низкой частоте кадров рамки быстро идущего человека могут не пересекаться —
        # добавляем близость центров (в долях роста) и похожесть размера
        pc = (pb[:, None, :2] + pb[:, None, 2:]) / 2
        dc = (db[None, :, :2] + db[None, :, 2:]) / 2
        ph = (pb[:, None, 3] - pb[:, None, 1])
        dh = (db[None, :, 3] - db[None, :, 1])
        dist = np.linalg.norm(pc - dc, axis=2) / np.maximum((ph + dh) / 2, 1.0)
        size = np.minimum(ph, dh) / np.maximum(np.maximum(ph, dh), 1.0)
        near = np.clip(1.0 - dist / 0.8, 0, None) * 0.6 * (size > 0.6)
        iou = np.maximum(iou, near)
        r, c = linear_sum_assignment(-iou)
        matches, ut, ud = [], set(range(len(tracks))), set(range(len(dets)))
        for i, j in zip(r, c):
            if iou[i, j] >= self.match_iou:
                matches.append((i, j))
                ut.discard(i)
                ud.discard(j)
        return matches, sorted(ut), sorted(ud)

    def update(self, dets: List[Det], t: float) -> List[Det]:
        hi = [d for d in dets if d.score >= self.high]
        lo = [d for d in dets if self.low <= d.score < self.high]
        m1, ut, ud = self._match(self.tracks, hi, t)
        for i, j in m1:
            self.tracks[i].update(hi[j], t)
        rest = [self.tracks[i] for i in ut]
        m2, ut2, _ = self._match(rest, lo, t)
        for i, j in m2:
            rest[i].update(lo[j], t)
        for i in ut2:
            rest[i].missed += 1
        for j in ud:
            self.tracks.append(Track(hi[j], t))
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= self.max_age_s]
        out = []
        for tr in self.tracks:
            if tr.last_t == t and tr.hits >= self.min_hits:
                tr.det.track_id = tr.id
                out.append(tr.det)
        return out
