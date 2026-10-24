"""Каркас правил-нарушений.

Правило — это модуль, который на каждом кадре возвращает список «сигналов»:
«у субъекта X прямо сейчас наблюдается признак нарушения». Правило ничего не знает
про подтверждение, дубли и оповещения — этим занимается общий EventEngine.

Новое нарушение = новый файл в sv/rules/ с классом-наследником Rule и декоратором @register.
Конвейер, база, API и интерфейс при этом не меняются.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..infer.yolo import Det, KP

REGISTRY: Dict[str, type] = {}


def register(cls):
    REGISTRY[cls.type] = cls
    return cls


# ---------------------------------------------------------------- зоны
@dataclass
class Zone:
    id: str
    name: str
    kind: str = "danger"              # danger — вход запрещён; area — просто участок (для фильтра правил)
    severity: str = "critical"
    polygon: List[List[float]] = field(default_factory=list)   # нормированные координаты 0..1
    schedule: Optional[dict] = None   # {"from": "07:55", "to": "12:00", "days": [1..7]} — когда зона активна

    def __post_init__(self):
        # YAML 1.1 читает 12:00 без кавычек как число 720 (шестидесятеричное) — приводим обратно к "HH:MM"
        if self.schedule:
            for k in ("from", "to"):
                v = self.schedule.get(k)
                if isinstance(v, int):
                    self.schedule[k] = f"{v // 60:02d}:{v % 60:02d}"
                elif isinstance(v, str) and len(v) == 4 and v[1] == ":":
                    self.schedule[k] = "0" + v

    def active(self, now: datetime) -> bool:
        s = self.schedule
        if not s:
            return True
        days = s.get("days")
        if days and now.isoweekday() not in days:
            return False
        f, t = s.get("from"), s.get("to")
        if not f or not t:
            return True
        cur = now.strftime("%H:%M")
        return f <= cur <= t if f <= t else (cur >= f or cur <= t)

    def contains(self, x: float, y: float, W: int, H: int) -> bool:
        if len(self.polygon) < 3:
            return False
        pts = np.array([[p[0] * W, p[1] * H] for p in self.polygon], np.float32)
        return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0


# ---------------------------------------------------------------- история треков
class TrackHistory:
    """Последние N секунд по каждому треку: время, рамка, ключевые точки."""

    def __init__(self, keep_s: float = 15.0):
        self.keep_s = keep_s
        self.h: Dict[int, deque] = {}

    def update(self, dets: List[Det], t: float):
        for d in dets:
            q = self.h.setdefault(d.track_id, deque())
            q.append((t, d.box.copy(), None if d.kpts is None else d.kpts.copy(), d.extra.get("lying")))
        for tid in list(self.h):
            q = self.h[tid]
            while q and t - q[0][0] > self.keep_s:
                q.popleft()
            if not q:
                del self.h[tid]

    def window(self, tid: int, t: float, seconds: float):
        return [x for x in self.h.get(tid, ()) if t - x[0] <= seconds]


@dataclass
class FrameCtx:
    cam_id: str
    t: float                       # время кадра (unix, сек)
    now: datetime
    W: int
    H: int
    persons: List[Det]
    objects: List[Det]             # прочие объекты детектора (техника, телефоны...)
    zones: List[Zone]
    history: TrackHistory

    def zones_at(self, x, y, kinds=None):
        return [z for z in self.zones if (kinds is None or z.kind in kinds) and z.active(self.now)
                and z.contains(x, y, self.W, self.H)]

    def in_scope(self, d: Det, zone_names: List[str]) -> bool:
        """Правило ограничено участками? Пустой список — весь кадр."""
        if not zone_names:
            return True
        fx, fy = d.foot
        return any(z.name in zone_names or z.id in zone_names for z in self.zones_at(fx, fy))


@dataclass
class Signal:
    key: str                  # субъект: обычно track_id, для зон — track@zone
    det: Det
    zone: str = ""
    info: str = ""


class Rule:
    type = "base"
    title = "Нарушение"
    severity = "warning"       # warning | critical
    # параметры подтверждения по умолчанию (переопределяются в конфиге камеры)
    defaults = dict(confirm_s=2.0, confirm_ratio=0.6, clear_s=3.0, cooldown_s=60.0)

    def __init__(self, params: dict):
        p = dict(Rule.defaults)
        p.update(self.defaults)
        p.update(params or {})
        self.p = p

    def signals(self, ctx: FrameCtx) -> List[Signal]:
        raise NotImplementedError


# ---------------------------------------------------------------- геометрия позы
def kp(det: Det, name: str, min_conf=0.35):
    if det.kpts is None:
        return None
    k = det.kpts[KP[name]]
    return k[:2] if k[2] >= min_conf else None


def mid(a, b):
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return (a + b) / 2


def torso_angle(det: Det) -> Optional[float]:
    """Угол корпуса (плечи -> бёдра) от вертикали, градусы. None — точек не видно."""
    s = mid(kp(det, "l_sh"), kp(det, "r_sh"))
    h = mid(kp(det, "l_hip"), kp(det, "r_hip"))
    if s is None or h is None:
        return None
    dx, dy = h[0] - s[0], h[1] - s[1]
    if abs(dx) + abs(dy) < 1e-3:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def is_lying(det: Det, angle_thr=55.0, aspect_thr=1.25) -> bool:
    a = torso_angle(det)
    aspect = det.w / max(det.h, 1.0)
    if a is not None:
        # корпус горизонтален и рамка не вытянута вертикально
        return a >= angle_thr and aspect >= 0.8
    return aspect >= aspect_thr


# ---------------------------------------------------------------- движок событий
@dataclass
class OpenEvent:
    id: int
    rule: str
    key: str
    box: np.ndarray
    onset_t: float
    alert_t: float
    last_t: float
    closed_t: Optional[float] = None


class EventEngine:
    """Подтверждение во времени + подавление дублей.

    * сигнал должен держаться confirm_s секунд в доле кадров >= confirm_ratio (случайный кадр не тревожит);
    * на одно длящееся нарушение — одно событие; пока признак есть, событие «открыто»;
    * событие закрывается, если признак пропал на clear_s секунд;
    * после закрытия тот же субъект (или рамка рядом, если трек сменил номер) молчит cooldown_s.
    """

    def __init__(self, on_open: Callable, on_close: Callable):
        self.on_open = on_open
        self.on_close = on_close
        self.pending: Dict[Tuple[str, str], dict] = {}
        self.open: Dict[Tuple[str, str], OpenEvent] = {}
        self.recent: List[OpenEvent] = []

    @staticmethod
    def _iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / (u + 1e-9)

    def step(self, rule: Rule, sigs: List[Signal], ctx: FrameCtx):
        p = rule.p
        t = ctx.t
        active = {}
        for s in sigs:
            active[(rule.type, s.key)] = s

        # кадр учитывается для всех ожидающих субъектов этого правила
        for k, st in list(self.pending.items()):
            if k[0] != rule.type:
                continue
            st["frames"] += 1
            if k in active:
                st["hits"] += 1
                st["last"] = t
            elif t - st["last"] > max(1.0, p["confirm_s"] / 2):
                del self.pending[k]   # признак пропал до подтверждения — сброс

        for k, s in active.items():
            if k in self.open:
                ev = self.open[k]
                ev.last_t = t
                ev.box = s.det.box.copy()
                continue
            if self._cooling(rule, k, s, t):
                continue
            st = self.pending.get(k)
            if st is None:
                st = self.pending[k] = {"start": t, "last": t, "frames": 1, "hits": 1}
            dur = t - st["start"]
            if dur >= p["confirm_s"] and st["hits"] / max(1, st["frames"]) >= p["confirm_ratio"]:
                del self.pending[k]
                ev = OpenEvent(0, rule.type, s.key, s.det.box.copy(), st["start"], t, t)
                ev.id = self.on_open(rule, s, ctx, ev)
                self.open[k] = ev

        # закрытие
        for k, ev in list(self.open.items()):
            if k[0] != rule.type:
                continue
            if t - ev.last_t > p["clear_s"]:
                ev.closed_t = t
                del self.open[k]
                self.recent.append(ev)
                self.on_close(ev)
        self.recent = [e for e in self.recent if t - (e.closed_t or t) < 3600]

    def _cooling(self, rule: Rule, k, s: Signal, t: float) -> bool:
        cd = rule.p["cooldown_s"]
        # тот же субъект или почти та же рамка (трек мог смениться) — молчим;
        # cooldown_any: после события это правило на камере молчит cooldown_s для всех (например, стоянка)
        for ev in self.recent:
            if ev.rule != rule.type or t - ev.closed_t > cd:
                continue
            if ev.key == s.key or self._iou(ev.box, s.det.box) > 0.4 or rule.p.get("cooldown_any"):
                return True
        for ok, ev in self.open.items():
            if ok[0] == rule.type and (self._iou(ev.box, s.det.box) > 0.5 or rule.p.get("cooldown_any")):
                ev.last_t = t      # это продолжение уже открытого события
                return True
        return False
