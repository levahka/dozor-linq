"""Реализованные поведенческие паттерны (номера — как в кейсе)."""
from __future__ import annotations

import math
from typing import List

import numpy as np

from .base import FrameCtx, Rule, Signal, is_lying, kp, register

VEHICLES = {"car", "truck", "bus", "motorcycle", "train", "forklift", "tractor", "excavator", "loader"}


@register
class FallRule(Rule):
    """07. Падение / человек лежит на полу."""

    type = "fall"
    title = "Падение человека"
    severity = "critical"
    defaults = dict(confirm_s=1.5, confirm_ratio=0.6, clear_s=4.0, cooldown_s=120.0,
                    angle_thr=55.0, aspect_thr=1.25, zones=[])

    def signals(self, ctx: FrameCtx) -> List[Signal]:
        out = []
        for d in ctx.persons:
            lying = is_lying(d, self.p["angle_thr"], self.p["aspect_thr"])
            d.extra["lying"] = lying
            if lying and ctx.in_scope(d, self.p["zones"]):
                # было ли падение (стоял недавно) или уже лежал
                hist = ctx.history.window(d.track_id, ctx.t, 4.0)
                was_up = any(x[3] is False for x in hist)
                out.append(Signal(str(d.track_id), d, info="резкое падение" if was_up else "человек лежит"))
        return out


@register
class ImmobileRule(Rule):
    """07. Неподвижное лежание — признак потери сознания."""

    type = "immobile"
    title = "Неподвижность, возможна потеря сознания"
    severity = "critical"
    defaults = dict(confirm_s=1.0, confirm_ratio=0.6, clear_s=4.0, cooldown_s=300.0,
                    window_s=8.0, motion_thr=0.12, zones=[])

    def signals(self, ctx: FrameCtx) -> List[Signal]:
        out = []
        for d in ctx.persons:
            if not d.extra.get("lying"):
                continue
            hist = ctx.history.window(d.track_id, ctx.t, self.p["window_s"])
            if not hist or ctx.t - hist[0][0] < self.p["window_s"] * 0.9:
                continue
            if sum(1 for x in hist if x[3]) < 0.8 * len(hist):
                continue
            c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for _, b, _, _ in hist])
            diag = math.hypot(d.w, d.h)
            spread = float(np.linalg.norm(c.max(0) - c.min(0)))
            if spread < self.p["motion_thr"] * diag and ctx.in_scope(d, self.p["zones"]):
                out.append(Signal(str(d.track_id), d, info=f"не двигается {self.p['window_s']:.0f}+ с"))
        return out


@register
class RunningRule(Rule):
    """01. Бег, резкие перемещения. Скорость меряется в «ростах человека в секунду» —
    так порог не зависит от расстояния до камеры."""

    type = "running"
    title = "Бег по цеху"
    severity = "warning"
    defaults = dict(confirm_s=1.0, confirm_ratio=0.7, clear_s=2.0, cooldown_s=60.0,
                    speed_thr=1.4, window_s=0.8, zones=[])

    def signals(self, ctx: FrameCtx) -> List[Signal]:
        out = []
        for d in ctx.persons:
            hist = ctx.history.window(d.track_id, ctx.t, self.p["window_s"])
            if len(hist) < 3:
                continue
            (t0, b0, _, _), (t1, b1, _, _) = hist[0], hist[-1]
            dt = t1 - t0
            if dt < self.p["window_s"] * 0.6:
                continue
            f0 = np.array([(b0[0] + b0[2]) / 2, b0[3]])
            f1 = np.array([(b1[0] + b1[2]) / 2, b1[3]])
            h = np.median([b[3] - b[1] for _, b, _, _ in hist])
            speed = float(np.linalg.norm(f1 - f0) / dt / max(h, 1.0))
            d.extra["speed"] = speed
            if speed >= self.p["speed_thr"] and not d.extra.get("lying") and ctx.in_scope(d, self.p["zones"]):
                out.append(Signal(str(d.track_id), d, info=f"скорость {speed:.1f} роста/с"))
        return out


@register
class ZoneRule(Rule):
    """Вход в опасную зону (полигон на кадре, может действовать по расписанию)."""

    type = "zone"
    title = "Вход в опасную зону"
    severity = "critical"
    defaults = dict(confirm_s=1.0, confirm_ratio=0.7, clear_s=3.0, cooldown_s=60.0)

    def signals(self, ctx: FrameCtx) -> List[Signal]:
        out = []
        for d in ctx.persons:
            fx, fy = d.foot
            for z in ctx.zones_at(fx, fy, kinds={"danger"}):
                out.append(Signal(f"{d.track_id}@{z.id}", d, zone=z.name, info=f"зона «{z.name}»"))
        return out


@register
class ProximityRule(Rule):
    """02. Опасное сближение человека с движущейся техникой/транспортом."""

    type = "proximity"
    title = "Опасное сближение с техникой"
    severity = "critical"
    defaults = dict(confirm_s=0.7, confirm_ratio=0.6, clear_s=3.0, cooldown_s=60.0,
                    margin=0.6, require_moving=True, min_speed=0.08, zones=[])

    def signals(self, ctx: FrameCtx) -> List[Signal]:
        vehicles = [o for o in ctx.objects if o.label in VEHICLES]
        if not vehicles:
            return []
        out = []
        for d in ctx.persons:
            if not ctx.in_scope(d, self.p["zones"]):
                continue
            fx, fy = d.foot
            m = self.p["margin"] * d.h
            for v in vehicles:
                if self.p["require_moving"] and v.extra.get("speed", 0.0) < self.p["min_speed"]:
                    continue
                x1, y1, x2, y2 = v.box
                if x1 - m <= fx <= x2 + m and y1 - m * 0.5 <= fy <= y2 + m:
                    # человек не «внутри» машины (водитель): перекрытие рамок не почти полное
                    inside = d.box[0] >= x1 and d.box[2] <= x2 and d.box[1] >= y1 and d.box[3] <= y2
                    if inside and d.h < 0.6 * (y2 - y1):
                        continue
                    out.append(Signal(str(d.track_id), d, info=f"{v.label} в {m:.0f}px"))
                    break
        return out


@register
class PhoneRule(Rule):
    """03. Телефон в руках/у уха при работе или перемещении."""

    type = "phone"
    title = "Использование телефона"
    severity = "warning"
    defaults = dict(confirm_s=2.0, confirm_ratio=0.4, clear_s=4.0, cooldown_s=120.0,
                    near=0.22, min_conf=0.25, zones=[])

    def signals(self, ctx: FrameCtx) -> List[Signal]:
        phones = [o for o in ctx.objects if o.label == "cell phone" and o.score >= self.p["min_conf"]]
        if not phones:
            return []
        out = []
        for d in ctx.persons:
            if not ctx.in_scope(d, self.p["zones"]):
                continue
            pts = [p for p in (kp(d, n, 0.3) for n in ("l_wr", "r_wr", "l_ear", "r_ear", "nose")) if p is not None]
            for ph in phones:
                cx, cy = ph.center
                x1, y1, x2, y2 = d.box
                if not (x1 - 0.1 * d.w <= cx <= x2 + 0.1 * d.w and y1 <= cy <= y1 + 0.75 * d.h):
                    continue
                if pts and min(math.hypot(cx - p[0], cy - p[1]) for p in pts) > self.p["near"] * d.h:
                    continue
                out.append(Signal(str(d.track_id), d, info=f"телефон {ph.score:.2f}"))
                break
        return out


# Шаблоны следующих модулей (архитектура готова, нужна модель/данные):
#   smoking   04 — детектор сигареты/огня (датасеты fire/smoke/cigarette на Roboflow) + рука у лица по позе
#   climbing  05 — ключевые точки выше линии ограждения + пересечение полигона-ограждения
#   harness   06 — зона «высота» + детектор страховочной привязи
#   carrying  08 — объект-груз в руках + наклон корпуса по позе
