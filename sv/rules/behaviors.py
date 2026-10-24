"""Реализованные поведенческие паттерны (номера — как в кейсе)."""
from __future__ import annotations

import math
from typing import List

import numpy as np

from .base import FrameCtx, Rule, Signal, is_lying, kp, register, torso_angle

VEHICLES = {"car", "truck", "bus", "motorcycle", "train", "forklift", "tractor", "excavator", "loader"}


@register
class FallRule(Rule):
    """07. Падение / человек лежит на полу."""

    type = "fall"
    title = "Падение человека"
    severity = "critical"
    defaults = dict(confirm_s=1.5, confirm_ratio=0.6, clear_s=4.0, cooldown_s=120.0,
                    angle_thr=55.0, aspect_thr=1.25, edge=0.012, zones=[])

    def signals(self, ctx: FrameCtx) -> List[Signal]:
        out = []
        e = self.p["edge"]
        for d in ctx.persons:
            x1, y1, x2, y2 = d.box
            # человек обрезан краем кадра (сидит у камеры, выходит из кадра): по форме рамки позу не судим,
            # верим только углу корпуса по ключевым точкам (плечи и бёдра должны быть видны)
            cut = x1 <= e * ctx.W or y1 <= e * ctx.H or x2 >= (1 - e) * ctx.W or y2 >= (1 - e) * ctx.H
            if cut:
                a = torso_angle(d)
                lying = a is not None and a >= self.p["angle_thr"] and d.w >= 0.8 * d.h
            else:
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
                    speed_thr=0.95, window_s=1.2, focal=0.75, zones=[])

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
            if any(x[3] for x in hist):   # лежал/вставал в этом окне — рамка «прыгает», это не бег
                continue
            hs = [b[3] - b[1] for _, b, _, _ in hist]
            h = max(hs)                   # рост стоя: если человек нагнулся, рамка ниже, но скорость не должна расти
            lateral = float(np.linalg.norm(f1 - f0) / max(h, 1.0))
            # движение к камере или от неё: рост в кадре меняется, а точка ног почти стоит.
            # Расстояние ~ f/рост, поэтому смещение в «ростах» = f·|1/h1 − 1/h0| (f ≈ focal·ширина кадра)
            ha = np.median(hs[: max(1, len(hs) // 3)])
            hb = np.median(hs[-max(1, len(hs) // 3):])
            touch = b1[3] >= 0.985 * ctx.H or b0[3] >= 0.985 * ctx.H or b1[1] <= 0.01 * ctx.H
            upright = all((b[3] - b[1]) >= 1.8 * (b[2] - b[0]) for _, b, _, _ in hist)
            touch = touch or not upright   # глубину считаем только по прямо идущему человеку
            depth = 0.0 if touch else self.p["focal"] * ctx.W * abs(1.0 / max(hb, 1.0) - 1.0 / max(ha, 1.0))
            speed = float(math.hypot(lateral, depth) / dt)
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
    """03. Телефон в руках/у уха при работе или перемещении.

    Два признака, любой из них:
    * детектор видит телефон рядом с кистью/ухом человека;
    * поза по ключевым точкам: обе кисти подняты к груди и сведены вместе (смотрит в телефон)
      или кисть у головы выше плеч (разговор). Телефон на заводской камере часто занимает
      несколько пикселей и детектору не виден, а поза видна всегда.
    """

    type = "phone"
    title = "Использование телефона"
    severity = "warning"
    defaults = dict(confirm_s=2.0, confirm_ratio=0.6, clear_s=4.0, cooldown_s=120.0,
                    near=0.22, min_conf=0.25, pose=True, text_wy=0.5, text_wd=0.6, call_head=0.45, walk_h=0.3, zones=[])

    @staticmethod
    def posture(d, p):
        g = lambda n: kp(d, n, 0.3)
        ls, rs, lh, rh = g("l_sh"), g("r_sh"), g("l_hip"), g("r_hip")
        if any(x is None for x in (ls, rs, lh, rh)):
            return None
        sh, hp = (ls + rs) / 2, (lh + rh) / 2
        T = float(np.linalg.norm(sh - hp))
        if T < 8:
            return None
        lw, rw = g("l_wr"), g("r_wr")
        head = [x for x in (g("l_ear"), g("r_ear"), g("nose")) if x is not None]
        if lw is not None and rw is not None:
            wy = min(hp[1] - lw[1], hp[1] - rw[1]) / T
            if wy >= p["text_wy"] and np.linalg.norm(lw - rw) / T <= p["text_wd"]:
                return "смотрит в телефон"
        for w in (lw, rw):
            if w is not None and head and (hp[1] - w[1]) / T >= 0.8 and \
                    min(np.linalg.norm(w - h) for h in head) / T <= p["call_head"]:
                return "говорит по телефону"
        return None

    def _walked(self, ctx, d) -> bool:
        """Поза без видимого телефона учитывается у человека, который перемещается по цеху
        (кейс: «при перемещении»); у неподвижных фигур (плакат, манекен) — нет."""
        hist = ctx.history.window(d.track_id, ctx.t, 15.0)
        if len(hist) < 3:
            return False
        ts = np.array([x[0] for x in hist])
        f = np.array([[(b[0] + b[2]) / 2, b[3]] for _, b, _, _ in hist])
        hs = np.array([b[3] - b[1] for _, b, _, _ in hist], float)
        cut = np.array([b[3] >= 0.985 * ctx.H for _, b, _, _ in hist])
        h = hs.max()
        fw = 0.75 * ctx.W
        # хоть раз шёл: медианное положение за секунду сдвинулось относительно предыдущей секунды
        # на walk_h роста вбок или к камере/от неё. Медианы гасят дрожание рамки и перескоки трекера,
        # поэтому неподвижные фигуры (плакат, манекен) сюда не попадают
        for t in np.arange(ts[0] + 2.0, ts[-1] + 1e-6, 0.5):
            A = (ts >= t - 2.0) & (ts < t - 1.0)
            B = (ts >= t - 1.0) & (ts <= t)
            if A.sum() < 2 or B.sum() < 2:
                continue
            lat = np.linalg.norm(np.median(f[B], 0) - np.median(f[A], 0)) / h
            ha, hb = max(np.median(hs[A]), 1), max(np.median(hs[B]), 1)
            # рост в кадре должен заметно измениться (≥8% за секунду), иначе это дрожание рамки
            # и точка ног должна сдвинуться вместе с ростом: идущий к камере «опускается» в кадре,
            # а у рамки, которая просто раздулась, низ стоит на месте
            dfoot = abs(np.median(f[B][:, 1]) - np.median(f[A][:, 1]))
            dep = 0.0 if cut[A | B].any() or abs(hb - ha) < 0.08 * ha or dfoot < 0.5 * abs(hb - ha) \
                else fw * abs(1 / hb - 1 / ha)
            if math.hypot(lat, dep) >= self.p["walk_h"]:
                return True
        return False

    def signals(self, ctx: FrameCtx) -> List[Signal]:
        phones = [o for o in ctx.objects if o.label == "cell phone" and o.score >= self.p["min_conf"]]
        out = []
        for d in ctx.persons:
            if d.extra.get("lying") or not ctx.in_scope(d, self.p["zones"]):
                continue
            pts = [p for p in (kp(d, n, 0.3) for n in ("l_wr", "r_wr", "l_ear", "r_ear", "nose")) if p is not None]
            hit = None
            for ph in phones:
                cx, cy = ph.center
                x1, y1, x2, y2 = d.box
                if not (x1 - 0.1 * d.w <= cx <= x2 + 0.1 * d.w and y1 <= cy <= y1 + 0.75 * d.h):
                    continue
                if pts and min(math.hypot(cx - p[0], cy - p[1]) for p in pts) > self.p["near"] * d.h:
                    continue
                hit = f"телефон {ph.score:.2f}"
                break
            if hit is None and self.p["pose"] and d.h >= 0.12 * ctx.H and self._walked(ctx, d):
                hit = self.posture(d, self.p)
            if hit:
                out.append(Signal(str(d.track_id), d, info=hit))
        return out


# Шаблоны следующих модулей (архитектура готова, нужна модель/данные):
#   smoking   04 — детектор сигареты/огня (датасеты fire/smoke/cigarette на Roboflow) + рука у лица по позе
#   climbing  05 — ключевые точки выше линии ограждения + пересечение полигона-ограждения
#   harness   06 — зона «высота» + детектор страховочной привязи
#   carrying  08 — объект-груз в руках + наклон корпуса по позе
