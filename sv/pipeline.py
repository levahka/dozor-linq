"""Конвейер обработки одной камеры.

источник (RTSP / файл / папка стоп-кадров / веб-камера)
  -> поза людей (YOLO-pose)  + детектор объектов (YOLO, раз в N кадров)   [LinQ TPU или CPU]
  -> трекинг (постоянные track_id)
  -> модули-правила -> EventEngine (подтверждение, антидребезг, дедупликация)
  -> журнал + кадр/фрагмент-доказательство + оповещения (веб, Telegram)
"""
from __future__ import annotations

import glob
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from .infer.runtime import make_runtime
from .infer.yolo import SKELETON, Det, YoloModel
from .rules import behaviors  # noqa: F401  (регистрация правил)
from .rules.base import REGISTRY, EventEngine, FrameCtx, Signal, TrackHistory, Zone, is_lying
from .tracker import ByteTrackLite

log = logging.getLogger("pipeline")

SEV_COLOR = {"critical": (40, 40, 230), "warning": (0, 165, 255)}


# ------------------------------------------------------------------ модели (общие на все камеры)
class Models:
    def __init__(self, cfg: dict):
        b = cfg.get("backend", {})
        kind = b.get("kind", "cpu")
        self.kind = kind
        kw_pose, kw_det = {}, {}
        if kind == "linq":
            kw_pose = dict(tpu_program=b["tpu_programs"]["pose"])
            kw_det = dict(tpu_program=b["tpu_programs"]["detect"])
        elif kind == "remote":
            kw_pose = dict(url=b["remote_url"], model_key="pose")
            kw_det = dict(url=b["remote_url"], model_key="detect")
        elif kind == "rabbit":
            r = b.get("rabbit", {})
            common = dict(host=r.get("host", "10.0.114.30"), login=r["login"], port=r.get("port", 5672),
                          user=r.get("user", "guest"), password=r.get("password", "guest"),
                          timeout=r.get("timeout", 10.0), raw=r.get("raw", True))
            kw_pose = dict(tpu_program=b["tpu_programs"]["pose"], **common)
            kw_det = dict(tpu_program=b["tpu_programs"]["detect"], **common)
        det_classes = b.get("detect_classes")  # None — все
        if kind == "relay":   # быстрый режим: посредник linq/relay.py на стенде LinQ
            from .infer.relay_client import RelayYolo
            r = b.get("rabbit", {})
            common = dict(host=r.get("host", "10.0.114.30"), login=r["login"], timeout=r.get("timeout", 10.0))
            self.pose = RelayYolo("pose", conf=b.get("pose_conf", 0.2), **common)
            self.detect = RelayYolo("detect", conf=b.get("detect_conf", 0.25), classes=det_classes, **common)
        else:
            self.pose = YoloModel(make_runtime(kind, b.get("pose", "models/yolo11n-pose"), **kw_pose), "pose",
                                  conf=b.get("pose_conf", 0.2))
            self.detect = YoloModel(make_runtime(kind, b.get("detect", "models/yolo11n"), **kw_det), "detect",
                                    conf=b.get("detect_conf", 0.25), classes=det_classes)
        # у каждой модели своё соединение/сессия — блокируем по модели, а позу и детектор можно гонять параллельно
        serial = kind in ("linq", "remote", "rabbit", "relay")
        self.locks = {id(self.pose): threading.Lock(), id(self.detect): threading.Lock()} if serial else {}
        self.parallel = kind in ("relay", "rabbit", "remote")   # сетевые бэкенды: ждём оба ответа одновременно
        if self.parallel:
            from concurrent.futures import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(max_workers=4)

    def run(self, model, frame):
        lk = self.locks.get(id(model))
        if lk:
            with lk:
                return model(frame), dict(model.last_timing)
        return model(frame), dict(model.last_timing)


# ------------------------------------------------------------------ источник кадров
class FrameSource(threading.Thread):
    """Держит самый свежий кадр. Файлы проигрываются в реальном темпе (как живая камера)."""

    def __init__(self, src, loop=True, fps=None):
        super().__init__(daemon=True)
        self.src, self.loop, self.force_fps = src, loop, fps
        self.latest = None   # (frame, ts, idx)
        self.cond = threading.Condition()
        self.stopped = False
        self.idx = 0
        self.src_fps = 0.0

    def _images(self):
        files = sorted(sum([glob.glob(os.path.join(self.src, e)) for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp")], []))
        fps = self.force_fps or 2.0
        self.src_fps = fps
        while not self.stopped:
            for f in files:
                img = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    self._push(img)
                time.sleep(1.0 / fps)
            if not self.loop:
                break

    def _push(self, frame):
        with self.cond:
            self.idx += 1
            self.latest = (frame, time.time(), self.idx)
            self.cond.notify_all()

    def run(self):
        if isinstance(self.src, str) and os.path.isdir(self.src):
            return self._images()
        src = int(self.src) if str(self.src).isdigit() else self.src
        while not self.stopped:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                log.error("не открыть источник %s, повтор через 3 с", self.src)
                time.sleep(3)
                continue
            is_file = isinstance(src, str) and os.path.isfile(src)
            fps = self.force_fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.src_fps = fps
            t0, n = time.time(), 0
            while not self.stopped:
                ok, frame = cap.read()
                if not ok:
                    break
                n += 1
                self._push(frame)
                if is_file:  # темп реального времени
                    delay = t0 + n / fps - time.time()
                    if delay > 0:
                        time.sleep(delay)
            cap.release()
            if not self.loop and is_file:
                break

    def get(self, after_idx: int, timeout=1.0):
        with self.cond:
            if self.latest is None or self.latest[2] <= after_idx:
                self.cond.wait(timeout)
            return self.latest


# ------------------------------------------------------------------ рисование
from PIL import ImageDraw, ImageFont

_FONT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "DejaVuSans-Bold.ttf")
_FONTS: Dict[int, object] = {}


def _font(size: int):
    if size not in _FONTS:
        try:
            _FONTS[size] = ImageFont.truetype(_FONT_PATH, size)
        except Exception:
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def put_text(img, text: str, org, color, size: int = 18):
    """Текст с кириллицей: рисуем через PIL только в маленьком прямоугольнике под надписью."""
    f = _font(size)
    x, y = int(org[0]), int(org[1])
    l, t, r, b = f.getbbox(text)
    w, h = r - l + 8, b - t + 6
    y = max(0, y - h)
    x = max(0, min(x, img.shape[1] - w))
    roi = img[y:y + h, x:x + w]
    if roi.size == 0:
        return
    bg = np.zeros_like(roi)
    bg[:] = (20, 20, 20)
    cv2.addWeighted(bg, 0.6, roi, 0.4, 0, roi)
    pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text((4 - l, 3 - t), text, font=f, fill=(int(color[2]), int(color[1]), int(color[0])))
    img[y:y + h, x:x + w] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def draw(frame, persons: List[Det], objects: List[Det], zones: List[Zone], now, alarms: Dict[int, str]):
    H, W = frame.shape[:2]
    over = frame.copy()
    labels = []
    for z in zones:
        if len(z.polygon) < 3:
            continue
        pts = np.array([[p[0] * W, p[1] * H] for p in z.polygon], np.int32)
        act = z.active(now)
        col = (40, 40, 230) if (z.kind == "danger" and act) else (160, 160, 160)
        if z.kind == "danger" and act:
            cv2.fillPoly(over, [pts], col)
        cv2.polylines(frame, [pts], True, col, 2)
        labels.append((z.name + ("" if act else " (не активна)"), tuple(pts[pts[:, 1].argmin()]), col))
    cv2.addWeighted(over, 0.25, frame, 0.75, 0, frame)
    th = max(2, W // 640)
    fs = max(14, W // 70)
    for o in objects:
        x1, y1, x2, y2 = o.box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), th)
        put_text(frame, f"{o.label} {o.score:.2f}", (x1, y1), (255, 200, 0), fs)
    for d in persons:
        sev = alarms.get(d.track_id)
        col = SEV_COLOR.get(sev, (80, 220, 80))
        x1, y1, x2, y2 = d.box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, th + (1 if sev else 0))
        if d.kpts is not None:
            for a, b in SKELETON:
                if d.kpts[a, 2] > 0.35 and d.kpts[b, 2] > 0.35:
                    cv2.line(frame, tuple(d.kpts[a, :2].astype(int)), tuple(d.kpts[b, :2].astype(int)), col, max(1, th - 1))
        lab = f"#{d.track_id}"
        if d.extra.get("lying"):
            lab += " лежит"
        if d.extra.get("speed", 0) > 1.0:
            lab += f" {d.extra['speed']:.1f}р/с"
        put_text(frame, lab, (x1, y1), col, fs)
    for text, org, col in labels:
        put_text(frame, text, org, col, fs)
    return frame


def highlight(frame, det: Det, title: str, color=(40, 40, 230)):
    img = frame.copy()
    x1, y1, x2, y2 = det.box.astype(int)
    th = max(3, img.shape[1] // 300)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, th)
    return img


# ------------------------------------------------------------------ камера
class CameraWorker(threading.Thread):
    def __init__(self, cam: dict, models: Models, on_open: Callable, on_close: Callable, media_dir: str):
        super().__init__(daemon=True, name=f"cam-{cam['id']}")
        self.cam = cam
        self.models = models
        self.media_dir = media_dir
        self._on_open_cb, self._on_close_cb = on_open, on_close
        self.source = FrameSource(cam["source"], cam.get("loop", True), cam.get("fps"))
        self.persons_tr = ByteTrackLite(high=0.45, low=0.2)
        self.objects_tr = ByteTrackLite(high=0.3, low=0.15, min_hits=1, max_age_s=1.5)
        self.history = TrackHistory()
        self.obj_speed_hist: Dict[int, deque] = {}
        self.engine = EventEngine(self._on_open, self._on_close)
        self.rules = []
        self.zones: List[Zone] = []
        self.apply_config(cam)
        self.jpeg: Optional[bytes] = None
        self.raw_jpeg: Optional[bytes] = None
        self.ring: deque = deque(maxlen=40)    # аннотированные кадры для GIF-фрагмента
        self.metrics = dict(fps=0.0, latency_ms=0.0, pose_ms=0.0, detect_ms=0.0, frames=0, persons=0,
                            backend=models.kind, pose_timing={}, detect_timing={})
        self.alarms: Dict[int, str] = {}
        self.stopped = False
        self.ring_ts: deque = deque(maxlen=40)
        self._n = 0
        self._objects: List[Det] = []
        self._persons: List[Det] = []
        self._pose_t: dict = {}
        self._last_ring = 0.0

    # конфиг можно менять на лету из интерфейса
    def apply_config(self, cam: dict):
        self.cam = cam
        self.zones = [Zone(**z) for z in cam.get("zones", [])]
        rules = []
        for rtype, params in (cam.get("rules") or {}).items():
            params = dict(params or {})
            if not params.pop("enabled", True) or rtype not in REGISTRY:
                continue
            rules.append(REGISTRY[rtype](params))
        self.rules = rules

    def stop(self):
        self.stopped = True
        self.source.stopped = True

    # ---------------------------------------------------------- события
    def _on_open(self, rule, sig: Signal, ctx: FrameCtx, ev):
        self.alarms[sig.det.track_id] = rule.severity
        return self._on_open_cb(self, rule, sig, ctx, ev)

    def _on_close(self, ev):
        self._on_close_cb(self, ev)

    def save_evidence(self, eid: int, frame, det: Det, title: str):
        os.makedirs(self.media_dir, exist_ok=True)
        snap = os.path.join(self.media_dir, f"ev{eid}.jpg")
        ok, buf = cv2.imencode(".jpg", highlight(frame, det, title), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:  # cv2.imwrite не умеет кириллицу в путях Windows
            with open(snap, "wb") as f:
                f.write(buf.tobytes())
        frames = list(self.ring)
        clip = None
        if len(frames) >= 4:
            clip = os.path.join(self.media_dir, f"ev{eid}.gif")
            step = max(1, len(frames) // 20)
            ims = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames[::step]]
            dur = int(1000 * (self._ring_span() / max(1, len(ims))))
            ims[0].save(clip, save_all=True, append_images=ims[1:], duration=max(60, dur), loop=0, optimize=True)
        return snap, clip

    def _ring_span(self):
        return self.ring_ts[-1] - self.ring_ts[0] if len(getattr(self, "ring_ts", [])) > 1 else 2.0

    # ---------------------------------------------------------- обработка одного кадра
    def process(self, frame, ts: float, render: bool = True):
        self._n += 1
        n = self._n
        every = int(self.cam.get("detect_every", 2))
        need_det = every <= 1 or n % every == 1
        M = self.models
        if need_det and M.parallel:
            fd = M.pool.submit(M.run, M.detect, frame)
            persons, self._pose_t = M.run(M.pose, frame)
            raw, det_t = fd.result()
        else:
            persons, self._pose_t = M.run(M.pose, frame)
            raw, det_t = (M.run(M.detect, frame) if need_det else (None, None))
        if need_det:
            objs = [d for d in raw if d.label != "person"]
            self._objects = self.objects_tr.update(objs, ts)
            self._object_speed(self._objects, ts)
            self.metrics["detect_ms"] = det_t.get("total_ms", 0)
            self.metrics["detect_timing"] = det_t
        objects = self._objects
        persons = self.persons_tr.update(persons, ts)
        fall_p = (self.cam.get("rules") or {}).get("fall") or {}
        for d in persons:
            d.extra["lying"] = is_lying(d, fall_p.get("angle_thr", 55.0), fall_p.get("aspect_thr", 1.25))
        self.history.update(persons, ts)
        now = datetime.fromtimestamp(ts)
        H, W = frame.shape[:2]
        ctx = FrameCtx(self.cam["id"], ts, now, W, H, persons, objects, self.zones, self.history)
        self._cur_frame = frame
        for rule in self.rules:
            self.engine.step(rule, rule.signals(ctx), ctx)
        live = {k[1].split("@")[0] for k in self.engine.open}
        self.alarms = {tid: s for tid, s in self.alarms.items() if str(tid) in live}
        self._persons = persons
        if not render:
            return persons, objects
        vis = draw(frame.copy(), persons, objects, self.zones, now, self.alarms)
        small = cv2.resize(vis, (960, int(960 * H / W))) if W > 960 else vis
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self.jpeg = buf.tobytes()
        if ts - self._last_ring >= 0.2:  # 5 к/с для фрагмента-доказательства
            g = cv2.resize(vis, (480, int(480 * H / W)))
            self.ring.append(g)
            self.ring_ts.append(ts)
            self._last_ring = ts
        if n % 10 == 1:
            rs = cv2.resize(frame, (960, int(960 * H / W))) if W > 960 else frame
            ok, b2 = cv2.imencode(".jpg", rs, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.raw_jpeg = b2.tobytes()
        return persons, objects

    # ---------------------------------------------------------- цикл
    def run(self):
        self.source.start()
        last_idx = 0
        ema = None
        while not self.stopped:
            item = self.source.get(last_idx)
            if item is None or item[2] == last_idx:
                continue
            frame, ts, last_idx = item
            t_start = time.time()
            try:
                self.process(frame, ts)
            except Exception:
                log.exception("ошибка обработки кадра")
                time.sleep(0.2)
                continue
            done = time.time()
            dt = done - t_start
            ema = dt if ema is None else 0.9 * ema + 0.1 * dt
            m = self.metrics
            m["fps"] = 1.0 / ema if ema else 0.0
            m["latency_ms"] = (done - ts) * 1000     # от получения кадра до готового решения
            m["pose_ms"] = self._pose_t.get("total_ms", 0)
            m["pose_timing"] = self._pose_t
            m["frames"] += 1
            m["persons"] = len(self._persons)
            m["src_fps"] = self.source.src_fps

    def _object_speed(self, objects: List[Det], t: float):
        for o in objects:
            q = self.obj_speed_hist.setdefault(o.track_id, deque(maxlen=15))
            q.append((t, o.center, o.w))
            if len(q) >= 2 and q[-1][0] - q[0][0] > 0.3:
                (t0, c0, w0), (t1, c1, _) = q[0], q[-1]
                o.extra["speed"] = float(np.hypot(c1[0] - c0[0], c1[1] - c0[1]) / (t1 - t0) / max(w0, 1.0))
        alive = {o.track_id for o in objects}
        for k in list(self.obj_speed_hist):
            if k not in alive and self.obj_speed_hist[k][-1][0] < t - 5:
                del self.obj_speed_hist[k]
