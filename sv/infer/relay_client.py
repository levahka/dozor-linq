"""Клиент посредника linq/relay.py: кадр -> JPEG -> RabbitMQ -> (стенд: TPU + декодирование) -> готовые рамки.

Используется как замена YoloModel в режиме backend.kind: relay.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .yolo import COCO_NAMES, Det


class RelayYolo:
    def __init__(self, task: str, host: str, login: str, conf: float = 0.25,
                 classes: Optional[Sequence[int]] = None, timeout: float = 10.0,
                 send_size: int = 640, quality: int = 80, names: Sequence[str] = COCO_NAMES):
        import pika  # noqa: F401

        self.task = task
        self.model_key = "pose" if task == "pose" else "detect"
        self.host, self.login = host, login
        self.conf, self.classes = conf, list(classes) if classes is not None else None
        self.timeout, self.send_size, self.quality = timeout, send_size, quality
        self.names = list(names) if task == "detect" else ["person"]
        self._tl = threading.local()     # своё соединение у каждого потока: камеры шлют кадры параллельно (на стенде 16 TPU)
        self._timing = threading.local()
        self._connect()

    @property
    def last_timing(self):
        return getattr(self._timing, "v", {})

    def _connect(self):
        import pika

        tl = self._tl
        tl.conn = pika.BlockingConnection(pika.ConnectionParameters(
            self.host, 5672, credentials=pika.PlainCredentials("guest", "guest"),
            heartbeat=600, blocked_connection_timeout=300))
        tl.ch = tl.conn.channel()
        tl.reply = f"dozor_r_{uuid.uuid4().hex[:10]}"
        tl.ch.queue_declare(queue=tl.reply, auto_delete=True)
        tl.ch.queue_bind(exchange="tpu.direct", queue=tl.reply, routing_key=tl.reply)
        tl.res = {}

        def on_msg(ch, m, p, body):
            try:
                d = json.loads(body)
                tl.res[d.get("id")] = d
            except Exception:
                pass

        tl.ch.basic_consume(queue=tl.reply, on_message_callback=on_msg, auto_ack=True)

    def _call(self, frame):
        import pika

        tl = self._tl
        if not hasattr(tl, "conn"):
            self._connect()
        H, W = frame.shape[:2]
        s = min(1.0, self.send_size / max(H, W))
        small = cv2.resize(frame, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        rid = uuid.uuid4().hex
        hdr = {"dozor_model": self.model_key, "dozor_reply": tl.reply, "dozor_id": rid,
               "dozor_conf": str(self.conf), "dozor_w": str(W), "dozor_h": str(H)}
        if self.classes is not None:
            hdr["dozor_classes"] = json.dumps(self.classes)
        t0 = time.perf_counter()
        tl.ch.basic_publish(exchange="tpu.direct", routing_key=f"dozor_{self.login}", body=jpg.tobytes(),
                            properties=pika.BasicProperties(headers=hdr))
        end = time.monotonic() + self.timeout
        while rid not in tl.res:
            if time.monotonic() > end:
                raise TimeoutError("посредник на стенде LinQ не ответил (запущен ли linq/relay.py?)")
            tl.conn.process_data_events(time_limit=0.003)
        d = tl.res.pop(rid)
        if "error" in d:
            raise RuntimeError(f"посредник: {d['error']}")
        rt = (time.perf_counter() - t0) * 1000
        t = d.get("t", {})
        self._timing.v = {"total_ms": rt, "net_ms": rt, "head_ms": t.get("tpu_infer_ms", 0.0),
                            "tpu_rt_ms": t.get("tpu_rt_ms", 0.0), "relay_ms": t.get("relay_ms", 0.0),
                            "network_ms": rt - t.get("relay_ms", 0.0), "jpeg_kb": len(jpg) / 1024}
        return d["dets"]

    def __call__(self, frame) -> List[Det]:
        try:
            rows = self._call(frame)
        except (TimeoutError, RuntimeError):
            raise
        except Exception:
            self._connect()
            rows = self._call(frame)
        out = []
        for r in rows:
            c = int(r[5])
            k = np.array(r[6:], np.float32).reshape(17, 3) if self.task == "pose" and len(r) > 6 else None
            out.append(Det(np.array(r[:4], np.float32), float(r[4]), c,
                           self.names[c] if c < len(self.names) else str(c), k))
        return out
