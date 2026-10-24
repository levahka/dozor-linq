"""Посредник ДОЗОР на сервере стенда LinQ (запускается в терминале Jupyter).

Зачем: по VPN до стенда медленно, а тензор кадра 640x640 весит 1,6 МБ даже в int8, ответ ~1 МБ.
Посредник принимает от ноутбука JPEG (~50-80 КБ), сам делает предобработку, отправляет кадр на TPU
(в той же сети, быстро), декодирует голову YOLO + NMS на numpy и возвращает готовые рамки (~1-3 КБ JSON).

Запуск (терминал Jupyter):
    nohup python relay.py --login hackathon27139 > relay.log 2>&1 &
    tail -f relay.log

Зависимости — только то, что уже есть в окружении Jupyter: numpy, Pillow, pika, tpu_pycloud_format.
Протокол с ноутбуком: exchange tpu.direct, очередь dozor_<login>.
  запрос: тело = JPEG, заголовки dozor_model (pose|detect), dozor_reply (имя очереди ответа), dozor_id, dozor_conf
  ответ:  JSON {"id", "dets": [[x1,y1,x2,y2,score,cls,(kx,ky,kc)*17]...], "t": {...}}
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
import uuid
import zipfile

import numpy as np
from PIL import Image

STRIDES = (8, 16, 32)


# ------------------------------------------------------------------ формат TPU (raw)
class TpuIO:
    def __init__(self, program_path):
        with zipfile.ZipFile(program_path) as z:
            md = json.loads(z.read("metadata.json"))
        self.ins = {k: v for d in md["inputs"].values() for k, v in d.items()}
        self.outs = {k: v for d in md["outputs"].values() for k, v in d.items()}
        self.word = int(md.get("hardware_parameters", {}).get("ddr_word_len", 32))
        self.in_name = next(iter(self.ins))

    def encode(self, x):
        v = self.ins[self.in_name]
        ts = v["tpu_shape"]
        B = ts[-1]
        N, C, H, W = x.shape
        xp = np.zeros((N, ts[0] * B, H, ts[2]), np.float32)
        xp[:, :C, :, :W] = x
        q = np.clip(np.round(xp * v["scales"][0] + v["zero_points"][0]), -128, 127).astype(np.int8)
        q = q.reshape(N, ts[0], B, H, ts[2]).transpose(1, 3, 4, 0, 2)
        return np.ascontiguousarray(q).view(np.uint8).reshape(-1, self.word)

    def decode(self, name, buf):
        v = self.outs[name]
        ts = v["tpu_shape"]
        a = np.asarray(buf).view(np.int8).ravel()[: int(np.prod(ts))].reshape(ts).transpose(3, 0, 4, 1, 2)
        N, Cg, B, H, Wp = a.shape
        us = v["user_shape"]
        a = a.reshape(N, Cg * B, H, Wp)[: us[0], : us[1], : us[2], : us[3]]
        return (a.astype(np.float32) - v["zero_points"][0]) * v["scales"][0]


# ------------------------------------------------------------------ декодер головы YOLOv8/11 на numpy
def head_outputs_by_scale(outs: dict):
    """Группирует выходы head по шкалам: {i: {'box': [64,H,W], 'cls': [nc,H,W], 'kpt': [51,H,W]}}."""
    res = {}
    for name, a in outs.items():
        # /model.23/cv2.0/cv2.0.2/Conv_output_0
        part = name.split("/")[2]           # cv2.0
        branch, idx = part.split(".")
        key = {"cv2": "box", "cv3": "cls", "cv4": "kpt"}[branch]
        res.setdefault(int(idx), {})[key] = a[0]
    return res


def decode_yolo(outs: dict, task: str, conf: float):
    boxes, scores, classes, kpts = [], [], [], []
    for i, d in sorted(head_outputs_by_scale(outs).items()):
        s = STRIDES[i]
        box, cls = d["box"], d["cls"]
        _, H, W = box.shape
        logits = cls.reshape(cls.shape[0], -1)
        prob = 1.0 / (1.0 + np.exp(-logits))
        c = prob.argmax(0)
        p = prob[c, np.arange(prob.shape[1])]
        m = p >= conf
        if not m.any():
            continue
        idx = np.nonzero(m)[0]
        b = box.reshape(4, 16, -1)[:, :, idx]
        b = np.exp(b - b.max(1, keepdims=True))
        b = (b / b.sum(1, keepdims=True) * np.arange(16)[None, :, None]).sum(1)   # [4, n] l,t,r,b
        gy, gx = np.divmod(idx, W)
        ax, ay = gx + 0.5, gy + 0.5
        xyxy = np.stack([ax - b[0], ay - b[1], ax + b[2], ay + b[3]], 1) * s
        boxes.append(xyxy)
        scores.append(p[idx])
        classes.append(c[idx])
        if task == "pose":
            k = d["kpt"].reshape(17, 3, -1)[:, :, idx]          # [17,3,n]
            kx = (k[:, 0] * 2.0 + gx) * s
            ky = (k[:, 1] * 2.0 + gy) * s
            kc = 1.0 / (1.0 + np.exp(-k[:, 2]))
            kpts.append(np.stack([kx, ky, kc], 2).transpose(1, 0, 2))   # [n,17,3]
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), None
    return (np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes),
            np.concatenate(kpts) if kpts else None)


def nms(boxes, scores, iou=0.5):
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
        order = order[1:][inter / (areas[i] + areas[order[1:]] - inter + 1e-9) <= iou]
    return np.array(keep, int)


def letterbox(img: Image.Image, size=640):
    w, h = img.size
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR), (dx, dy))
    x = np.asarray(canvas, np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x), r, dx, dy


def postprocess(outs, task, conf, r, dx, dy, W, H, classes=None):
    boxes, scores, cls, kpts = decode_yolo(outs, task, conf)
    if classes is not None and len(cls):
        m = np.isin(cls, classes)
        boxes, scores, cls = boxes[m], scores[m], cls[m]
        kpts = kpts[m] if kpts is not None else None
    if not len(scores):
        return []
    keep = nms(boxes + cls[:, None] * 4096.0, scores, 0.5)
    res = []
    for i in keep:
        b = boxes[i].copy()
        b[[0, 2]] = ((b[[0, 2]] - dx) / r).clip(0, W - 1)
        b[[1, 3]] = ((b[[1, 3]] - dy) / r).clip(0, H - 1)
        row = [round(float(v), 1) for v in b] + [round(float(scores[i]), 3), int(cls[i])]
        if kpts is not None:
            k = kpts[i].copy()
            k[:, 0] = (k[:, 0] - dx) / r
            k[:, 1] = (k[:, 1] - dy) / r
            row += [round(float(v), 2) for v in k.ravel()]
        res.append(row)
    return res


# ------------------------------------------------------------------ сервис
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--login", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--pose", default="yolo11n-pose.head_b1_o5_h1m.tpu")
    ap.add_argument("--detect", default="yolo11n.head_b1_o5_h1m.tpu")
    ap.add_argument("--workers", type=int, default=4, help="параллельных обработчиков (на стенде 16 TPU)")
    a = ap.parse_args()

    import pika
    from tpu_pycloud_format import (MESSAGE_ID_ATTR_NAME, RESPONSE_ROUTING_KEYS_ATTR_NAME,
                                    TARGET_URI_ATTR_NAME, PyTask, deserialize, serialize)

    cache = f"/auto/cache/{a.login}"
    progs = {"pose": os.path.join(cache, a.pose), "detect": os.path.join(cache, a.detect)}
    io_ = {k: TpuIO(p) for k, p in progs.items()}
    params = pika.ConnectionParameters(a.host, 5672, credentials=pika.PlainCredentials("guest", "guest"),
                                       heartbeat=600, blocked_connection_timeout=300)
    stats = {"n": 0}
    import threading

    threads = [threading.Thread(target=serve, args=(k, a, params, progs, io_, stats), daemon=True)
               for k in range(max(1, a.workers))]
    for t in threads:
        t.start()
    print(f"Посредник ДОЗОР запущен: {len(threads)} обработчиков, очередь dozor_{a.login}, программы {progs}", flush=True)
    for t in threads:
        t.join()


def serve(k, a, params, progs, io_, stats):
    import pika
    from tpu_pycloud_format import (MESSAGE_ID_ATTR_NAME, RESPONSE_ROUTING_KEYS_ATTR_NAME,
                                    TARGET_URI_ATTR_NAME, PyTask, deserialize, serialize)

    # соединение для TPU (синхронные вызовы)
    tconn = pika.BlockingConnection(params)
    tch = tconn.channel()
    tq = f"dozor_tpu_{uuid.uuid4().hex[:8]}"
    tch.queue_declare(queue=tq, auto_delete=True)
    tch.queue_bind(exchange="tpu.direct", queue=tq, routing_key=tq)
    tres = {}
    tch.basic_consume(queue=tq, auto_ack=True,
                      on_message_callback=lambda ch, m, p, b: tres.__setitem__(p.headers.get(MESSAGE_ID_ATTR_NAME), b))

    def tpu(model, x):
        tid = uuid.uuid4().hex
        tch.basic_publish(exchange="tpu.direct", routing_key="input",
                          body=serialize(PyTask({io_[model].in_name: io_[model].encode(x)})),
                          properties=pika.BasicProperties(headers={
                              TARGET_URI_ATTR_NAME: f"file://{progs[model]}?raw=True",
                              RESPONSE_ROUTING_KEYS_ATTR_NAME: json.dumps([tq]),
                              MESSAGE_ID_ATTR_NAME: tid}))
        end = time.monotonic() + 30
        while tid not in tres:
            if time.monotonic() > end:
                raise TimeoutError("TPU не ответил")
            tconn.process_data_events(time_limit=0.002)
        data, _, worker = deserialize(tres.pop(tid))
        if not data:
            raise RuntimeError(worker.get("error_message"))
        return {k: io_[model].decode(k, v) for k, v in data.items()}, worker

    # соединение для запросов ноутбука
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    q = f"dozor_{a.login}"
    ch.queue_declare(queue=q, auto_delete=False, arguments={"x-message-ttl": 5000})
    ch.queue_bind(exchange="tpu.direct", queue=q, routing_key=q)
    ch.basic_qos(prefetch_count=1)

    def on_req(chan, method, props, body):
        h = props.headers or {}
        t0 = time.perf_counter()
        out = {"id": h.get("dozor_id")}
        try:
            model = h.get("dozor_model", "pose")
            conf = float(h.get("dozor_conf", 0.25))
            classes = json.loads(h["dozor_classes"]) if h.get("dozor_classes") else None
            img = Image.open(io.BytesIO(body))
            W, H = int(h.get("dozor_w", img.width)), int(h.get("dozor_h", img.height))
            sx, sy = W / img.width, H / img.height
            x, r, dx, dy = letterbox(img)
            t1 = time.perf_counter()
            outs, worker = tpu(model, x)
            t2 = time.perf_counter()
            dets = postprocess(outs, "pose" if model == "pose" else "detect", conf, r, dx, dy, img.width, img.height, classes)
            if sx != 1 or sy != 1:   # кадр мог быть уменьшен на ноутбуке — вернуть в исходные координаты
                for d in dets:
                    d[0] *= sx; d[2] *= sx; d[1] *= sy; d[3] *= sy
                    for j in range(6, len(d), 3):
                        d[j] *= sx; d[j + 1] *= sy
            t3 = time.perf_counter()
            out.update(dets=dets, t={"pre_ms": (t1 - t0) * 1000, "tpu_rt_ms": (t2 - t1) * 1000,
                                     "tpu_infer_ms": float(worker.get("inference_time", 0)) * 1000,
                                     "post_ms": (t3 - t2) * 1000, "relay_ms": (t3 - t0) * 1000})
        except Exception as e:  # noqa: BLE001
            out["error"] = repr(e)
        chan.basic_publish(exchange="tpu.direct", routing_key=h.get("dozor_reply", ""), body=json.dumps(out).encode())
        chan.basic_ack(method.delivery_tag)
        stats["n"] += 1
        if stats["n"] % 50 == 1:
            print(time.strftime("%H:%M:%S"), "запросов:", stats["n"], out.get("t") or out.get("error"), flush=True)

    ch.basic_consume(queue=q, on_message_callback=on_req, auto_ack=False)
    while True:
        conn.process_data_events(time_limit=0.05)
        tconn.process_data_events(time_limit=0)


if __name__ == "__main__":
    main()
