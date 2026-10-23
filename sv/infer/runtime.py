"""Сменные бэкенды инференса.

Все бэкенды принимают тензор float32 NCHW [1,3,H,W] (0..1) и возвращают «output0» YOLO.
Пре- и постобработка (letterbox, декодирование, NMS) общие — см. yolo.py.

  cpu    — полная ONNX-модель в onnxruntime (эталон, работает везде).
  split  — head.onnx + tail.onnx в onnxruntime (проверка разреза без TPU).
  linq   — head скомпилирован в .tpu и выполняется на LinQ TPU через pytpu,
           tail (декодер головы) — в onnxruntime на CPU.
  remote — то же, что linq, но TPU стоит на другой машине: head считается
           через HTTP-сервис linq/tpu_server.py (запущен рядом с ускорителем).
"""
from __future__ import annotations

import io
import json
import os
import time
from typing import Dict, List

import numpy as np
import onnxruntime as ort


def _ort_session(path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


class Runtime:
    name = "base"
    imgsz = 640

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # время последнего вызова по стадиям, мс — для страницы «Производительность»
    last_timing: Dict[str, float]


class OrtRuntime(Runtime):
    name = "cpu"

    def __init__(self, model_path: str):
        self.sess = _ort_session(model_path)
        self.inp = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.imgsz = shape[2] if isinstance(shape[2], int) else 640
        self.last_timing = {}

    def __call__(self, x):
        t = time.perf_counter()
        out = self.sess.run(None, {self.inp: x})[0]
        self.last_timing = {"net_ms": (time.perf_counter() - t) * 1000}
        return out


class _SplitBase(Runtime):
    """head (что-то) + tail (onnxruntime)."""

    def __init__(self, base_path: str):
        meta_path = base_path + ".split.json"
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.imgsz = self.meta["imgsz"]
        self.cuts: List[str] = self.meta["head_outputs"]
        self.tail = _ort_session(base_path + ".tail.onnx")
        self.last_timing = {}

    def head(self, x) -> List[np.ndarray]:
        raise NotImplementedError

    def __call__(self, x):
        t0 = time.perf_counter()
        h = self.head(x)
        t1 = time.perf_counter()
        out = self.tail.run(None, dict(zip(self.cuts, h)))[0]
        t2 = time.perf_counter()
        self.last_timing = {"head_ms": (t1 - t0) * 1000, "tail_ms": (t2 - t1) * 1000, "net_ms": (t2 - t0) * 1000}
        return out


class SplitOrtRuntime(_SplitBase):
    name = "split"

    def __init__(self, base_path: str):
        super().__init__(base_path)
        self.head_sess = _ort_session(base_path + ".head.onnx")
        self.inp = self.head_sess.get_inputs()[0].name

    def head(self, x):
        return self.head_sess.run(self.cuts, {self.inp: x})


class LinqRuntime(_SplitBase):
    """Head на LinQ TPU (pytpu), tail — CPU.

    tpu_program: путь к .tpu, собранному tools/compile_linq.py из <base>.head.onnx.
    Имена входа/выходов в .tpu совпадают с именами тензоров в head.onnx.
    """

    name = "linq"

    def __init__(self, base_path: str, tpu_program: str, device_index: int = 0):
        super().__init__(base_path)
        import pytpu as tpu  # есть только на машине с ускорителем

        devices = tpu.Device.list_devices()
        if not devices:
            raise RuntimeError("TPU не найден (ls /dev/tpu*)")
        self._dev_cm = tpu.Device.open(devices[device_index])
        self._dev = self._dev_cm.__enter__()
        self._prog_cm = self._dev.load(tpu_program)
        self._prog = self._prog_cm.__enter__()
        self._inf_cm = self._prog.inference()
        self._inf = self._inf_cm.__enter__()
        self.inp = self.meta["input"]

    def head(self, x):
        res = self._inf.run({self.inp: x})
        return [np.asarray(res[c], dtype=np.float32).reshape(s) for c, s in zip(self.cuts, self.meta["head_output_shapes"])]

    def close(self):
        for cm in (self._inf_cm, self._prog_cm, self._dev_cm):
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass


class RemoteLinqRuntime(_SplitBase):
    """Head считается на удалённой машине с TPU через linq/tpu_server.py (HTTP, npz)."""

    name = "remote"

    def __init__(self, base_path: str, url: str, model_key: str):
        super().__init__(base_path)
        import requests

        self.http = requests.Session()
        self.url = url.rstrip("/")
        self.model_key = model_key
        self.inp = self.meta["input"]

    def head(self, x):
        buf = io.BytesIO()
        np.save(buf, x.astype(np.float32), allow_pickle=False)
        r = self.http.post(f"{self.url}/infer/{self.model_key}", data=buf.getvalue(), timeout=10)
        r.raise_for_status()
        z = np.load(io.BytesIO(r.content), allow_pickle=False)
        self.remote_tpu_ms = float(r.headers.get("X-TPU-ms", "nan"))
        return [z[c].astype(np.float32) for c in self.cuts]


class RabbitLinqRuntime(_SplitBase):
    """Head на облачном LinQ TPU организаторов через RabbitMQ (протокол из examples/example.ipynb).

    Программа .tpu должна лежать на сервере в /auto/cache/<login>/.
    Нужны пакеты: pika и tpu_pycloud_format (whl из папки packages в Jupyter).
    """

    name = "rabbit"

    def __init__(self, base_path: str, tpu_program: str, host: str = "10.0.114.30", login: str = "",
                 port: int = 5672, user: str = "guest", password: str = "guest", timeout: float = 10.0,
                 exchange: str = "tpu.direct", input_queue: str = "input", raw: bool = True):
        super().__init__(base_path)
        import pika  # noqa: F401
        import tpu_pycloud_format as f

        self.f = f
        self.host, self.port, self.cred = host, port, (user, password)
        self.exchange, self.input_queue, self.timeout = exchange, input_queue, timeout
        prog = os.path.basename(tpu_program)
        # raw=True: гоняем по сети тензоры в родном формате TPU (int8, раскладка из metadata.json) —
        # в 3-4 раза меньше данных, чем float32. Проверено: косинус с float-моделью 0.985-0.9995.
        self.raw = raw
        self.target = f"file:///auto/cache/{login}/{prog}?raw={'True' if raw else 'False'}"
        self.inp = self.meta["input"]
        if raw:
            import zipfile

            with zipfile.ZipFile(tpu_program) as z:
                md = json.loads(z.read("metadata.json"))
            self.md_in = {k: v for d in md["inputs"].values() for k, v in d.items()}
            self.md_out = {k: v for d in md["outputs"].values() for k, v in d.items()}
            self.word = int(md.get("hardware_parameters", {}).get("ddr_word_len", 32))
        self.worker_info = {}
        self._connect()

    def _connect(self):
        import uuid

        import pika

        self.conn = pika.BlockingConnection(pika.ConnectionParameters(
            self.host, self.port, credentials=pika.PlainCredentials(*self.cred),
            heartbeat=600, blocked_connection_timeout=300))
        self.ch = self.conn.channel()
        self.reply_q = f"dozor_{uuid.uuid4().hex[:10]}"
        self.ch.queue_declare(queue=self.reply_q, auto_delete=True)
        self.ch.queue_bind(exchange=self.exchange, queue=self.reply_q, routing_key=self.reply_q)
        self.responses = {}

        def on_msg(ch, method, props, body):
            mid = (props.headers or {}).get(self.f.MESSAGE_ID_ATTR_NAME)
            self.responses[mid] = body

        self.ch.basic_consume(queue=self.reply_q, on_message_callback=on_msg, auto_ack=True)

    def _roundtrip(self, x):
        import uuid

        import pika

        f = self.f
        tid = uuid.uuid4().hex
        payload = {self.inp: self._encode(x)} if self.raw else {self.inp: x}
        self.ch.basic_publish(
            exchange=self.exchange, routing_key=self.input_queue,
            body=f.serialize(f.PyTask(payload)),
            properties=pika.BasicProperties(headers={
                f.TARGET_URI_ATTR_NAME: self.target,
                f.RESPONSE_ROUTING_KEYS_ATTR_NAME: json.dumps([self.reply_q]),
                f.MESSAGE_ID_ATTR_NAME: tid,
            }))
        t_end = time.monotonic() + self.timeout
        while tid not in self.responses:
            if time.monotonic() > t_end:
                raise TimeoutError(f"LinQ не ответил за {self.timeout} с ({self.target})")
            self.conn.process_data_events(time_limit=0.005)
        data, metadata, worker = f.deserialize(self.responses.pop(tid))
        self.worker_info = dict(worker or {})
        if not data:
            raise RuntimeError(f"ошибка воркера LinQ: {self.worker_info.get('error_message')}")
        return data

    def head(self, x):
        try:
            data = self._roundtrip(x)
        except (TimeoutError, RuntimeError):
            raise
        except Exception:  # обрыв соединения — переподключаемся один раз
            self._connect()
            data = self._roundtrip(x)
        if self.raw:
            return [self._decode(c, data[c]) for c in self.cuts]
        return [np.asarray(data[c], dtype=np.float32).reshape(s)
                for c, s in zip(self.cuts, self.meta["head_output_shapes"])]

    # ---- формат TPU: [C/B, H, W, N, B] int8, квантование q = x*scale, ширина/каналы с выравниванием
    def _encode(self, x):
        v = self.md_in[self.inp]
        ts = v["tpu_shape"]                      # [Cg, H, W, N, B]
        B = ts[-1]
        N, C, H, W = x.shape
        Cp, Wp = ts[0] * B, ts[2]
        xp = np.zeros((N, Cp, H, Wp), np.float32)
        xp[:, :C, :, :W] = x
        q = np.clip(np.round(xp * v["scales"][0] + v["zero_points"][0]), -128, 127).astype(np.int8)
        q = q.reshape(N, ts[0], B, H, Wp).transpose(1, 3, 4, 0, 2)
        return np.ascontiguousarray(q).view(np.uint8).reshape(-1, self.word)

    def _decode(self, name, buf):
        v = self.md_out[name]
        ts = v["tpu_shape"]
        a = np.asarray(buf).view(np.int8).ravel()[: int(np.prod(ts))].reshape(ts)
        a = a.transpose(3, 0, 4, 1, 2)          # [N, Cg, B, H, Wp]
        N, Cg, B, H, Wp = a.shape
        us = v["user_shape"]
        a = a.reshape(N, Cg * B, H, Wp)[: us[0], : us[1], : us[2], : us[3]]
        return (a.astype(np.float32) - v["zero_points"][0]) * v["scales"][0]


def make_runtime(kind: str, model_base: str, **kw) -> Runtime:
    """model_base — путь без расширения, например models/yolo11n-pose."""
    if kind == "cpu":
        p = model_base + ".static.onnx"
        return OrtRuntime(p if os.path.exists(p) else model_base + ".onnx")
    if kind == "split":
        return SplitOrtRuntime(model_base)
    if kind == "linq":
        return LinqRuntime(model_base, kw["tpu_program"], kw.get("device_index", 0))
    if kind == "remote":
        return RemoteLinqRuntime(model_base, kw["url"], kw["model_key"])
    if kind == "rabbit":
        return RabbitLinqRuntime(model_base, **kw)
    raise ValueError(f"неизвестный бэкенд {kind}")
