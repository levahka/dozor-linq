"""Микросервис инференса на машине с LinQ TPU (режим backend.kind: remote).

Ноутбук/сервер с камерами шлёт тензор, сервис отвечает выходами head-сети.
Только стандартная библиотека + numpy + pytpu — ставить ничего не нужно.

    python linq/tpu_server.py --port 8765 \
        --pose models/yolo11n-pose.head_b1_o5_h1m.tpu --detect models/yolo11n.head_b1_o5_h1m.tpu

POST /infer/<pose|detect>   тело: np.save(float32 [1,3,640,640])  ->  ответ: np.savez(выходы)
GET  /health                -> json со временем инференса
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


class Engine:
    def __init__(self, programs: dict, input_name: str = "images"):
        import pytpu as tpu

        self.input_name = input_name

        dev = tpu.Device.list_devices()[0]
        self.dev = tpu.Device.open(dev).__enter__()
        self.sessions = {}
        for key, path in programs.items():
            if not path:
                continue
            prog = self.dev.load(path).__enter__()
            self.sessions[key] = prog.inference().__enter__()
        self.lock = threading.Lock()
        self.stats = {k: [] for k in self.sessions}

    def run(self, key, x):
        with self.lock:
            t = time.perf_counter()
            inf = self.sessions[key]
            out = inf.run({self.input_name: x})
            ms = (time.perf_counter() - t) * 1000
        s = self.stats[key]
        s.append(ms)
        del s[:-200]
        return out, ms


def make_handler(engine: Engine):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({k: {"n": len(v), "mean_ms": float(np.mean(v)) if v else None}
                               for k, v in engine.stats.items()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            key = self.path.rstrip("/").split("/")[-1]
            if key not in engine.sessions:
                self.send_error(404, "unknown model")
                return
            n = int(self.headers.get("Content-Length", 0))
            x = np.load(io.BytesIO(self.rfile.read(n)), allow_pickle=False)
            out, ms = engine.run(key, x)
            buf = io.BytesIO()
            np.savez(buf, **{k: np.asarray(v, np.float32) for k, v in out.items()})
            data = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-TPU-ms", f"{ms:.3f}")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--pose")
    ap.add_argument("--detect")
    ap.add_argument("--input", default="images", help="имя входа (см. models/*.split.json)")
    a = ap.parse_args()
    eng = Engine({"pose": a.pose, "detect": a.detect}, a.input)
    print("TPU-сервис на порту", a.port, "модели:", list(eng.sessions))
    ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(eng)).serve_forever()


if __name__ == "__main__":
    main()
