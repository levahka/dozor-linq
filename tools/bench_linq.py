"""Проверка и замер нейросетей на облачном LinQ TPU (через RabbitMQ) прямо с ноутбука.

    python tools/bench_linq.py                 # обе модели, 30 кадров из data/
    python tools/bench_linq.py --n 100

Что делает для каждой модели:
  * гоняет кадры через LinQ: время «туда-обратно» (сеть + очередь + TPU) — среднее, p50, p95, к/с;
  * сравнивает выходы TPU (INT8) с той же сетью на CPU (float32): косинус сходства;
  * сравнивает итоговые детекции TPU и CPU: сколько объектов нашлось и насколько совпадают рамки.
Результат — в linq_bench.json и на экран (это цифры для презентации).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from sv.infer.runtime import make_runtime  # noqa: E402
from sv.infer.yolo import YoloModel  # noqa: E402
from sv.tracker import iou_matrix  # noqa: E402


def frames_from(paths, n):
    out = []
    per = max(1, n // len(paths))
    for p in paths:
        cap = cv2.VideoCapture(p)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for i in np.linspace(0, total - 1, per).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                out.append(f)
    return out[:n]


def match_rate(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    m = iou_matrix(np.array([d.box for d in a]), np.array([d.box for d in b]))
    return float((m.max(1) > 0.5).sum() / max(len(a), len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--videos", nargs="*", default=["data/ceh1_worker_zone.mp4", "data/parking_car.mp4"])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    b = cfg["backend"]
    r = b.get("rabbit", {})
    frames = frames_from(a.videos, a.n)
    print(f"Кадров: {len(frames)}; сервер {r.get('host')}, логин {r.get('login')}")
    report = {"host": r.get("host"), "frames": len(frames), "models": {}}
    for key, task in (("pose", "pose"), ("detect", "detect")):
        base = b[key]
        print(f"\n=== {base} ({task})")
        cpu_rt = make_runtime("split", base)
        tpu_rt = make_runtime("rabbit", base, tpu_program=b["tpu_programs"][key], host=r.get("host", "10.0.114.30"),
                              login=r["login"], timeout=r.get("timeout", 15))
        cls = b.get("detect_classes") if task == "detect" else None
        conf = b.get("pose_conf", 0.2) if task == "pose" else b.get("detect_conf", 0.25)
        cpu_m = YoloModel(cpu_rt, task, conf=conf, classes=cls)
        tpu_m = YoloModel(tpu_rt, task, conf=conf, classes=cls)
        tpu_m(frames[0])  # прогрев (загрузка программы на TPU)
        rt, cos, match, n_cpu, n_tpu = [], [], [], 0, 0
        for f in frames:
            dc = cpu_m(f)
            t = time.perf_counter()
            dt = tpu_m(f)
            rt.append(tpu_m.last_timing.get("head_ms", (time.perf_counter() - t) * 1000))
            # косинус по сырым выходам head
            from sv.infer.yolo import letterbox
            x = letterbox(f, cpu_rt.imgsz)[0]
            hc = cpu_rt.head(x)
            ht = tpu_rt.head(x)
            for u, v in zip(hc, ht):
                u, v = u.ravel(), v.ravel()
                cos.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)))
            match.append(match_rate(dc, dt))
            n_cpu += len(dc)
            n_tpu += len(dt)
        rt = np.array(rt)
        res = {"roundtrip_mean_ms": float(rt.mean()), "roundtrip_p50_ms": float(np.percentile(rt, 50)),
               "roundtrip_p95_ms": float(np.percentile(rt, 95)), "fps_one_stream": float(1000 / rt.mean()),
               "cosine_mean": float(np.mean(cos)), "cosine_min": float(np.min(cos)),
               "objects_cpu": n_cpu, "objects_tpu": n_tpu, "box_match": float(np.mean(match)),
               "worker": {k: v for k, v in tpu_rt.worker_info.items() if k != "error_message"}}
        report["models"][base] = res
        print(f"LinQ туда-обратно: среднее {res['roundtrip_mean_ms']:.1f} мс, p50 {res['roundtrip_p50_ms']:.1f}, "
              f"p95 {res['roundtrip_p95_ms']:.1f} -> {res['fps_one_stream']:.1f} к/с на поток")
        print(f"Сходство INT8 (TPU) с float (CPU): косинус {res['cosine_mean']:.4f} (мин {res['cosine_min']:.4f})")
        print(f"Объектов: CPU {n_cpu}, TPU {n_tpu}; совпадение рамок {res['box_match']:.0%}")
        print("Ответ воркера:", res["worker"])
    json.dump(report, open("linq_bench.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nСохранено: linq_bench.json")


if __name__ == "__main__":
    main()
