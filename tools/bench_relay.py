"""Замер быстрого режима LinQ (посредник linq/relay.py на стенде) с ноутбука.

    python tools/bench_relay.py --n 30
Печатает задержку по частям (сеть, посредник, TPU) и сверяет найденные объекты с CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from sv.infer.relay_client import RelayYolo  # noqa: E402
from sv.infer.runtime import make_runtime  # noqa: E402
from sv.infer.yolo import YoloModel  # noqa: E402
from sv.tracker import iou_matrix  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--video", default="data/ceh1_worker_zone.mp4")
    a = ap.parse_args()
    b = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))["backend"]
    r = b["rabbit"]
    cap = cv2.VideoCapture(a.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, total - 1, a.n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    report = {}
    for task, base, conf in (("pose", b["pose"], b.get("pose_conf", 0.2)), ("detect", b["detect"], b.get("detect_conf", 0.25))):
        cls = b.get("detect_classes") if task == "detect" else None
        relay = RelayYolo(task, r["host"], r["login"], conf=conf, classes=cls, timeout=30)
        cpu = YoloModel(make_runtime("cpu", base), task, conf=conf, classes=cls)
        relay(frames[0])
        tm, match = [], []
        for f in frames:
            d1 = relay(f)
            tm.append(dict(relay.last_timing))
            d0 = cpu(f)
            if d0 and d1:
                m = iou_matrix(np.array([d.box for d in d0]), np.array([d.box for d in d1]))
                match.append(float((m.max(1) > 0.5).sum() / max(len(d0), len(d1))))
            else:
                match.append(1.0 if not d0 and not d1 else 0.0)
        g = lambda k: float(np.mean([t.get(k, 0) for t in tm]))
        res = {k: round(g(k), 1) for k in ("total_ms", "network_ms", "relay_ms", "tpu_rt_ms", "head_ms", "jpeg_kb")}
        res["p95_total_ms"] = round(float(np.percentile([t["total_ms"] for t in tm], 95)), 1)
        res["fps"] = round(1000 / res["total_ms"], 2)
        res["box_match_vs_cpu"] = round(float(np.mean(match)), 3)
        report[task] = res
        print(f"{task}: полный путь {res['total_ms']:.0f} мс (p95 {res['p95_total_ms']:.0f}) = сеть {res['network_ms']:.0f} + "
              f"посредник {res['relay_ms']:.0f} (из них TPU туда-обратно {res['tpu_rt_ms']:.0f}, расчёт на TPU "
              f"{res['head_ms']:.1f}); JPEG {res['jpeg_kb']:.0f} КБ; совпадение с CPU {res['box_match_vs_cpu']:.0%}")
    json.dump(report, open("linq/relay_bench.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
