"""Подбор размера кадра для канала до стенда LinQ: чем хуже сеть, тем меньше JPEG.

    python tools/net_probe.py --n 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from sv.infer.relay_client import RelayYolo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--video", default="data/ceh1_worker_zone.mp4")
    a = ap.parse_args()
    r = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))["backend"]["rabbit"]
    cap = cv2.VideoCapture(a.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, total - 1, a.n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    for size, q in ((640, 80), (480, 70), (416, 60), (320, 55)):
        m = RelayYolo("pose", r["host"], r["login"], conf=0.2, timeout=30, send_size=size, quality=q)
        m(frames[0])
        tm, n = [], 0
        for f in frames:
            d = m(f)
            n += len(d)
            tm.append(dict(m.last_timing))
        tot = np.mean([t["total_ms"] for t in tm]); kb = np.mean([t["jpeg_kb"] for t in tm])
        print(f"{size}px q{q}: {kb:5.1f} КБ, путь {tot:6.0f} мс, сеть {np.mean([t['network_ms'] for t in tm]):6.0f} мс, людей {n}")
    # параллельно: 4 кадра в полёте (как 4 камеры)
    m = RelayYolo("pose", r["host"], r["login"], conf=0.2, timeout=30, send_size=416, quality=60)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(m, frames * 2))
    dt = time.perf_counter() - t0
    print(f"параллельно 4 потока, 416px: {len(frames)*2} кадров за {dt:.1f} с = {len(frames)*2/dt:.2f} к/с суммарно")


if __name__ == "__main__":
    main()
