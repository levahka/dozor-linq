"""Прогон видео через конвейер без веб-сервера — для отладки правил и замера метрик.

    python tools/replay.py data/ceh1_worker_zone.mp4 --camera cam01
    python tools/replay.py my_fall.mp4 --camera cam03 --fps 10 --gt gt.csv

Время берётся из видео (кадр / fps), поэтому прогон быстрее реального времени, а события
считаются так же, как вживую. --fps прореживает кадры до заданной частоты обработки
(имитация того, сколько кадров в секунду реально успевает система).

--gt  CSV разметки «истины»: type;start_s;end_s  (секунды от начала ролика).
      Скрипт посчитает полноту/точность по событиям и задержку тревоги.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sv.pipeline import CameraWorker, Models  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--camera", help="id камеры из конфига, чьи зоны/правила использовать")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--gt")
    ap.add_argument("--save", help="сохранить аннотированное видео (mp4)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cam = next((c for c in cfg["cameras"] if c["id"] == args.camera), None) if args.camera else None
    cam = dict(cam or {"id": "replay", "rules": {r: {} for r in ("fall", "immobile", "running", "zone", "proximity", "phone")}})
    cam["source"] = args.video
    models = Models(cfg)
    events = []

    def on_open(w, rule, sig, ctx, ev):
        events.append(dict(id=len(events) + 1, type=rule.type, title=rule.title, key=sig.key, info=sig.info,
                           onset=ev.onset_t, alert=ev.alert_t, end=None))
        if args.verbose:
            print(f"  [{ev.alert_t:7.2f}s] ОТКРЫТО #{len(events)} {rule.title} ({sig.key}) {sig.info}")
        return len(events)

    def on_close(w, ev):
        for e in events:
            if e["id"] == ev.id:
                e["end"] = ev.closed_t

    w = CameraWorker(cam, models, on_open, on_close, media_dir="data/replay_media")
    cap = cv2.VideoCapture(args.video)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(vfps / args.fps))
    writer = None
    i, t0, nproc = 0, time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            ts = i / vfps
            w.process(frame, ts, render=bool(args.save))
            nproc += 1
            if args.save:
                vis = cv2.imdecode(__import__("numpy").frombuffer(w.jpeg, "uint8"), 1)
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), vfps / step,
                                             (vis.shape[1], vis.shape[0]))
                writer.write(vis)
        i += 1
    if writer:
        writer.release()
    el = time.time() - t0
    dur = i / vfps
    print(f"\nВидео {dur:.1f} с, обработано {nproc} кадров за {el:.1f} с "
          f"({nproc / el:.1f} к/с; поза {w.models.pose.last_timing.get('total_ms', 0):.0f} мс/кадр)")
    print(f"Событий: {len(events)}")
    for e in events:
        end = f"{e['end']:.1f}" if e["end"] else "…"
        print(f"  #{e['id']:<3} {e['alert']:7.1f}s  {e['title']:<42} субъект {e['key']:<8} "
              f"длит. до {end}s  задержка {e['alert'] - e['onset']:.1f}s  {e['info']}")

    if args.gt:
        gt = []
        with open(args.gt, encoding="utf-8") as f:
            for row in csv.reader(f, delimiter=";"):
                if row and not row[0].startswith("#"):
                    gt.append(dict(type=row[0], start=float(row[1]), end=float(row[2]), hit=False))
        tp = 0
        for e in events:
            m = [g for g in gt if g["type"] == e["type"] and g["start"] - 1 <= e["alert"] <= g["end"] + 3]
            if m:
                tp += 1
                for g in m:
                    g["hit"] = True
                    g.setdefault("delay", e["alert"] - g["start"])
        types = sorted({g["type"] for g in gt} | {e["type"] for e in events})
        print("\nМетрики по событиям:")
        for t in types:
            g = [x for x in gt if x["type"] == t]
            ev = [x for x in events if x["type"] == t]
            hit = sum(x["hit"] for x in g)
            good = sum(1 for e in ev if any(x["type"] == t and x["start"] - 1 <= e["alert"] <= x["end"] + 3 for x in g))
            rec = hit / len(g) if g else float("nan")
            prec = good / len(ev) if ev else float("nan")
            delays = [x["delay"] for x in g if "delay" in x]
            d = sum(delays) / len(delays) if delays else float("nan")
            print(f"  {t:<10} полнота {rec:6.1%} ({hit}/{len(g)})  точность {prec:6.1%} ({good}/{len(ev)})  "
                  f"ср. задержка тревоги {d:.1f} с")


if __name__ == "__main__":
    main()
