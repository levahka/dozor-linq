"""Метрики на роликах команды: data/my/*.mp4 + data/my/разметка.csv.

    python tools/eval_my.py --fps 5                      # ролики в data/my
    python tools/eval_my.py --fps 5 --dir data/my/small  # уменьшенные копии (960 px, 15 к/с) — так считались метрики
Настройки камер для роликов — config/eval_my.yaml, результат — metrics/team_videos_<fps>fps.json.
Разметка: файл;нарушение;начало;конец;комментарий (время м:сс). Файл ищется по номеру в начале имени.
Тревога засчитывается верной, если она пришла в интервале нарушения того же типа (−1…+3 с).
Тревоги «лежит/неподвижность» внутри интервала «падение» тоже верные: человек и правда лежит.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import cv2
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from sv.pipeline import CameraWorker, Models  # noqa: E402

RU = {"падение": "fall", "лежит": "immobile", "бег": "running", "зона": "zone", "машина": "proximity",
      "телефон": "phone", "норма": None, "нет": None}
OK_INSIDE = {"fall": {"fall", "immobile"}, "immobile": {"immobile", "fall"}}
NAMES = {"fall": "Падение", "immobile": "Лежит неподвижно", "running": "Бег", "zone": "Опасная зона",
         "proximity": "Сближение с машиной", "phone": "Телефон"}


def sec(s):
    s = (s or "").strip()
    if not s:
        return None
    if ":" in s:
        m, x = s.split(":")
        return int(m) * 60 + float(x)
    return float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/my")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--only")
    ap.add_argument("--save", action="store_true", help="сохранить ролики с разметкой в data/my/out")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    ev_cfg = yaml.safe_load(open("config/eval_my.yaml", encoding="utf-8"))
    vids = {}
    for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV"):
        for p in sorted(glob.glob(os.path.join(a.dir, ext))):
            vids.setdefault(os.path.basename(p)[:2], p)
    vids = dict(sorted(vids.items()))
    lab = os.path.join(a.dir, "разметка.csv")
    if not os.path.exists(lab):
        lab = os.path.join(os.path.dirname(os.path.normpath(a.dir)), "разметка.csv")
    gt = {}
    with open(lab, encoding="utf-8-sig") as f:
        for row in csv.reader(f, delimiter=";"):
            if len(row) < 2 or not row[0].strip() or row[0].startswith("файл"):
                continue
            k = row[0].strip()[:2]
            gt.setdefault(k, [])
            t = RU.get(row[1].strip().lower())
            if t:
                gt[k].append(dict(type=t, start=sec(row[2]), end=sec(row[3]), hit=False))
    models = Models(cfg)
    report, all_ev = [], []
    for k, path in vids.items():
        if a.only and k not in a.only.split(","):
            continue
        vc = (ev_cfg.get("videos") or {}).get(k, {})
        cam = {"id": f"my{k}", "name": os.path.basename(path), "source": path, "zones": vc.get("zones", []),
               "rules": {**ev_cfg["default_rules"], **vc.get("rules", {})}, "detect_every": 1}
        events = []

        def on_open(w, rule, sig, ctx, ev):
            events.append(dict(type=rule.type, alert=ev.alert_t, onset=ev.onset_t, info=sig.info))
            return len(events)

        w = CameraWorker(cam, models, on_open, lambda *x: None, media_dir="data/replay_media")
        cap = cv2.VideoCapture(path)
        vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(vfps / a.fps))
        writer, i, t0, n = None, 0, time.time(), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                w.process(frame, i / vfps, render=a.save)
                n += 1
                if a.save:
                    vis = cv2.imdecode(__import__("numpy").frombuffer(w.jpeg, "uint8"), 1)
                    if writer is None:
                        os.makedirs(os.path.join(a.dir, "out"), exist_ok=True)
                        writer = cv2.VideoWriter(os.path.join(a.dir, "out", os.path.basename(path)),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), vfps / step, (vis.shape[1], vis.shape[0]))
                    writer.write(vis)
            i += 1
        if writer:
            writer.release()
        g = gt.get(k, [])
        for e in events:
            e["ok"] = False
            for x in g:
                if e["type"] in OK_INSIDE.get(x["type"], {x["type"]}) and x["start"] - 1 <= e["alert"] <= x["end"] + 3:
                    e["ok"] = True
                    if e["type"] == x["type"] and not x["hit"]:
                        x["hit"], x["delay"] = True, max(0.0, e["alert"] - x["start"])
        line = f"{os.path.basename(path):<16} {i / vfps:5.1f} с | разметка: " + \
               (", ".join(f"{NAMES[x['type']]} {x['start']:.0f}–{x['end']:.0f}с" for x in g) or "нарушений нет") + \
               " | тревоги: " + (", ".join(f"{NAMES[e['type']]} {e['alert']:.1f}с{'' if e['ok'] else ' ✗ЛОЖНАЯ'}" for e in events) or "нет")
        print(line, flush=True)
        report.append(dict(video=os.path.basename(path), gt=g, events=events, sec=i / vfps, frames=n, proc_s=time.time() - t0))
        all_ev += events
    # итог
    gts = [x for r in report for x in r["gt"]]
    hit = sum(x["hit"] for x in gts)
    good = sum(e["ok"] for e in all_ev)
    delays = [x["delay"] for x in gts if "delay" in x]
    neg = [r for r in report if not r["gt"]]
    neg_clean = sum(1 for r in neg if not r["events"])
    summ = dict(fps=a.fps, videos=len(report), violations=len(gts), found=hit, recall=hit / len(gts) if gts else None,
                alerts=len(all_ev), true_alerts=good, precision=good / len(all_ev) if all_ev else None,
                mean_delay_s=sum(delays) / len(delays) if delays else None, max_delay_s=max(delays) if delays else None,
                clean_videos=len(neg), clean_videos_no_alerts=neg_clean)
    print(f"\nИТОГ при {a.fps:g} к/с: нарушений {len(gts)}, найдено {hit} (полнота {summ['recall']:.0%}); "
          f"тревог {len(all_ev)}, верных {good} (точность {summ['precision'] if summ['precision'] is not None else 0:.0%}); "
          f"задержка тревоги в среднем {summ['mean_delay_s'] or 0:.1f} с, максимум {summ['max_delay_s'] or 0:.1f} с; "
          f"роликов без нарушений {len(neg)}, из них без единой тревоги {neg_clean}")
    os.makedirs("metrics", exist_ok=True)
    json.dump(dict(summary=summ, videos=report), open(f"metrics/team_videos_{a.fps:g}fps.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
