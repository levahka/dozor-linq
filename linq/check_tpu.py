"""Проверка и замер программы на LinQ TPU. Запускать в Jupyter на hackathon.lin-q.ru
(загрузите туда папку проекта целиком или хотя бы sv/, models/*.tpu, models/*.tail.onnx, models/*.split.json).

    python linq/check_tpu.py --model models/yolo11n-pose --tpu models/yolo11n-pose.head_b1_o5_h1m.tpu --images data/frames
    # или в ячейке ноутбука:  %run linq/check_tpu.py --model ... --tpu ...

Что делает:
  1. печатает входы/выходы .tpu (из metadata.json) — чтобы сверить имена с head.onnx;
  2. гоняет N кадров: время инференса на TPU (среднее, p50, p95, FPS);
  3. если есть onnxruntime и head.onnx — сравнивает выходы TPU с float-моделью (косинус, макс. ошибка);
  4. декодирует результат (tail + NMS) и печатает найденных людей/объекты — sanity-check квантования.
Итоговые цифры — в linq_bench.json, их показываем на защите.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def tpu_meta(path):
    with zipfile.ZipFile(path) as z:
        m = json.loads(z.read("metadata.json"))
    ins = {k: v for d in m["inputs"].values() for k, v in d.items()}
    outs = {k: v for d in m["outputs"].values() for k, v in d.items()}
    return ins, outs


def load_frames(src, n, imgsz):
    import cv2
    from sv.infer.yolo import letterbox

    frames = []
    if src and os.path.isdir(src):
        for f in sorted(glob.glob(os.path.join(src, "*.*")))[:n]:
            im = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                frames.append(im)
    elif src:
        cap = cv2.VideoCapture(src)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
        for i in np.linspace(0, total - 1, n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                frames.append(fr)
    if not frames:
        print("Кадров нет — использую случайный шум (только для замера скорости)")
        return [None] * n, [np.random.rand(1, 3, imgsz, imgsz).astype(np.float32) for _ in range(n)]
    return frames, [letterbox(f, imgsz)[0] for f in frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="база модели, напр. models/yolo11n-pose")
    ap.add_argument("--tpu", required=True)
    ap.add_argument("--images", default="data/ceh1_worker_zone.mp4", help="папка с кадрами или видео")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--task", choices=["pose", "detect"], default=None)
    args = ap.parse_args()

    import pytpu as tpu

    meta = json.load(open(args.model + ".split.json", encoding="utf-8"))
    ins, outs = tpu_meta(args.tpu)
    print("TPU-программа:", args.tpu)
    for k, v in ins.items():
        print(f"  вход  {k}: {v['user_shape']} {v['user_dtype']} -> {v['tpu_dtype']}")
    for k, v in outs.items():
        print(f"  выход {k}: {v['user_shape']} {v['user_dtype']}")
    missing = [c for c in meta["head_outputs"] if c not in outs]
    if missing:
        print("!! имена выходов .tpu не совпадают с head.onnx:", missing[:3], "... — поправьте сопоставление в sv/infer/runtime.py")

    frames, xs = load_frames(args.images, args.n, meta["imgsz"])
    in_name = list(ins)[0]
    devs = tpu.Device.list_devices()
    print("Устройства TPU:", devs)
    times, tpu_outs = [], []
    with tpu.Device.open(devs[0]) as dev, dev.load(args.tpu) as prog, prog.inference() as inf:
        inf.run({in_name: xs[0]})  # прогрев
        for x in xs:
            t = time.perf_counter()
            r = inf.run({in_name: x})
            times.append((time.perf_counter() - t) * 1000)
            tpu_outs.append(r)
    t = np.array(times)
    res = {"program": os.path.basename(args.tpu), "frames": len(t), "mean_ms": float(t.mean()),
           "p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)), "fps": float(1000 / t.mean())}
    print(f"\nTPU head: среднее {res['mean_ms']:.2f} мс, p50 {res['p50_ms']:.2f}, p95 {res['p95_ms']:.2f} -> {res['fps']:.0f} к/с")

    try:
        import onnxruntime as ort

        head = ort.InferenceSession(args.model + ".head.onnx", providers=["CPUExecutionProvider"])
        tail = ort.InferenceSession(args.model + ".tail.onnx", providers=["CPUExecutionProvider"])
        cos = []
        for x, r in list(zip(xs, tpu_outs))[:10]:
            ref = head.run(meta["head_outputs"], {meta["input"]: x})
            for c, rf in zip(meta["head_outputs"], ref):
                a = np.asarray(r[c], np.float32).ravel()
                b = rf.ravel()
                cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        res["cosine_min"] = float(np.min(cos))
        res["cosine_mean"] = float(np.mean(cos))
        print(f"Сходство с float-моделью: косинус средн. {res['cosine_mean']:.4f}, мин. {res['cosine_min']:.4f} "
              f"({'OK' if res['cosine_min'] > 0.98 else 'ПРОВЕРЬТЕ калибровку'})")

        from sv.infer.yolo import YoloModel

        task = args.task or ("pose" if "pose" in args.model else "detect")

        class _TpuOut:  # подставляем готовый выход TPU как «рантайм»
            imgsz = meta["imgsz"]
            last_timing = {}

            def __init__(self, r):
                self.r = r

            def __call__(self, x):
                h = [np.asarray(self.r[c], np.float32).reshape(s) for c, s in zip(meta["head_outputs"], meta["head_output_shapes"])]
                return tail.run(None, dict(zip(meta["head_outputs"], h)))[0]

        if frames[0] is not None:
            for i in range(min(5, len(frames))):
                dets = YoloModel(_TpuOut(tpu_outs[i]), task)(frames[i])
                print(f"  кадр {i}: " + ", ".join(f"{d.label} {d.score:.2f}" for d in dets[:8]) or "ничего")
    except ImportError:
        print("onnxruntime нет — сравнение с float пропущено (pip install onnxruntime)")

    out = os.path.join(ROOT, "linq_bench.json")
    allres = json.load(open(out)) if os.path.exists(out) else {}
    allres[os.path.basename(args.tpu)] = res
    json.dump(allres, open(out, "w"), indent=2, ensure_ascii=False)
    print("\nСохранено в", out)


if __name__ == "__main__":
    main()
