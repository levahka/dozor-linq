"""Сборка программы LinQ TPU из head.onnx (адаптированный compile.py из tpu_framework).

Где запускать: Linux (WSL Ubuntu 24.04 подходит), Python 3.12, установленный tpu_framework:
    python3.12 -m venv ~/linq-venv && source ~/linq-venv/bin/activate
    cd tpu_framework/packets && pip install -U pip && pip install -r requirements.txt && cd -

Шаги:
    1) python tools/split_onnx.py models/yolo11n-pose.onnx   (уже сделано, файлы лежат в models/)
    2) python linq/compile_linq.py models/yolo11n-pose.head.onnx --calib data/ --n 64
       -> models/yolo11n-pose.head_b1_o5_h1m.tpu
    3) то же для models/yolo11n.head.onnx
    4) загрузить .tpu в Jupyter (http://hackathon.lin-q.ru/) и проверить linq/check_tpu.py

Отличия от примера организаторов:
  * калибровка на РЕАЛЬНЫХ кадрах с камер (с тем же letterbox, что в рантайме), а не на шуме —
    для INT8 это критично: на шуме пороги активаций получаются неверными и детектор «слепнет»;
  * O5 с автоматическим откатом на O1, если компилятор не справился;
  * печатает итоговые имена входа/выходов для конфигурации рантайма.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time

import cv2
import numpy as np
import onnx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sv.infer.yolo import letterbox  # noqa: E402  — ровно та же предобработка, что в рантайме

from tpu_framework import O1, O5, TPU_H1M, Network, QuantizedModel, RegularModel, TpuProgram, compiler  # noqa: E402
from tpu_framework import preliminary_model_conditioning  # noqa: E402


def calib_frames(paths, n, imgsz):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.jpg", "*.jpeg", "*.png"):
                files += glob.glob(os.path.join(p, "**", ext), recursive=True)
        else:
            files.append(p)
    imgs = []
    vids = [f for f in files if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))]
    pics = [f for f in files if f not in vids]
    per_vid = max(1, n // max(1, len(vids))) if vids else 0
    for v in vids:
        cap = cv2.VideoCapture(v)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for idx in np.linspace(0, total - 1, per_vid).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if ok:
                imgs.append(fr)
    random.shuffle(pics)
    for p in pics[: max(0, n - len(imgs))]:
        im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            imgs.append(im)
    if not imgs:
        raise SystemExit("Нет кадров для калибровки — укажите --calib с видео или картинками")
    imgs = imgs[:n]
    print(f"Калибровка: {len(imgs)} кадров из {len(vids)} видео и {len(pics)} картинок")
    return np.concatenate([letterbox(im, imgsz)[0] for im in imgs], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="head.onnx после tools/split_onnx.py")
    ap.add_argument("--calib", nargs="+", default=["data"], help="видео/папки с кадрами для калибровки")
    ap.add_argument("--n", type=int, default=64, help="число калибровочных кадров")
    ap.add_argument("--preset", choices=["o1", "o5"], default="o5")
    ap.add_argument("--samplewise", type=float, default=None,
                    help="samplewise_percentile (напр. 99.9) — устойчивее к выбросам на большой выборке")
    args = ap.parse_args()

    base = os.path.splitext(args.model)[0]
    work = base + "_linq"
    os.makedirs(work, exist_ok=True)

    print("\n1/3. Оптимизация графа")
    m = onnx.load(args.model)
    names_init = {i.name for i in m.graph.initializer}
    for inp in list(m.graph.input):
        if inp.name in names_init:
            m.graph.input.remove(inp)
    shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in m.graph.input}
    for k, s in shapes.items():
        if 0 in s:
            raise SystemExit(f"динамическая форма входа {k}={s}: прогоните tools/split_onnx.py")
    opt = preliminary_model_conditioning(m)
    opt_path = os.path.join(work, "opt.onnx")
    onnx.save(opt, opt_path)
    print("   входы:", shapes, "\n   выходы:", [o.name for o in opt.graph.output])

    print("\n2/3. Квантование INT8 (калибровка на реальных кадрах)")
    opt = onnx.load(opt_path)
    in_name, in_shape = next(iter(shapes.items()))
    calib = calib_frames(args.calib, args.n, in_shape[2])
    rm = RegularModel(model=opt)
    mapping = getattr(rm, "anchors_mapping", None) or {}
    cal = {mapping.get(in_name, in_name): calib}
    t = time.time()
    kw = {}
    if args.samplewise:
        kw["samplewise_percentile"] = args.samplewise
    thresholds = rm.calibrate_torch(calibration_data=cal, **kw)
    RegularModel.dump_thresholds(os.path.join(work, "thresholds.json"), thresholds)
    qm = rm.quantize(thresholds)
    qm_name = "quant.qm.onnx"
    qm.save(file_dir=work, file_name=qm_name)
    print(f"   готово за {time.time() - t:.0f} с")

    print("\n3/3. Компиляция под LinQ (H1M)")
    presets = [O5, O1] if args.preset == "o5" else [O1]
    for preset in presets:
        tag = "o5" if preset == O5 else "o1"
        try:
            q = QuantizedModel.load(os.path.join(work, qm_name))
            network, _ = Network.from_quantized_model(q)
            t = time.time()
            exe, desc = compiler.compile_(hardware_parameters=TPU_H1M, network=network, parameters=preset)
            prog = TpuProgram.from_executable(exe, desc)
            out = f"{base}_b{in_shape[0]}_{tag}_h1m.tpu"
            prog.to_file(out)
            print(f"\nOK: {out}  ({time.time() - t:.0f} с, пресет {tag.upper()})")
            print("Пропишите путь в config/config.yaml -> backend.tpu_programs")
            return
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"   пресет {tag.upper()} не скомпилировался: {e}")
    raise SystemExit("Компиляция не удалась ни с O5, ни с O1 — пишите экспертам LinQ с логом выше.")


if __name__ == "__main__":
    main()
