"""Разрезает YOLO ONNX на две части для LinQ TPU.

head.onnx  — вход 'images' -> «сырые» выходы последних свёрток детекционной головы.
             Это всё сеть из Conv/Sigmoid/Mul/Concat/Resize — то, что хорошо ложится на TPU.
tail.onnx  — сырые выходы -> 'output0' (reshape, DFL softmax, декодирование координат).
             Считается на CPU через onnxruntime за доли миллисекунды.

Так же поступает сам SDK LinQ для yolov8n_pose (см. tpu_framework.pdf, гл. 6, сноска 6).
Постобработка (output0 -> рамки -> NMS) после этого одинакова для CPU и TPU.

Запуск:
    python tools/split_onnx.py models/yolo11n-pose.onnx
    python tools/split_onnx.py models/yolo11n.onnx --imgsz 640
Результат: models/yolo11n-pose.head.onnx, models/yolo11n-pose.tail.onnx, models/yolo11n-pose.split.json

Работает и с пакетом `onnx`, и без него (тогда нужен onnx_ml_pb2.py, сгенерированный protoc).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys

import numpy as np

try:  # штатный путь
    import onnx  # type: ignore
    ModelProto = onnx.ModelProto
    TensorProto = onnx.TensorProto
    ValueInfoProto = onnx.ValueInfoProto
except ImportError:  # запасной путь: protobuf-классы, сгенерированные из onnx-ml.proto
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_pb"))
    import onnx_ml_pb2 as _pb  # type: ignore
    ModelProto = _pb.ModelProto
    TensorProto = _pb.TensorProto
    ValueInfoProto = _pb.ValueInfoProto

import onnxruntime as ort

# последние свёртки головы YOLOv8/YOLO11: .../cv2.0/cv2.0.2/Conv, .../cv3.1/cv3.1.2/Conv, .../cv4.2/cv4.2.2/Conv
HEAD_CONV_RE = re.compile(r"/cv[2-5]\.(\d+)/cv[2-5]\.\1\.2/Conv$")

FLOAT = 1  # TensorProto.FLOAT


def load(path: str):
    m = ModelProto()
    with open(path, "rb") as f:
        m.ParseFromString(f.read())
    return m


def save(m, path: str) -> None:
    with open(path, "wb") as f:
        f.write(m.SerializeToString())


def make_vi(name: str, shape):
    vi = ValueInfoProto()
    vi.name = name
    tt = vi.type.tensor_type
    tt.elem_type = FLOAT
    for d in shape:
        tt.shape.dim.add().dim_value = int(d)
    return vi


def freeze_input(m, imgsz: int) -> None:
    """Динамические оси входа (batch/height/width) -> статические: TPU не поддерживает динамику."""
    for inp in m.graph.input:
        dims = inp.type.tensor_type.shape.dim
        fixed = [1, 3, imgsz, imgsz]
        for i, d in enumerate(dims):
            d.ClearField("dim_param")
            d.dim_value = fixed[i]
    # выходы тоже очищаем от символьных размеров, ORT сам выведет
    for out in m.graph.output:
        for d in out.type.tensor_type.shape.dim:
            if d.dim_param:
                d.ClearField("dim_param")
    del m.graph.value_info[:]


def find_cut_tensors(m) -> list[str]:
    cuts = [n.output[0] for n in m.graph.node if n.op_type == "Conv" and HEAD_CONV_RE.search(n.name)]
    if not cuts:
        raise SystemExit("Не нашёл свёртки головы YOLO (cv2/cv3/cv4). Укажите --cut вручную.")
    return cuts


def tensor_shapes(m, names: list[str], imgsz: int) -> dict[str, list[int]]:
    probe = copy.deepcopy(m)
    for n in names:
        o = probe.graph.output.add()
        o.name = n
        o.type.tensor_type.elem_type = FLOAT
    sess = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    x = np.zeros((1, 3, imgsz, imgsz), np.float32)
    outs = sess.run(names, {sess.get_inputs()[0].name: x})
    return {n: list(o.shape) for n, o in zip(names, outs)}


def extract(m, input_names: list[str], output_names: list[str], shapes: dict[str, list[int]]):
    """Подграф от input_names до output_names (обратный обход зависимостей)."""
    producers = {}
    for n in m.graph.node:
        for o in n.output:
            producers[o] = n
    inputs = set(input_names)
    needed_nodes, needed_tensors = [], set()
    stack = list(output_names)
    seen = set()
    while stack:
        t = stack.pop()
        if t in seen or t in inputs:
            continue
        seen.add(t)
        node = producers.get(t)
        if node is None:
            needed_tensors.add(t)  # инициализатор или константа
            continue
        if id(node) not in {id(x) for x in needed_nodes}:
            needed_nodes.append(node)
        stack.extend([i for i in node.input if i])
    order = {id(n): i for i, n in enumerate(m.graph.node)}
    needed_nodes.sort(key=lambda n: order[id(n)])

    sub = ModelProto()
    sub.CopyFrom(m)
    g = sub.graph
    del g.node[:]
    g.node.extend([copy.deepcopy(n) for n in needed_nodes])
    used = set()
    for n in needed_nodes:
        used.update(n.input)
    inits = [copy.deepcopy(i) for i in m.graph.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(inits)
    del g.input[:]
    orig_inputs = {i.name: i for i in m.graph.input}
    for name in input_names:
        g.input.append(copy.deepcopy(orig_inputs[name]) if name in orig_inputs else make_vi(name, shapes[name]))
    del g.output[:]
    orig_outputs = {o.name: o for o in m.graph.output}
    for name in output_names:
        g.output.append(copy.deepcopy(orig_outputs[name]) if name in orig_outputs else make_vi(name, shapes[name]))
    del g.value_info[:]
    return sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--cut", nargs="*", help="имена тензоров разреза (по умолчанию — найти автоматически)")
    args = ap.parse_args()

    m = load(args.model)
    freeze_input(m, args.imgsz)
    in_name = m.graph.input[0].name
    out_names = [o.name for o in m.graph.output]
    cuts = args.cut or find_cut_tensors(m)
    shapes = tensor_shapes(m, cuts, args.imgsz)
    shapes[in_name] = [1, 3, args.imgsz, args.imgsz]

    head = extract(m, [in_name], cuts, shapes)
    tail = extract(m, cuts, out_names, shapes)

    base = os.path.splitext(args.model)[0]
    save(head, base + ".head.onnx")
    save(tail, base + ".tail.onnx")
    save(m, base + ".static.onnx")

    # проверка: head+tail == full
    full = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    hs = ort.InferenceSession(head.SerializeToString(), providers=["CPUExecutionProvider"])
    ts = ort.InferenceSession(tail.SerializeToString(), providers=["CPUExecutionProvider"])
    x = np.random.RandomState(0).rand(1, 3, args.imgsz, args.imgsz).astype(np.float32)
    ref = full.run(None, {in_name: x})[0]
    h = hs.run(cuts, {in_name: x})
    got = ts.run(None, dict(zip(cuts, h)))[0]
    err = float(np.abs(ref - got).max())

    meta = {
        "input": in_name,
        "imgsz": args.imgsz,
        "head_outputs": cuts,
        "head_output_shapes": [shapes[c] for c in cuts],
        "tail_outputs": out_names,
        "check_max_abs_err": err,
    }
    with open(base + ".split.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"head: {len(head.graph.node)} узлов, tail: {len(tail.graph.node)} узлов; расхождение {err:.2e}")
    if err > 1e-3:
        raise SystemExit("Расхождение head+tail и полной модели слишком велико!")


if __name__ == "__main__":
    main()
