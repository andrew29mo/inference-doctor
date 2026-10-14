"""Generate original synthetic ONNX graphs and independent NumPy references."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def generate(directory, kind="mlp", batch=16):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261014)
    width = {"linear": 8, "mlp": 64, "residual": 128}[kind]
    hidden = 32 if kind == "mlp" else width
    outputs = 4 if kind == "linear" else 10
    # Input generation must not advance the weight RNG when batch size changes.
    x = np.random.default_rng(20261015).normal(size=(batch, width)).astype(np.float32)
    initializers, nodes = [], []

    def weight(name, shape):
        array = (rng.normal(size=shape) * .1).astype(np.float32)
        initializers.append(numpy_helper.from_array(array, name))
        return array

    if kind == "linear":
        w = weight("W", (width, outputs))
        b = weight("b", (outputs,))
        nodes.append(helper.make_node("Gemm", ["X", "W", "b"], ["Y"], name="linear"))
        references = {"Y": x @ w + b}
    else:
        w1 = weight("W1", (width, hidden))
        b1 = weight("b1", (hidden,))
        w2 = weight("W2", (hidden, outputs))
        b2 = weight("b2", (outputs,))
        nodes.extend([helper.make_node("Gemm", ["X", "W1", "b1"], ["A"], name="dense1"),
                      helper.make_node("Relu", ["A"], ["H"], name="relu")])
        h = np.maximum(x @ w1 + b1, 0)
        if kind == "residual":
            nodes.append(helper.make_node("Add", ["H", "X"], ["R"], name="residual"))
            h = h + x
        nodes.append(helper.make_node("Gemm", ["R" if kind == "residual" else "H", "W2", "b2"], ["Y"], name="dense2"))
        references = {"Y": h @ w2 + b2}
        if kind == "mlp":
            references["H"] = h
    metadata = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", outputs])]
    if kind == "mlp":
        metadata.append(helper.make_tensor_value_info("H", TensorProto.FLOAT, ["N", hidden]))
    graph = helper.make_graph(nodes, kind,
                              [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", width])],
                              metadata, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10,
                              producer_name="inference-doctor synthetic example generator")
    onnx.checker.check_model(model)
    onnx.save(model, directory / "model.onnx")
    np.savez(directory / "inputs.npz", X=x)
    np.savez(directory / "reference.npz", **references)
    manifest = {"kind": kind, "batch": batch, "weight_seed": 20261014, "input_seed": 20261015,
                "purpose": "Synthetic correctness and deployment measurement; no training or dataset accuracy.",
                "reference": "Independent NumPy matrix multiplication, bias, ReLU, residual addition.",
                "array_sha256": {name: hashlib.sha256(a.tobytes()).hexdigest()
                                 for name, a in {"X": x, **references}.items()}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="examples/generated/mlp-b16")
    parser.add_argument("--kind", choices=["linear", "mlp", "residual"], default="mlp")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if args.batch <= 0:
        parser.error("batch must be positive")
    generate(args.output, args.kind, args.batch)
