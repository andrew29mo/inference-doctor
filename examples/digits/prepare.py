"""Train once, retain public numeric sklearn state, and export portable fixtures."""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import sklearn
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
STATE = Path(__file__).with_name("model.json")
PARAMS = dict(hidden_layer_sizes=(32,), activation="relu", solver="lbfgs", alpha=1e-4,
              max_iter=1000, max_fun=100000, tol=1e-6, random_state=42)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def dataset():
    x, y = load_digits(return_X_y=True)
    train, test = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=42)
    train, test = np.sort(train), np.sort(test)
    if set(train) & set(test) or len(train) + len(test) != len(y):
        raise ValueError("Invalid split")
    if {x[i].tobytes() for i in train} & {x[i].tobytes() for i in test}:
        raise ValueError("Exact duplicate images cross the split")
    provenance = {"dataset": "sklearn.datasets.load_digits", "rows": len(y),
                  "raw_pixels_uint8_sha256": digest(x.astype(np.uint8)), "labels_int64_sha256": digest(y.astype("<i8")),
                  "train_ids": train.tolist(), "test_ids": test.tolist(), "split_seed": 42,
                  "test_fraction": .25, "normalization": "pixels / 16; fixed physical range, no fitted statistics"}
    return x / 16., y.astype(np.int64), train, test, provenance


def fit_classifier(x_train, y_train):
    # Only training arrays reach fit. No early stopping, tuning, calibration or
    # normalization fitted on the held-out data.
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return MLPClassifier(**PARAMS).fit(x_train, y_train)


def restore(state):
    model = MLPClassifier(**{**PARAMS, **state["params"], "hidden_layer_sizes": tuple(state["params"]["hidden_layer_sizes"])})
    for name in ("coefs_", "intercepts_"):
        setattr(model, name, [np.asarray(a, dtype=np.float64) for a in state[name]])
    model.classes_ = np.asarray(state["classes_"], dtype=np.int64)
    for name in ("n_features_in_", "n_layers_", "n_outputs_", "out_activation_"):
        setattr(model, name, state[name])
    return model


def train_state():
    x, y, train, test, provenance = dataset()
    model = fit_classifier(x[train], y[train])
    state = {"format": 1, "provenance": provenance, "sklearn_version": sklearn.__version__,
             "params": PARAMS, "coefs_": [a.tolist() for a in model.coefs_],
             "intercepts_": [a.tolist() for a in model.intercepts_], "classes_": model.classes_.tolist(),
             "n_features_in_": model.n_features_in_, "n_layers_": model.n_layers_,
             "n_outputs_": model.n_outputs_, "out_activation_": model.out_activation_,
             "training_iterations": model.n_iter_, "training_loss": float(model.loss_)}
    # JSON conversion also normalizes tuple parameters for the persisted form.
    state = json.loads(json.dumps(state))
    with threadpool_limits(limits=1):
        original = model.predict_proba(x[test])
        recovered = restore(state).predict_proba(x[test])
    if not np.array_equal(original, recovered):
        raise ValueError("Restoring public numeric sklearn state changed its predictions")
    return state


def prepare(output, state_path=STATE):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state = json.loads(Path(state_path).read_text())
    x, y, train, test, provenance = dataset()
    if state["provenance"] != provenance or state["sklearn_version"] != sklearn.__version__:
        raise ValueError("Installed dataset/split/reference version differs from training provenance")
    model = restore(state)
    x_test = x[test].astype(np.float32)
    with threadpool_limits(limits=1):
        sklearn_probabilities = model.predict_proba(x[test])
        probabilities = sklearn_probabilities.astype(np.float32)
    if not np.array_equal(sklearn_probabilities.argmax(axis=1), probabilities.argmax(axis=1)):
        raise ValueError("Reference float32 conversion changes a sklearn predicted class")
    w1, w2 = [a.astype(np.float32) for a in model.coefs_]
    b1, b2 = [a.astype(np.float32) for a in model.intercepts_]
    with threadpool_limits(limits=1):
        hidden = np.maximum(x_test @ w1 + b1, 0)
        logits = hidden @ w2 + b2
        exp = np.exp(logits.astype(np.float64) - logits.max(axis=1, keepdims=True))
        exported_probabilities = exp / exp.sum(axis=1, keepdims=True)
    if not np.allclose(exported_probabilities, probabilities, rtol=1e-4, atol=1e-5):
        raise ValueError("Float32 export differs from the frozen sklearn reference")
    if not np.array_equal(exported_probabilities.argmax(axis=1), probabilities.argmax(axis=1)):
        raise ValueError("Float32 export changes a predicted class")
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["A"], name="dense1"),
             helper.make_node("Relu", ["A"], ["H"], name="relu"),
             helper.make_node("Gemm", ["H", "W2", "b2"], ["logits"], name="dense2"),
             helper.make_node("Softmax", ["logits"], ["probabilities"], name="softmax", axis=1)]
    graph = helper.make_graph(nodes, "trained-digits-mlp", [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 64])],
                              [helper.make_tensor_value_info(n, TensorProto.FLOAT, ["N", 10]) for n in ("logits", "probabilities")],
                              initializer=[numpy_helper.from_array(a, n) for a, n in ((w1, "W1"), (b1, "b1"), (w2, "W2"), (b2, "b2"))])
    exported = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10,
                                 producer_name="inference-doctor numeric sklearn MLP export")
    onnx.checker.check_model(exported)
    onnx.save(exported, output / "model.onnx")
    np.savez(output / "inputs.npz", X=x_test)
    np.savez(output / "reference.npz", logits=logits, probabilities=probabilities)
    np.savez(output / "labels.npz", labels=y[test], sample_ids=test.astype(np.int64), classes=model.classes_)
    predictions = model.classes_[probabilities.argmax(axis=1)]
    manifest = {"provenance": provenance, "model_state_sha256": hashlib.sha256(Path(state_path).read_bytes()).hexdigest(),
                "architecture": "64 -> 32 ReLU -> 10 logits -> Softmax", "training_rows": len(train), "test_rows": len(test),
                "reference": "Frozen sklearn float64 MLP predict_proba converted to float32; NumPy float32 logits",
                "correct": int(np.sum(predictions == y[test])), "accuracy": float(np.mean(predictions == y[test])),
                "predictions_int64_sha256": digest(predictions.astype("<i8")),
                "sample_ids_int64_sha256": digest(test.astype("<i8")), "sklearn_version": sklearn.__version__}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="examples/generated/digits")
    parser.add_argument("--state", default=str(STATE))
    parser.add_argument("--train", action="store_true", help="Train a new state into OUTPUT/model.json; never overwrite the retained model implicitly")
    args = parser.parse_args()
    if args.train:
        directory = Path(args.output)
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / "model.json"
        state_path.write_text(json.dumps(train_state(), indent=2, allow_nan=False) + "\n")
    else:
        state_path = Path(args.state)
    print(json.dumps({k: v for k, v in prepare(args.output, state_path).items() if k != "provenance"}, indent=2))
