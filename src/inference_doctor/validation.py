"""Validation used by the actual inference worker, not a parallel test engine."""

import numpy as np


class DiagnosticError(ValueError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.details = details


DTYPES = {
    "tensor(float)": "float32", "tensor(double)": "float64",
    "tensor(float16)": "float16", "tensor(bool)": "bool",
    **{f"tensor({t})": t for t in (
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"
    )},
}


def load_arrays(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except Exception as exc:
        raise DiagnosticError("invalid_npz", "Cannot read a numeric NPZ archive.") from exc
    if not arrays or any(a.dtype.kind not in "biuf" for a in arrays.values()):
        raise DiagnosticError("unsupported_data", "NPZ must contain numeric or boolean tensors.")
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise DiagnosticError("nonfinite_data", "Input/reference tensors must be finite.")
    return arrays


def validate_inputs(metadata, arrays):
    if set(arrays) != {m.name for m in metadata}:
        raise DiagnosticError("input_names", "Input names must exactly match the model inputs.")
    symbols = {}
    for m in metadata:
        a = arrays[m.name]
        if m.type not in DTYPES:
            raise DiagnosticError("unsupported_type", "Only numeric/boolean tensor inputs are supported.")
        if a.dtype != np.dtype(DTYPES[m.type]):
            raise DiagnosticError("input_dtype", "Input dtype does not match the model; no implicit cast is applied.",
                                  {"input": m.name, "expected": DTYPES[m.type], "actual": str(a.dtype)})
        if a.ndim != len(m.shape):
            raise DiagnosticError("input_rank", "Input rank does not match the model.",
                                  {"input": m.name, "expected_rank": len(m.shape), "actual_shape": list(a.shape)})
        for actual, expected in zip(a.shape, m.shape):
            if actual <= 0:
                raise DiagnosticError("empty_dimension", "Zero-sized dimensions are outside this MVP's scope.")
            if isinstance(expected, int) and actual != expected:
                raise DiagnosticError("input_shape", "Input shape does not match a fixed model dimension.",
                                      {"input": m.name, "expected_shape": m.shape, "actual_shape": list(a.shape)})
            if isinstance(expected, str) and expected:
                if symbols.setdefault(expected, actual) != actual:
                    raise DiagnosticError("symbolic_shape", "Shared symbolic dimensions have inconsistent sizes.")


def compare_outputs(names, outputs, reference, rtol, atol):
    if len(names) != len(outputs) or set(names) != set(reference):
        raise DiagnosticError("output_names", "Reference names must exactly match all model outputs.")
    details = []
    for name, actual in zip(names, outputs):
        expected = reference[name]
        if not isinstance(actual, np.ndarray) or actual.shape != expected.shape:
            raise DiagnosticError("output_shape", "Output/reference shapes must match exactly; broadcasting is forbidden.")
        if actual.dtype != expected.dtype:
            raise DiagnosticError("output_dtype", "Output/reference dtypes must match exactly.")
        if not np.isfinite(actual).all():
            raise DiagnosticError("nonfinite_output", "Model produced a non-finite output.")
        if actual.dtype.kind in "biu":
            passed = np.array_equal(actual, expected)
        else:
            passed = bool(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False))
        # Cast before subtraction to avoid integer overflow in diagnostics.
        error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if actual.size else 0.0
        details.append({"name": name, "shape": list(actual.shape), "dtype": str(actual.dtype),
                        "passed": passed, "max_absolute_error": error if np.isfinite(error) else None})
    if not all(d["passed"] for d in details):
        raise DiagnosticError("output_mismatch", "At least one output differs from the reference beyond tolerance.",
                              {"outputs": details, "rtol": rtol, "atol": atol})
    return details
