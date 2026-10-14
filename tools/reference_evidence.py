"""Lossless numeric serialization of the synthetic suite's measured references."""

import json
from pathlib import Path

import numpy as np


def save_reference(source, destination):
    with np.load(source, allow_pickle=False) as archive:
        tensors = [{"name": name, "dtype": str(archive[name].dtype), "shape": list(archive[name].shape),
                    "values": archive[name].tolist()} for name in archive.files]
    Path(destination).write_text(json.dumps({"format": 1, "tensors": tensors}, indent=2, allow_nan=False) + "\n")


def load_reference(path):
    evidence = json.loads(Path(path).read_text())
    if evidence["format"] != 1:
        raise ValueError("Unsupported reference evidence format")
    return {t["name"]: np.array(t["values"], dtype=t["dtype"]).reshape(t["shape"])
            for t in evidence["tensors"]}
