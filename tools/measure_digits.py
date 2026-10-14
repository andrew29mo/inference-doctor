"""Measure the same trained model and full held-out dataset at multiple batches."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from digits_policy import BATCHES, REPEATS, WARMUP, ITERATIONS, RTOL, ATOL
from reference_evidence import save_reference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/digits"))
from prepare import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/local/digits")
    parser.add_argument("--include-coreml", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()
    if any(b <= 0 for b in args.batches) or args.iterations <= 0 or args.repeats <= 0:
        parser.error("batches, iterations and repeats must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fixture = ROOT / "examples/generated/digits"
    prepare(fixture)
    save_reference(fixture / "reference.npz", output / "reference.json")
    save_reference(fixture / "labels.npz", output / "labels.json")
    shutil.copyfile(fixture / "manifest.json", output / "manifest.json")
    index = {"experiment": "trained digits MLP; same artifacts and complete held-out data for every case", "cases": []}
    for batch in args.batches:
        case = f"b{batch}"
        command = [sys.executable, "-m", "inference_doctor", str(fixture / "model.onnx"),
                   "--inputs", str(fixture / "inputs.npz"), "--reference", str(fixture / "reference.npz"),
                   "--labels", str(fixture / "labels.npz"), "--batch-size", str(batch),
                   "--classification-output", "probabilities", "--iterations", str(args.iterations),
                   "--warmup", str(WARMUP), "--repeats", str(args.repeats), "--threads", "1",
                   "--rtol", str(RTOL), "--atol", str(ATOL), "--output", str(output / case),
                   "--provider", "CPUExecutionProvider"]
        if args.include_coreml:
            command += ["--provider", "CoreMLExecutionProvider"]
        completed = subprocess.run(command, check=False)
        report = json.loads((output / case / "report.json").read_text())
        index["cases"].append({"batch_size": batch, "report": f"{case}/report.json", "status": report["status"], "exit_code": completed.returncode})
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return int(any(c["exit_code"] != 0 for c in index["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
