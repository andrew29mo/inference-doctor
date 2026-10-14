"""Run real ONNX sessions; preserve all raw samples rather than select a best run."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from make_examples import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/local/suite")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-coreml", action="store_true", help="Record CoreML outcome even if unavailable/failing")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    index = {"description": "Original synthetic graphs, actual ORT execution, independent NumPy reference.",
             "cases": []}
    for kind, batch in [("linear", 1), ("mlp", 1), ("mlp", 16), ("mlp", 64), ("residual", 16)]:
        case = f"{kind}-b{batch}"
        fixture = generate(Path("examples/generated") / case, kind, batch)
        destination = output / case
        command = [sys.executable, "-m", "inference_doctor", str(fixture / "model.onnx"),
                   "--inputs", str(fixture / "inputs.npz"), "--reference", str(fixture / "reference.npz"),
                   "--output", str(destination), "--iterations", str(args.iterations), "--warmup", "50",
                   "--repeats", str(args.repeats)]
        if args.include_coreml and case == "mlp-b16":
            command += ["--provider", "CPUExecutionProvider", "--provider", "CoreMLExecutionProvider"]
        completed = subprocess.run(command, check=False)
        report = json.loads((destination / "report.json").read_text())
        index["cases"].append({"name": case, "kind": kind, "batch": batch,
                               "report": f"{case}/report.json", "status": report["status"],
                               "exit_code": completed.returncode})
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return int(any(c["exit_code"] != 0 for c in index["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
