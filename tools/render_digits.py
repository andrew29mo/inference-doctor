"""Create the real-model overview directly from measured evidence."""

import argparse
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def render():
    base = ROOT / "results/digits-macos"
    index = json.loads((base / "index.json").read_text())
    manifest = json.loads((base / "manifest.json").read_text())
    lines = ["# Real-model dataset measurements", "",
             f'Trained 64→32 ReLU→10 Softmax MLP on {manifest["training_rows"]} digits; all {manifest["test_rows"]} held-out rows are scored. Independent sklearn reference: **{manifest["correct"]}/{manifest["test_rows"]} ({manifest["accuracy"]*100:.2f}%)**.', "",
             "All cases share identical ONNX model, input, reference and label artifacts. Four fresh trials per requested provider alternate CPU/CoreML order; profiling follows all timing trials within each case. Each trial performs a first batch, one untimed complete evaluation, five warmup calls on the first batch, and five timed complete passes.", "",
             "**Inference µs/example** is the median of the four trial means: sum of actual synchronous call times divided by 450×5 valid examples. **Validated wall** also includes slicing, output validation, class scoring and coverage hashes. Neither is raw kernel throughput. Short final batches are computed and checked, with no padding.", "",
             "| Maximum batch | Calls/pass | Final rows | Requested primary | Correct | Inference µs/example | Validated wall µs/example | Peak worker RSS MiB |",
             "|---:|---:|---:|---|---:|---:|---:|---:|"]
    host = None
    for case in index["cases"]:
        report = json.loads((base / case["report"]).read_text())
        host = report["host"] if host is None else host
        meta = report.get("dataset", {})
        for provider in report["providers"]:
            summary = provider.get("summary", {})
            dataset = summary.get("dataset")
            if dataset:
                lines.append(f'| {case["batch_size"]} | {meta["batches_per_pass"]} | {meta["last_batch_rows"]} | {provider["requested"]} | '
                             f'{dataset["correct"]}/{dataset["test_rows"]} | {median(dataset["per_trial_inference_us_per_example"]):.3f} | '
                             f'{median(dataset["per_trial_validated_wall_us_per_example"]):.3f} | {summary["peak_process_rss_bytes"]/1024**2:.2f} |')
            else:
                lines.append(f'| {case["batch_size"]} | — | — | {provider["requested"]} | Failed | — | — | — |')
    if host:
        lines += ["", f'Reported host: {host["cpu_model"] or "unknown CPU"}, {host["os"]} {host["os_version"]}, {host["machine"]}.']
    lines += ["", "Each report retains raw call times paired with actual row counts, every pass's correctness/coverage/prediction fingerprint, startup/first-call timing, per-trial normalized performance and RSS stages. These are descriptive measurements on one machine, without thermal control or statistical ranking claims.", "",
              "**RSS is the whole validated worker, not model-only RAM or total deployment memory.** It includes the full loaded test/reference arrays and checks, and excludes controller preflight, profiling processes, CoreML helper services and device memory. Differences across batches can be allocator/system noise. EP attribution is not proof of Neural Engine/GPU use or identical assignment in every timing trial.", "",
              "The split is local to sklearn's bundled 1,797-row subset, not the original UCI evaluation protocol or a verified writer-independent split. There is no hyperparameter search on the held-out set. Pixel normalization is fixed division by 16, with no fitted preprocessing. Exact image duplicates and ID overlaps across the split are rejected.", "",
              "## Evidence", "",
              "- [Training/test provenance and reference accuracy](manifest.json)",
              "- [Retained reference tensors](reference.json)",
              "- [Retained test labels, row IDs and class mapping](labels.json)",
              "- [Frozen numeric sklearn state](../../examples/digits/model.json)",
              "- [Preparation and export](../../examples/digits/prepare.py)"]
    for case in index["cases"]:
        batch = case["batch_size"]
        lines.append(f'- Batch {batch}: [HTML](b{batch}/report.html) · [JSON](b{batch}/report.json) · `{case["status"]}`')
    lines += ["", "Download/open HTML locally; GitHub does not host the report UI.", "",
              "Reproduce with `python tools/measure_digits.py --output results/local/digits --include-coreml`; install `requirements-real-model.txt` first. Check recorded evidence with `python tools/check_digits.py`.", "",
              "[Dataset attribution](../../THIRD_PARTY_NOTICES.md) · [Methodology](../../docs/METHODOLOGY.md)", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "results/digits-macos/README.md"
    if args.check:
        if destination.read_text() != render():
            raise SystemExit("Real-model overview is stale")
    else:
        destination.write_text(render())
