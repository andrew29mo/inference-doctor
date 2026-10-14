"""Verify real-model lineage and every saved trial/pass against fixed policy."""

import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

from check_project import require, verify_summary, quantile
from digits_policy import BATCHES, REPEATS, WARMUP, ITERATIONS, RTOL, ATOL, ROWS, CORRECT
from reference_evidence import load_reference
from inference_doctor import __version__
from inference_doctor.runner import source_hash, artifact
from inference_doctor.report import write_report, LIMITATIONS
from inference_doctor.model import inspect_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/digits"))
from prepare import prepare


def fingerprint(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def verify_case(report, fixture, canonical, expected_batch):
    require(report["schema_version"] == 2 and report["tool_version"] == __version__, "Report version differs")
    require(report["source_sha256"] == source_hash(), "Source differs")
    require(report["model_inspection"] == inspect_model(fixture / "model.onnx"), "Model inspection differs")
    require(report["settings"] == {"iterations": ITERATIONS, "warmup": WARMUP, "threads": 1, "rtol": RTOL, "atol": ATOL}, "Dataset settings differ")
    require(report["repeats"] == REPEATS, "Dataset trial count differs")
    require(report["budgets"] == {"max_peak_mib": None, "max_p95_ms": None}, "Unexpected budgets")
    require(report["measurement_scope"] == LIMITATIONS, "Measurement scope differs")
    for name in ("model", "inputs", "labels"):
        require(report["artifacts"][name] == artifact(fixture / f"{name}.{'onnx' if name == 'model' else 'npz'}"), "Dataset artifacts differ")
    with tempfile.TemporaryDirectory() as directory:
        restored = Path(directory) / "reference.npz"
        np.savez(restored, **canonical)
        require(report["artifacts"]["reference"] == artifact(restored), "Measured reference integrity differs")
    with np.load(fixture / "reference.npz") as archive:
        require(set(archive.files) == set(canonical) == {"logits", "probabilities"}, "Reference output inventory differs")
        for name in archive.files:
            require(archive[name].shape == canonical[name].shape == (ROWS, 10), "Reference shape differs")
            require(archive[name].dtype == canonical[name].dtype == np.dtype("float32"), "Reference dtype differs")
            require(np.isfinite(canonical[name]).all() and np.allclose(archive[name], canonical[name], rtol=RTOL, atol=ATOL), "Recomputed reference differs beyond fixed tolerance")
    with np.load(fixture / "labels.npz") as archive:
        y, ids, classes = (archive[k] for k in ("labels", "sample_ids", "classes"))
    reference_predictions = classes[canonical["probabilities"].argmax(axis=1)]
    require(int(np.sum(reference_predictions == y)) == CORRECT, "Frozen model accuracy changed")
    id_hash, prediction_hash = fingerprint(ids), fingerprint(reference_predictions)
    batches = math.ceil(ROWS / expected_batch)
    pattern = [min(expected_batch, ROWS - start) for start in range(0, ROWS, expected_batch)]
    expected_data = {"rows": ROWS, "batch_size": expected_batch, "batches_per_pass": batches,
                     "last_batch_rows": pattern[-1], "classification_output": "probabilities", "classes": classes.tolist(),
                     "reference_correct": CORRECT, "reference_accuracy": CORRECT / ROWS,
                     "sample_ids_sha256": id_hash, "reference_predictions_sha256": prediction_hash}
    require(report["dataset"] == expected_data, "Dataset metadata differs")
    providers = ["CPUExecutionProvider", "CoreMLExecutionProvider"]
    require([p["requested"] for p in report["providers"]] == providers, "Missing CPU/CoreML comparison")
    order = [{"mode": "measure", "provider": p, "trial": trial + 1} for trial in range(REPEATS)
             for p in (providers if trial % 2 == 0 else list(reversed(providers)))]
    order += [{"mode": "profile", "provider": p, "trial": 1} for p in providers]
    require(report["execution_order"] == order, "Dataset provider order differs")

    def check_pass(evidence):
        require(evidence["valid_rows"] == ROWS and evidence["calls"] == batches, "Incomplete dataset pass")
        require(evidence["correct"] == CORRECT and evidence["accuracy"] == CORRECT / ROWS, "Accuracy differs")
        require(evidence["sample_ids_sha256"] == id_hash and evidence["predictions_sha256"] == prediction_hash, "Coverage/prediction fingerprints differ")

    def check_outputs(evidence, count):
        require(evidence["status"] == "passed" and evidence["checked_calls"] == count, "Unchecked dataset calls")
        require(sorted(o["name"] for o in evidence["outputs"]) == sorted(canonical), "Missing output evidence")
        for o in evidence["outputs"]:
            require(o["passed"] is True and o["shape"] == [pattern[-1], 10] and o["dtype"] == "float32", "Last-batch output evidence differs")
            limit = ATOL + RTOL * float(np.max(np.abs(canonical[o["name"]][-pattern[-1]:])))
            require(math.isfinite(o["max_absolute_error"]) and 0 <= o["max_absolute_error"] <= limit, "Output error contradicts the fixed tolerance")

    diagnostics = []
    environment = None
    for entry in report["providers"]:
        provider = entry["requested"]
        require("error" not in entry and "profile_error" not in entry and len(entry["runs"]) == REPEATS, "Failed/incomplete dataset trials")
        samples, peaks, inference_per_example, wall_per_example = [], [], [], []
        for run in entry["runs"]:
            require(run["provider"] == provider and run["registered_providers"][0] == provider, "Primary provider differs")
            environment = run["environment"] if environment is None else environment
            require(run["environment"] == environment, "CPU/CoreML environments differ")
            require(run["inputs"] == [{"name": "X", "shape": [ROWS, 64], "dtype": "float32", "bytes": ROWS * 64 * 4}], "Input metadata differs")
            times = run["latency_samples_ms"]
            require(len(times) == ITERATIONS * batches and all(t > 0 and math.isfinite(t) for t in times), "Dataset call timing count differs")
            require(run["latency_sample_rows"] == pattern * ITERATIONS, "Tail batch is incomplete, padded or reordered")
            require(run["dataset"]["metadata"] == expected_data, "Worker dataset differs")
            check_outputs(run["correctness"], 1 + batches + WARMUP + ITERATIONS * batches)
            check_pass(run["dataset"]["evaluation"])
            passes = run["dataset"]["timed_passes"]
            require(len(passes) == ITERATIONS, "Dataset pass count differs")
            for index, evidence in enumerate(passes):
                check_pass(evidence)
                total = sum(times[index * batches:(index + 1) * batches])
                require(math.isclose(evidence["inference_ms"], total, rel_tol=1e-12), "Pass time differs from call samples")
                require(math.isfinite(evidence["validated_wall_ms"]) and evidence["validated_wall_ms"] >= total, "Invalid validated-wall time")
            inference_per_example.append(sum(times) * 1000 / (ROWS * ITERATIONS))
            wall_per_example.append(sum(p["validated_wall_ms"] for p in passes) * 1000 / (ROWS * ITERATIONS))
            samples.extend(times)
            memory = run["memory"]
            peaks.append(max(memory["process_high_water_rss_bytes"], memory["parent_sampled_peak_rss_bytes"], *memory["rss_bytes_by_stage"].values()))
        summary = entry["summary"]
        verify_summary(summary["pooled_warm_latency_ms"], samples)
        for name in ("session_creation_ms", "first_run_ms"):
            require(all(math.isfinite(r[name]) and r[name] > 0 for r in entry["runs"]), "Invalid startup timing")
            verify_summary(summary[name], [r[name] for r in entry["runs"]])
        require(summary["per_run_p95_ms"] == [quantile(r["latency_samples_ms"], .95) for r in entry["runs"]], "p95 differs")
        require(summary["peak_process_rss_bytes"] == max(peaks), "Dataset worker RSS differs")
        require(summary["dataset"] == {"per_trial_inference_us_per_example": inference_per_example,
                                       "per_trial_validated_wall_us_per_example": wall_per_example,
                                       "accuracy": CORRECT / ROWS, "correct": CORRECT, "test_rows": ROWS}, "Normalized performance/accuracy differs")
        check_outputs(entry["profile_correctness"], 1 + batches)
        check_pass(entry["profile"]["dataset_evaluation"])
        counts, unknown = {}, 0
        for node in entry["profile"]["nodes"]:
            if node["provider"]:
                counts[node["provider"]] = counts.get(node["provider"], 0) + 1
            else:
                unknown += 1
        require(counts == entry["profile"]["provider_kernel_events"] and unknown == entry["profile"]["unattributed_kernel_events"], "Profile node counts differ")
        if not counts or unknown:
            diagnostics.append(("warning", "incomplete_assignment", provider))
        if not counts.get(provider, 0):
            diagnostics.append(("warning", "requested_ep_unobserved", provider))
        if provider != "CPUExecutionProvider":
            if counts.get("CPUExecutionProvider", 0):
                diagnostics.append(("warning", "cpu_partition_observed", provider))
            diagnostics.append(("info", "hardware_unverified", provider))
    require([(d["severity"], d["code"], d["provider"]) for d in report["diagnostics"]] == diagnostics, "Dataset diagnostics overstate evidence")
    require(report["status"] == ("warning" if any(d[0] == "warning" for d in diagnostics) else "passed"), "Dataset status differs")


def main():
    base = ROOT / "results/digits-macos"
    index = json.loads((base / "index.json").read_text())
    require([c["batch_size"] for c in index["cases"]] == list(BATCHES), "Dataset batch inventory differs")
    canonical = load_reference(base / "reference.json")
    labels = load_reference(base / "labels.json")
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        manifest = prepare(fixture)
        require(json.loads((base / "manifest.json").read_text()) == manifest, "Training/dataset provenance differs")
        with np.load(fixture / "labels.npz") as archive:
            require(set(labels) == set(archive.files) and all(labels[k].dtype == archive[k].dtype and labels[k].shape == archive[k].shape and np.array_equal(labels[k], archive[k]) for k in archive.files), "Retained labels differ from installed dataset")
        for case in index["cases"]:
            require(case["report"] == f'b{case["batch_size"]}/report.json', "Unexpected dataset case path")
            path = base / case["report"]
            report = json.loads(path.read_text())
            verify_case(report, fixture, canonical, case["batch_size"])
            require(case["status"] == report["status"] and case["exit_code"] == 0, "Dataset index differs")
            with tempfile.TemporaryDirectory() as rendered:
                write_report(dict(report), rendered)
                require(path.with_suffix(".html").read_text() == (Path(rendered) / "report.html").read_text(), "Dataset HTML differs from JSON")
    from render_digits import render
    require((base / "README.md").read_text() == render(), "Dataset overview is stale")
    print("Verified trained-model lineage, four CPU/CoreML batch sizes, all outputs/rows/passes, accuracy, timings and worker RSS.")


if __name__ == "__main__":
    main()
