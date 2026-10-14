"""Verify committed measurement evidence, reproducible fixtures and publication hygiene."""

import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import sys

import numpy as np

from inference_doctor.runner import artifact, source_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from reference_evidence import load_reference
from suite_policy import ATOL, RTOL, REPEATS, WARMUP, ITERATIONS, THREADS, CASES, providers_for
from inference_doctor.report import write_report, LIMITATIONS
from inference_doctor import __version__
from inference_doctor.model import inspect_model
spec = importlib.util.spec_from_file_location("make_examples", ROOT / "tools/make_examples.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def quantile(values, fraction):
    data = sorted(values)
    index = (len(data) - 1) * fraction
    lower = math.floor(index)
    return data[lower] + (data[math.ceil(index)] - data[lower]) * (index - lower)


def verify_summary(actual, values):
    expected = {"count": len(values), "min": min(values), "max": max(values),
                "median": quantile(values, .5), "p95": quantile(values, .95), "p99": quantile(values, .99)}
    require(set(actual) == set(expected), "Summary keys differ")
    for key, value in expected.items():
        require(math.isclose(actual[key], value, rel_tol=1e-12, abs_tol=1e-12), f"Invalid summary: {key}")


def verify_report(report, fixture, reference_path):
    require(report["source_sha256"] == source_hash(), "Report source hash is stale")
    require(report["schema_version"] == 2, "Unsupported report schema")
    require(report["tool_version"] == __version__, "Tool version differs")
    require(report["model_inspection"] == inspect_model(fixture / "model.onnx"), "Model metadata differs")
    policy = {"rtol": RTOL, "atol": ATOL, "iterations": ITERATIONS, "warmup": WARMUP, "threads": THREADS}
    require(report["settings"] == policy, "Recorded settings/tolerances differ from fixed suite policy")
    require(report["repeats"] == REPEATS, "Recorded trial count differs from policy")
    require(report["budgets"] == {"max_peak_mib": None, "max_p95_ms": None}, "Unexpected recorded budgets")
    require(report["measurement_scope"] == LIMITATIONS, "Measurement scope differs from current source")
    manifest = json.loads((fixture / "manifest.json").read_text())
    expected_providers = providers_for(manifest["kind"], manifest["batch"])
    require([p["requested"] for p in report["providers"]] == expected_providers, "Provider inventory differs")
    expected_order = []
    for trial in range(REPEATS):
        order = expected_providers if trial % 2 == 0 else list(reversed(expected_providers))
        expected_order.extend({"mode": "measure", "provider": p, "trial": trial + 1} for p in order)
    expected_order.extend({"mode": "profile", "provider": p, "trial": 1} for p in expected_providers)
    require(report["execution_order"] == expected_order, "Timing/profile order differs")
    for key, name in [("model", "model.onnx"), ("inputs", "inputs.npz")]:
        require(report["artifacts"][key] == artifact(fixture / name), f"Fixture differs: {key}")
    canonical = load_reference(reference_path)
    require(bool(canonical), "Empty reference evidence")
    with tempfile.TemporaryDirectory() as directory:
        restored = Path(directory) / "reference.npz"
        np.savez(restored, **canonical)
        require(report["artifacts"]["reference"] == artifact(restored), "Recorded reference integrity differs")
    reference_drift = 0.0
    with np.load(fixture / "reference.npz", allow_pickle=False) as regenerated:
        require(set(regenerated.files) == set(canonical), "Reference output names differ")
        for name, expected in canonical.items():
            actual = regenerated[name]
            require(actual.shape == expected.shape and actual.dtype == expected.dtype, "Reference tensor metadata differs")
            require(np.isfinite(expected).all(), "Nonfinite canonical reference")
            require(np.allclose(actual, expected, rtol=RTOL, atol=ATOL),
                    f"Independent recomputed reference differs beyond tolerance: {name}")
            reference_drift = max(reference_drift, float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))))
    with np.load(fixture / "inputs.npz", allow_pickle=False) as inputs:
        input_metadata = [{"name": k, "shape": list(inputs[k].shape), "dtype": str(inputs[k].dtype), "bytes": inputs[k].nbytes}
                          for k in inputs.files]

    def verify_correctness(evidence, calls):
        require(evidence["status"] == "passed", "Unverified reference")
        require(type(evidence["checked_calls"]) is int and evidence["checked_calls"] == calls, "Unchecked calls")
        outputs = evidence["outputs"]
        require(sorted(o["name"] for o in outputs) == sorted(canonical), "Incomplete output evidence")
        for output in outputs:
            expected = canonical[output["name"]]
            require(output["passed"] is True, "Output failed")
            require(output["shape"] == list(expected.shape) and output["dtype"] == str(expected.dtype), "Output metadata differs")
            error = output["max_absolute_error"]
            limit = ATOL + RTOL * float(np.max(np.abs(expected)))
            require(isinstance(error, (float, int)) and math.isfinite(error) and 0 <= error <= limit, "Output error contradicts tolerance")

    expected_diagnostics = []
    environment = None
    for provider in report["providers"]:
        requested = provider["requested"]
        require("error" not in provider and "profile_error" not in provider, "Recorded suite contains failed evidence")
        require(len(provider["runs"]) == REPEATS, "Trial count differs")
        samples, peaks = [], []
        for run in provider["runs"]:
            require(run["provider"] == requested and run["registered_providers"][0] == requested, "Primary provider mismatch")
            require(run["inputs"] == input_metadata, "Input tensor metadata differs")
            if environment is None:
                environment = run["environment"]
            require(run["environment"] == environment, "Timing trial software/hardware environments differ")
            require(len(run["latency_samples_ms"]) == ITERATIONS, "Call count differs")
            require(all(t > 0 and math.isfinite(t) for t in run["latency_samples_ms"]), "Invalid timing sample")
            verify_correctness(run["correctness"], 1 + WARMUP + ITERATIONS)
            samples.extend(run["latency_samples_ms"])
            memory = run["memory"]
            values = [memory["process_high_water_rss_bytes"], memory["parent_sampled_peak_rss_bytes"], *memory["rss_bytes_by_stage"].values()]
            require(all(type(v) is int and v > 0 for v in values), "Invalid RSS observation")
            peaks.append(max(values))
        summary = provider["summary"]
        verify_summary(summary["pooled_warm_latency_ms"], samples)
        for name in ("session_creation_ms", "first_run_ms"):
            require(all(math.isfinite(r[name]) and r[name] > 0 for r in provider["runs"]), "Invalid startup timing")
            verify_summary(summary[name], [r[name] for r in provider["runs"]])
        require(summary["peak_process_rss_bytes"] == max(peaks), "RSS summary differs")
        require(summary["per_run_p95_ms"] == [quantile(r["latency_samples_ms"], .95) for r in provider["runs"]], "Trial p95 differs")
        verify_correctness(provider["profile_correctness"], 1)
        profile = provider["profile"]
        require(isinstance(profile, dict), "Missing profile evidence")
        counts, unknown = {}, 0
        for node in profile["nodes"]:
            ep = node["provider"]
            if ep:
                counts[ep] = counts.get(ep, 0) + 1
            else:
                unknown += 1
        require(counts == profile["provider_kernel_events"] and unknown == profile["unattributed_kernel_events"], "Profile counts differ from node evidence")
        if not counts or unknown:
            expected_diagnostics.append(("warning", "incomplete_assignment", requested))
        if not counts.get(requested, 0):
            expected_diagnostics.append(("warning", "requested_ep_unobserved", requested))
        if requested != "CPUExecutionProvider":
            if counts.get("CPUExecutionProvider", 0):
                expected_diagnostics.append(("warning", "cpu_partition_observed", requested))
            expected_diagnostics.append(("info", "hardware_unverified", requested))
    observed = [(d["severity"], d["code"], d["provider"]) for d in report["diagnostics"]]
    require(observed == expected_diagnostics, "Diagnostics differ from profile evidence")
    expected_status = "warning" if any(d[0] == "warning" for d in expected_diagnostics) else "passed"
    require(report["status"] == expected_status, "Diagnosis status differs")
    return reference_drift


def main():
    index_path = ROOT / "results/macos-arm64/index.json"
    index = json.loads(index_path.read_text())
    mlp_model_hashes = set()
    require([(c["kind"], c["batch"]) for c in index["cases"]] == list(CASES), "Suite case inventory differs")
    for case in index["cases"]:
        require(case["name"] == f'{case["kind"]}-b{case["batch"]}' and case["report"] == f'{case["name"]}/report.json', "Case identity differs")
        path = index_path.parent / case["report"]
        report = json.loads(path.read_text())
        if case["kind"] == "mlp":
            mlp_model_hashes.add(report["artifacts"]["model"]["sha256"])
        with tempfile.TemporaryDirectory() as directory:
            fixture = generator.generate(directory, case["kind"], case["batch"])
            drift = verify_report(report, fixture, path.parent / "reference.json")
            print(f'{case["name"]}: regenerated reference max absolute difference {drift:.3g}; fixed rtol={RTOL:g}, atol={ATOL:g}')
        require(case["status"] == report["status"], "Index status differs")
        require(case["exit_code"] == (2 if report["status"] == "failed" else 0), "Index exit code differs")
        with tempfile.TemporaryDirectory() as directory:
            write_report(dict(report), directory)
            require(path.with_suffix(".html").read_text() == (Path(directory) / "report.html").read_text(), "HTML differs from verified JSON")
    require(len(mlp_model_hashes) == 1, "MLP weights/graph differ across batch sizes")
    overview_spec = importlib.util.spec_from_file_location("render_results", ROOT / "tools/render_results.py")
    overview = importlib.util.module_from_spec(overview_spec)
    overview_spec.loader.exec_module(overview)
    require((ROOT / "results/README.md").read_text() == overview.render(), "Results overview is stale")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    private_path = re.compile(rb"/(?:Users|home)/[A-Za-z0-9_.-]+/|[A-Za-z]:\\Users\\")
    secrets = re.compile(rb"github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
    forbidden = {"AGENTS.md", ".DS_Store", ".env", "id_rsa", "id_ed25519"}
    require(bool([p for p in tracked if p]), "No tracked files; stage intended publication files first")
    for raw_path in tracked:
        if not raw_path:
            continue
        relative = raw_path.decode()
        path = ROOT / relative
        require(not path.is_symlink(), f"Tracked symlink: {relative}")
        require(not any(part in forbidden or part.startswith(".env.") or part in (".venv", "__pycache__")
                        for part in path.relative_to(ROOT).parts), f"Private/generated file: {relative}")
        require(path.suffix not in (".npz", ".onnx", ".so", ".dylib", ".pyc", ".whl"), f"Generated binary: {relative}")
        data = path.read_bytes()
        require(b"\0" not in data, f"Unexpected binary: {relative}")
        data.decode("utf-8")
        require(not private_path.search(data), f"Private path: {relative}")
        require(not secrets.search(data), f"Credential pattern: {relative}")
        if path.suffix == ".md":
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", data.decode("utf-8")):
                if "://" not in target and not target.startswith("#"):
                    require((path.parent / target.split("#", 1)[0]).exists(), f"Broken local link in {relative}: {target}")
    print(f"Verified {len(index['cases'])} measured cases, fixture hashes, source provenance, summaries and tracked-file hygiene.")


if __name__ == "__main__":
    main()
