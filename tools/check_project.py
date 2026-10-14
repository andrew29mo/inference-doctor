"""Verify committed measurement evidence, reproducible fixtures and publication hygiene."""

import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

from inference_doctor.runner import artifact, source_hash

ROOT = Path(__file__).resolve().parents[1]
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


def verify_report(report, fixture):
    require(report["source_sha256"] == source_hash(), "Report source hash is stale")
    for key, name in [("model", "model.onnx"), ("inputs", "inputs.npz"), ("reference", "reference.npz")]:
        require(report["artifacts"][key] == artifact(fixture / name), f"Fixture differs: {key}")
    settings = report["settings"]
    for provider in report["providers"]:
        if "error" in provider:
            require(any(d["severity"] == "error" and d["provider"] == provider["requested"]
                        for d in report["diagnostics"]), "Provider error is hidden")
            continue
        require(len(provider["runs"]) == report["repeats"], "Trial count differs")
        samples = []
        peaks = []
        for run in provider["runs"]:
            require(run["registered_providers"][0] == provider["requested"], "Primary provider mismatch")
            require(len(run["latency_samples_ms"]) == settings["iterations"], "Call count differs")
            require(all(t > 0 and math.isfinite(t) for t in run["latency_samples_ms"]), "Invalid timing sample")
            require(run["correctness"]["status"] == "passed", "Unverified reference")
            require(run["correctness"]["checked_calls"] == 1 + settings["warmup"] + settings["iterations"], "Unchecked calls")
            require(all(o["passed"] for o in run["correctness"]["outputs"]), "Output failed")
            samples.extend(run["latency_samples_ms"])
            memory = run["memory"]
            require(memory["process_high_water_rss_bytes"] >= max(memory["rss_bytes_by_stage"].values()), "RSS high-water below a stage")
            peaks.append(max(memory["process_high_water_rss_bytes"], memory["parent_sampled_peak_rss_bytes"]))
        summary = provider["summary"]
        verify_summary(summary["pooled_warm_latency_ms"], samples)
        for name in ("session_creation_ms", "first_run_ms"):
            verify_summary(summary[name], [r[name] for r in provider["runs"]])
        require(summary["peak_process_rss_bytes"] == max(peaks), "RSS summary differs")
        require(summary["per_run_p95_ms"] == [quantile(r["latency_samples_ms"], .95) for r in provider["runs"]], "Trial p95 differs")
    severities = {d["severity"] for d in report["diagnostics"]}
    expected = "failed" if "error" in severities else "warning" if "warning" in severities else "passed"
    require(report["status"] == expected, "Diagnosis status differs")


def main():
    index_path = ROOT / "results/macos-arm64/index.json"
    index = json.loads(index_path.read_text())
    mlp_model_hashes = set()
    for case in index["cases"]:
        path = index_path.parent / case["report"]
        report = json.loads(path.read_text())
        if case["kind"] == "mlp":
            mlp_model_hashes.add(report["artifacts"]["model"]["sha256"])
        require(report["repeats"] == 3 and report["settings"]["iterations"] == 500
                and report["settings"]["warmup"] == 50 and report["settings"]["threads"] == 1,
                "Recorded settings differ from results overview")
        with tempfile.TemporaryDirectory() as directory:
            fixture = generator.generate(directory, case["kind"], case["batch"])
            verify_report(report, fixture)
        require(case["status"] == report["status"], "Index status differs")
        require(case["exit_code"] == (2 if report["status"] == "failed" else 0), "Index exit code differs")
        require(path.with_suffix(".html").exists(), "Missing HTML report")
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
