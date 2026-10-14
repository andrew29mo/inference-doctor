"""Process isolation, artifact provenance, and conservative diagnoses."""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import psutil


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summary(values):
    return {"count": len(values), "min": min(values), "median": percentile(values, .5),
            "p95": percentile(values, .95), "p99": percentile(values, .99), "max": max(values)}


def artifact(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def source_hash():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()


def run_worker(config, timeout):
    with tempfile.TemporaryDirectory(prefix="inference-doctor-") as directory:
        config = dict(config, workdir=directory)
        config_path, result_path = Path(directory) / "config.json", Path(directory) / "result.json"
        config_path.write_text(json.dumps(config))
        env = dict(os.environ)
        # Fix numerical library thread configuration consistently for every child.
        env.update({name: str(config["threads"]) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
        )})
        with tempfile.TemporaryFile() as logs:
            start = time.monotonic()
            optimization = ["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []
            process = subprocess.Popen([sys.executable, *optimization, "-m", "inference_doctor.worker",
                                        str(config_path), str(result_path)],
                                       env=env, stdout=logs, stderr=logs)
            child = psutil.Process(process.pid)
            peak, samples = 0, 0
            timed_out = False
            while process.poll() is None:
                try:
                    peak = max(peak, child.memory_info().rss)
                    samples += 1
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() - start > timeout:
                    process.kill()
                    timed_out = True
                    break
                time.sleep(.002)
            process.wait()
        if timed_out:
            return {"ok": False, "error": {"code": "worker_timeout", "message": "Worker exceeded the per-process timeout."}}
        if process.returncode or not result_path.exists():
            return {"ok": False, "error": {"code": "worker_failed", "message": "Worker terminated without a complete result.",
                                            "returncode": process.returncode}}
        result = json.loads(result_path.read_text())
        if result["ok"]:
            result["result"]["memory"].update({"parent_sampled_peak_rss_bytes": peak,
                                               "parent_rss_samples": samples,
                                               "parent_sampling_interval_ms": 2})
            result["result"]["worker_wall_ms"] = (time.monotonic() - start) * 1000
        return result


def diagnose(config, providers, repeats, timeout, max_peak_mib=None, max_p95_ms=None):
    # One frozen copy per artifact shared by every timing and profiling child.
    # Snapshotting/preflight happen in the controller and are outside worker metrics.
    with tempfile.TemporaryDirectory(prefix="inference-doctor-inputs-") as directory:
        frozen = dict(config)
        for key in ("model", "inputs", "reference"):
            if config.get(key):
                source = Path(config[key])
                target = Path(directory) / key / source.name
                target.parent.mkdir()
                try:
                    shutil.copyfile(source, target)
                    frozen[key] = str(target)
                except OSError:
                    # The implementation below emits a structured unreadable-artifact report.
                    frozen[key] = str(target.parent / "unreadable")
        return _diagnose(frozen, providers, repeats, timeout, max_peak_mib, max_p95_ms)


def _diagnose(config, providers, repeats, timeout, max_peak_mib=None, max_p95_ms=None):
    from datetime import datetime, timezone
    from . import __version__
    report = {"schema_version": 1, "tool_version": __version__, "source_sha256": source_hash(),
              "generated_utc": datetime.now(timezone.utc).isoformat(), "artifacts": {},
              "settings": {k: config[k] for k in ("iterations", "warmup", "threads", "rtol", "atol")},
              "repeats": repeats, "budgets": {"max_peak_mib": max_peak_mib, "max_p95_ms": max_p95_ms},
              "providers": [], "diagnostics": []}
    diagnostics = report["diagnostics"]

    def issue(severity, code, message, provider=None):
        diagnostics.append({"severity": severity, "code": code, "provider": provider, "message": message})

    try:
        for key in ("model", "inputs", "reference"):
            if config.get(key):
                report["artifacts"][key] = artifact(config[key])
    except OSError:
        issue("error", "artifact_unreadable", "A model/input/reference artifact cannot be read.")
    if not any(d["severity"] == "error" for d in diagnostics):
        try:
            from .model import inspect_model
            report["model_inspection"] = inspect_model(config["model"])
        except Exception as exc:
            from .validation import DiagnosticError
            issue("error", exc.code if isinstance(exc, DiagnosticError) else "invalid_model",
                  str(exc) if isinstance(exc, DiagnosticError) else "ONNX preflight validation failed.")
    if config.get("reference") is None:
        issue("warning", "no_reference", "No independent reference supplied; output correctness is NOT verified.")
    if not diagnostics or not any(d["severity"] == "error" for d in diagnostics):
        for provider in providers:
            entry = {"requested": provider, "runs": [], "profile": None}
            report["providers"].append(entry)
            for _ in range(repeats):
                outcome = run_worker(dict(config, provider=provider, mode="measure"), timeout)
                if not outcome["ok"]:
                    issue("error", outcome["error"]["code"], outcome["error"]["message"], provider)
                    entry["error"] = outcome["error"]
                    break
                entry["runs"].append(outcome["result"])
            if len(entry["runs"]) != repeats:
                continue
            entry["summary"] = {
                "session_creation_ms": summary([r["session_creation_ms"] for r in entry["runs"]]),
                "first_run_ms": summary([r["first_run_ms"] for r in entry["runs"]]),
                "pooled_warm_latency_ms": summary([t for r in entry["runs"] for t in r["latency_samples_ms"]]),
                "per_run_p95_ms": [percentile(r["latency_samples_ms"], .95) for r in entry["runs"]],
                "peak_process_rss_bytes": max(max(r["memory"]["process_high_water_rss_bytes"],
                                                  r["memory"]["parent_sampled_peak_rss_bytes"])
                                              for r in entry["runs"]),
            }
            s = entry["summary"]
            if max_peak_mib is not None and s["peak_process_rss_bytes"] > max_peak_mib * 1024**2:
                issue("error", "memory_budget", "At least one trial exceeded the process RSS budget.", provider)
            if max_p95_ms is not None and max(s["per_run_p95_ms"]) > max_p95_ms:
                issue("error", "latency_budget", "At least one trial exceeded the warm-call p95 budget.", provider)
            outcome = run_worker(dict(config, provider=provider, mode="profile"), timeout)
            if not outcome["ok"]:
                issue("warning", "profiling_failed", "Separate profiling pass failed; execution assignment is unknown.", provider)
                continue
            entry["profile"] = outcome["result"]["profile"]
            counts = entry["profile"]["provider_kernel_events"]
            if not counts or entry["profile"]["unattributed_kernel_events"]:
                issue("warning", "incomplete_assignment", "Profiling does not fully attribute observed kernel events.", provider)
            if provider != "CPUExecutionProvider":
                if counts.get("CPUExecutionProvider", 0):
                    issue("warning", "cpu_partition_observed", "CPU kernel events were observed in the separate profiling session.", provider)
                if not counts.get(provider, 0):
                    issue("warning", "requested_ep_unobserved", "No kernel events attributed to the requested provider were observed.", provider)
                issue("info", "hardware_unverified", "Execution-provider attribution does not prove physical GPU/NPU/ANE execution.", provider)
    report["status"] = "failed" if any(d["severity"] == "error" for d in diagnostics) else (
        "warning" if any(d["severity"] == "warning" for d in diagnostics) else "passed")
    return report
