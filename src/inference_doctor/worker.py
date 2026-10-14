"""One fresh process per timing trial or separate profiling pass."""

import json
import platform
import resource
import sys
import time
from pathlib import Path

import psutil


def rss():
    return psutil.Process().memory_info().rss


def measure(config):
    baseline = rss()
    import numpy as np
    import onnxruntime as ort
    from .validation import DiagnosticError, compare_outputs, load_arrays, validate_inputs

    stages = {"before_numpy_ort_import": baseline, "after_import": rss()}
    provider = config["provider"]
    available = ort.get_available_providers()
    if provider not in available:
        raise DiagnosticError("provider_unavailable", "Requested execution provider is not available in this installation.",
                              {"available_providers": available})
    inputs = load_arrays(config["inputs"])
    reference = load_arrays(config["reference"]) if config.get("reference") else None
    stages["after_data_load"] = rss()
    options = ort.SessionOptions()
    options.intra_op_num_threads = config["threads"]
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    options.enable_profiling = config["mode"] == "profile"
    if options.enable_profiling:
        options.profile_file_prefix = str(Path(config["workdir"]) / "trace")
    requested = [provider] if provider == "CPUExecutionProvider" else [provider, "CPUExecutionProvider"]

    def runtime_call(phase, action):
        try:
            return action()
        except DiagnosticError:
            raise
        except Exception as exc:
            if phase == "session_creation":
                message = "Session creation failed. Check model/operator support, provider prerequisites and temporary-directory access."
            else:
                message = f"Runtime failed during {phase.replace('_', ' ')}. Check provider support for these inputs and runtime compatibility."
            raise DiagnosticError(f"{phase}_failed", message) from exc

    start = time.perf_counter_ns()
    session = runtime_call("session_creation", lambda: ort.InferenceSession(
        config["model"], sess_options=options, providers=requested))
    session_ms = (time.perf_counter_ns() - start) / 1e6
    session.disable_fallback()  # Disable run-time retry; graph partition fallback remains observable.
    registered = session.get_providers()
    if not registered or registered[0] != provider:
        raise DiagnosticError("provider_initialization", "Runtime did not register the requested primary provider.")
    stages["after_session"] = rss()
    validate_inputs(session.get_inputs(), inputs)
    names = [m.name for m in session.get_outputs()]
    start = time.perf_counter_ns()
    first = runtime_call("first_run", lambda: session.run(None, inputs))
    first_ms = (time.perf_counter_ns() - start) / 1e6
    stages["after_first_run"] = rss()
    checks = 0
    comparison = []

    def validate(outputs):
        nonlocal checks, comparison
        if reference is not None:
            comparison = compare_outputs(names, outputs, reference, config["rtol"], config["atol"])
            checks += 1
        elif any(not isinstance(a, np.ndarray) or a.dtype.kind not in "biuf" or not np.isfinite(a).all() for a in outputs):
            raise DiagnosticError("invalid_output", "Only finite numeric tensor outputs are supported.")

    validate(first)
    del first
    samples = []
    if config["mode"] == "measure":
        for _ in range(config["warmup"]):
            validate(runtime_call("warmup", lambda: session.run(None, inputs)))
        stages["after_warmup"] = rss()
        for _ in range(config["iterations"]):
            start = time.perf_counter_ns()
            outputs = runtime_call("timed_run", lambda: session.run(None, inputs))
            samples.append((time.perf_counter_ns() - start) / 1e6)
            # Every timed result is checked outside the timed interval.
            validate(outputs)
            del outputs  # Do not retain the previous call's outputs during the next timer.
    profile = None
    if options.enable_profiling:
        trace = json.loads(Path(session.end_profiling()).read_text())
        counts = {}
        nodes = []
        unknown = 0
        for event in trace:
            if event.get("cat") != "Node" or not event.get("name", "").endswith("_kernel_time"):
                continue
            ep = event.get("args", {}).get("provider")
            if ep:
                counts[ep] = counts.get(ep, 0) + 1
            else:
                unknown += 1
            nodes.append({"name": event.get("name"), "provider": ep,
                          "duration_us": event.get("dur")})
        profile = {"provider_kernel_events": counts, "unattributed_kernel_events": unknown,
                   "nodes": nodes, "scope": "Separate session, one inference; optimized/fused graph events, not original-op coverage or hardware proof."}
    stages["after_work"] = rss()
    highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        highwater *= 1024
    result = {
        "provider": provider, "available_providers": available, "registered_providers": registered,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "onnxruntime": ort.__version__, "psutil": psutil.__version__,
                        "os": platform.system(), "os_release": platform.release(),
                        "machine": platform.machine(), "logical_cpus": psutil.cpu_count(),
                        "python_optimization": sys.flags.optimize},
        "inputs": [{"name": k, "shape": list(v.shape), "dtype": str(v.dtype), "bytes": v.nbytes}
                   for k, v in inputs.items()],
        "session_creation_ms": session_ms, "first_run_ms": first_ms,
        "latency_samples_ms": samples,
        "correctness": {"status": "passed" if reference is not None else "not_checked",
                        "checked_calls": checks, "outputs": comparison},
        "memory": {"rss_bytes_by_stage": stages, "process_high_water_rss_bytes": int(highwater)},
        "profile": profile,
    }
    return result


def main():
    config_path, result_path = sys.argv[1:]
    config = json.loads(Path(config_path).read_text())
    try:
        result = {"ok": True, "result": measure(config)}
    except Exception as exc:
        # Runtime exception messages can expose absolute paths and model contents.
        from .validation import DiagnosticError
        if isinstance(exc, DiagnosticError):
            error = {"code": exc.code, "message": str(exc)}
            if exc.details is not None:
                error["details"] = exc.details
            if exc.__cause__ is not None:
                error["exception_type"] = type(exc.__cause__).__name__
        else:
            error = {"code": "runtime_error", "message": "ONNX Runtime/data execution failed.",
                     "exception_type": type(exc).__name__}
        result = {"ok": False, "error": error}
    Path(result_path).write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
