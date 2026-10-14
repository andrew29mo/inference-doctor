# Measurement and diagnosis contract

## What is actually measured

The controller snapshots the model, inputs and optional independent reference into a temporary directory. ONNX preflight rejects malformed graphs and external tensor storage, so the model SHA-256 covers all model weights. Every child receives the same immutable-by-convention snapshot. Input files are never modified. User inputs must be trusted; this is not a model execution sandbox.

Each provider gets sequential fresh executable worker processes, three by default. Each worker imports NumPy/ORT, loads NPZ tensors, creates a session, performs the first inference, warms up, then measures repeated synchronous `session.run` calls. A trial reuses one fixed set of input tensors; it does not traverse a dataset. Numeric/boolean tensors only; no implicit casts or shape broadcasting. Symbolic dimension names must agree across inputs.

Each first, warmup and timed output is checked against every named reference output. Shapes and dtypes must match exactly. Floating outputs use `abs(actual - reference) <= atol + rtol * abs(reference)`; defaults are `atol=1e-5`, `rtol=1e-4`. Integer/boolean outputs require exact equality. Nonfinite data/results are rejected. References are supplied by the caller: the tool records their hashes, but cannot establish their independence or validity for an arbitrary user model. The included examples generate references independently with NumPy.

Without a reference the tool checks only supported output types/finiteness, sets correctness to `not_checked`, and returns a warning. This is not an accuracy evaluation. Example weights and inputs are seeded synthetic data; there is no training, test split, or classification-accuracy claim.

## Timing boundaries

- `session_creation_ms`: creation of the ORT session; excludes process launch, imports, NPZ loading, snapshotting and ONNX preflight.
- `first_run_ms`: first synchronous inference in a fresh session. OS caches and accelerator caches are not reset. Do not call this a cold-machine measurement.
- `latency_samples_ms`: one synchronous call each, including Python dispatch (also the diagnostic wrapper), transfers, runtime work, and output allocation. Output comparison occurs after the timer on every call.
- `worker_wall_ms`: parent-observed process launch through exit and result retrieval; includes imports, work, checks, sampling sleeps and report serialization. It is not session startup latency.

The environment sets ORT intra-op threads to the chosen value (default one), inter-op threads to one, sequential execution, full graph optimization, and matching OMP/OpenBLAS/MKL/Accelerate thread environment variables before worker imports. Provider internals may schedule work independently; this is not a promise of one physical core. CoreML uses its default provider options; no device-specific compute-unit request is made.

Raw samples from all trials are retained. Warm median/p95/p99 pool the calls; per-trial p95 values are also reported. Budgets use the worst per-trial p95, not a selected best trial. Startup summaries contain only a few observations and are descriptive, not reliable tail estimates. No outliers are removed. Trials are sequential, without randomized provider order or thermal control. These small graphs emphasize dispatch overhead; results cannot establish general accelerator performance.

Correctness checks outside the timer still influence caches, memory and inter-call spacing. These are validated synchronous-call measurements, not pure kernel throughput. Do not compare these numbers to another tool with different timing or validation boundaries.

## Memory boundaries

Each worker reports resident-set-size snapshots before NumPy/ORT import, after imports, after data load, after session creation, after first inference, after warmup, and after work. It also reports OS process lifetime high-water RSS: `ru_maxrss` bytes on macOS and KiB converted to bytes on Linux. A parent process samples worker RSS every requested 2 ms (actual scheduling may be slower).

The reported peak is the maximum of the worker's OS high-water value and parent sampled peak, across all timing trials. It covers the interpreter, loaded libraries, input/reference tensors, validation arrays, allocator retention and runtime. It excludes controller snapshot/preflight memory and excludes the separate profiling worker. It is **not model-only RAM**, a sum of allocations, total system RAM, GPU/ANE memory, or memory of external CoreML helper processes. Shared pages can be included in process RSS. Stage deltas do not attribute ownership. Short spikes after the final worker high-water checkpoint can evade parent sampling.

RSS budgets are budgets for this validated worker process, not deployment arena limits. File size is reported separately and must not be interpreted as RAM usage.

## Provider evidence

1. A requested provider must appear in ORT's available list.
2. It must register as the primary provider after session construction. A silently replaced provider is a failure.
3. ORT's Python run-time fallback retry is disabled. Graph partitioning may still assign unsupported work to CPU.
4. A **separate profiling session**, one inference using the same snapshot, records optimized/fused graph kernel events and their provider attributes. Profiling overhead is excluded from reported timing/RSS summaries.
5. Observed CPU events under a non-CPU primary, missing attribution, or no events from the requested primary produce warnings. Events are not original graph operation counts or percentages. Session registration alone is not execution proof.

Profile events describe only that separate session. They do not prove assignment in every timing trial. CoreML attribution does not prove execution on Neural Engine, GPU, or any particular physical compute unit. The MVP does not collect CoreML compute-plan or device tracing evidence.

Provider initialization and inference failures are structured errors; raw runtime logs are discarded because they can contain private paths or model details. A generic phase message identifies where to investigate, not a proven root cause. The timeout applies to each worker, not controller preflight/copying.

## Report and exit semantics

- `passed`: all requested timing trials passed supplied references and configured budgets, and separate profiling had no warning conditions.
- `warning`: no errors, but evidence is missing or execution needs attention. Exit code 0; callers requiring a warning-free result must inspect `status`.
- `failed`: any provider, reference, input, runtime, or configured budget error. Exit code 2, with JSON/HTML still produced for diagnosed failures.
- Invalid CLI arguments use argparse exit code 2 and do not create a report.

No default performance or memory threshold is invented. Passing does not certify production suitability. Source hashes identify the package's `.py` files; fixture artifact hashes identify exact binary files. `tools/check_project.py` regenerates fixtures, verifies source provenance, recomputes recorded summaries, checks validation counts and scans tracked files. This checks evidence consistency; it cannot prove past hardware events from JSON alone.

## Official references

- [ORT Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [ORT profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)
- [CoreML execution provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [ORT session configuration keys](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)
