# Recorded deployment measurements

These are descriptive local observations, not a general benchmark ranking or a speedup claim.

- Platform: Darwin 27.0.1, arm64; reported CPU Apple M5 Pro; Python 3.12.14, ONNX Runtime 1.31.0, NumPy 1.26.4.
- Each provider/case: 4 fresh timing processes; 50 warmups and 500 timed calls per process; one ORT intra-op thread, sequential execution.
- Provider order alternates each round (CPU/CoreML, then CoreML/CPU); profiling starts only after all timing trials within that invocation. Even trial count balances first/second position.
- Every first, warmup and timed output checked against an independent NumPy reference (2,204 calls per provider/case), plus one separately checked profiling inference.
- Same MLP weights across batches; CPU/CoreML batch-16 use identical model/input/reference hashes. Models are synthetic, not trained; no task accuracy is measured.
- Latencies below are per call/batch, not per example. Warm quantiles pool 2,000 calls; trial-level distributions and every raw sample are retained.
- Process RSS includes Python/runtime/data/reference/validation and excludes controller, profiling workers, CoreML helper processes and device memory.
- Measurements ran with system access needed by CoreML. A restricted sandbox initially prevented compilation; that failed diagnostic was not a performance measurement.

| Case | Requested primary | Session median ms | First call median ms | Warm median µs | Warm p95 µs | Peak worker RSS MiB |
|---|---|---:|---:|---:|---:|---:|
| linear-b1 ([JSON](macos-arm64/linear-b1/report.json)) | CPUExecutionProvider | 1.600 | 0.124 | 2.833 | 3.167 | 61.20 |
| mlp-b1 ([JSON](macos-arm64/mlp-b1/report.json)) | CPUExecutionProvider | 1.764 | 0.117 | 3.583 | 4.125 | 61.16 |
| mlp-b16 ([JSON](macos-arm64/mlp-b16/report.json)) | CPUExecutionProvider | 1.538 | 0.111 | 4.375 | 5.000 | 61.09 |
| mlp-b16 ([JSON](macos-arm64/mlp-b16/report.json)) | CoreMLExecutionProvider | 36.740 | 4.812 | 42.125 | 48.711 | 69.30 |
| mlp-b64 ([JSON](macos-arm64/mlp-b64/report.json)) | CPUExecutionProvider | 1.697 | 0.115 | 5.750 | 6.167 | 61.17 |
| residual-b16 ([JSON](macos-arm64/residual-b16/report.json)) | CPUExecutionProvider | 1.762 | 0.126 | 6.208 | 6.667 | 61.27 |

## Interpretation

These are validated synchronous Python calls on tiny synthetic graphs, including dispatch, transfers and output allocation. Comparisons share exact model/input/reference artifacts and use balanced trial order. They do not establish general CPU/accelerator superiority or isolate dispatch, compute and transfer costs.

The separate CoreML profile attributed a fused subgraph kernel event to CoreML. This is provider evidence for that profile session, not proof of Neural Engine/GPU execution or assignment in every timing trial. The CPU profile showed CPU events. No unsupported physical-device claim is made.

Session creation and first-use results include visible trial variability; caches and thermal state were not controlled. A few startup observations do not establish startup tail latency. Each HTML report shows the full scope alongside measurements.

## Reports

- [linear-b1 HTML](macos-arm64/linear-b1/report.html) · [JSON](macos-arm64/linear-b1/report.json) · [original reference](macos-arm64/linear-b1/reference.json) · status `passed`
- [mlp-b1 HTML](macos-arm64/mlp-b1/report.html) · [JSON](macos-arm64/mlp-b1/report.json) · [original reference](macos-arm64/mlp-b1/reference.json) · status `passed`
- [mlp-b16 HTML](macos-arm64/mlp-b16/report.html) · [JSON](macos-arm64/mlp-b16/report.json) · [original reference](macos-arm64/mlp-b16/reference.json) · status `passed`
- [mlp-b64 HTML](macos-arm64/mlp-b64/report.html) · [JSON](macos-arm64/mlp-b64/report.json) · [original reference](macos-arm64/mlp-b64/reference.json) · status `passed`
- [residual-b16 HTML](macos-arm64/residual-b16/report.html) · [JSON](macos-arm64/residual-b16/report.json) · [original reference](macos-arm64/residual-b16/reference.json) · status `passed`

Download/open HTML locally; GitHub does not render standalone HTML reports as a hosted application.

Original synthetic reference tensors are retained losslessly as JSON and must reconstruct the measured NPZ hash exactly. Cross-platform NumPy recomputation is checked numerically at fixed independently enforced suite tolerances (rtol=1e-4, atol=1e-5), because BLAS rounding can differ. Reports cannot relax these gates. Model/input hashes still require exact equality.

Regenerate with `python tools/measure_suite.py --output results/macos-arm64 --include-coreml` on an appropriately configured Mac, then `python tools/render_results.py`. New runs will differ. Review changes and verify with `python tools/check_project.py`; do not expect exact timing reproduction.

[Full methodology](../docs/METHODOLOGY.md) · [Generator](../tools/make_examples.py) · [Suite runner](../tools/measure_suite.py)
