# Recorded deployment measurements

These are descriptive local observations, not a general benchmark ranking or a speedup claim.

- Platform: macOS arm64; Python 3.12.14, ONNX Runtime 1.31.0, NumPy 1.26.4.
- Each provider/case: 3 fresh timing processes; 50 warmups and 500 timed calls per process; one ORT intra-op thread, sequential execution.
- Every first, warmup and timed output checked against an independent NumPy reference (1,653 calls per provider/case).
- Same MLP weights across batches; CPU/CoreML batch-16 use identical model/input/reference hashes. Models are synthetic, not trained; no task accuracy is measured.
- Latencies below are per call/batch, not per example. Warm quantiles pool 1,500 calls; trial-level distributions and every raw sample are retained.
- Process RSS includes Python/runtime/data/reference/validation and excludes controller, profiling workers, CoreML helper processes and device memory.
- Measurements ran with system access needed by CoreML. A restricted sandbox initially prevented compilation; that failed diagnostic was not a performance measurement.

| Case | Provider | Session median ms | First call median ms | Warm median µs | Warm p95 µs | Peak worker RSS MiB |
|---|---|---:|---:|---:|---:|---:|
| linear-b1 ([JSON](macos-arm64/linear-b1/report.json)) | CPUExecutionProvider | 1.721 | 0.144 | 2.792 | 3.416 | 61.31 |
| mlp-b1 ([JSON](macos-arm64/mlp-b1/report.json)) | CPUExecutionProvider | 1.709 | 0.114 | 3.916 | 4.458 | 61.09 |
| mlp-b16 ([JSON](macos-arm64/mlp-b16/report.json)) | CPUExecutionProvider | 1.885 | 0.130 | 4.625 | 5.083 | 61.41 |
| mlp-b16 ([JSON](macos-arm64/mlp-b16/report.json)) | CoreMLExecutionProvider | 44.053 | 5.220 | 42.833 | 54.796 | 69.58 |
| mlp-b64 ([JSON](macos-arm64/mlp-b64/report.json)) | CPUExecutionProvider | 1.730 | 0.117 | 5.958 | 6.292 | 61.22 |
| residual-b16 ([JSON](macos-arm64/residual-b16/report.json)) | CPUExecutionProvider | 1.798 | 0.142 | 6.333 | 6.833 | 61.27 |

## Interpretation

All recorded cases passed the supplied numerical references. On the tiny batch-16 MLP, the recorded CoreML calls were slower than CPU, with greater worker-process RSS. This does not establish total accelerator memory use or general CPU superiority; small workloads can be dominated by dispatch/transfer costs, and these trials do not isolate the cause.

The separate CoreML profile attributed a fused subgraph kernel event to CoreML. This is provider evidence for that profile session, not proof of Neural Engine/GPU execution or assignment in every timing trial. The CPU profile showed CPU events. No unsupported physical-device claim is made.

Session creation and first-use results include visible trial variability; caches and thermal state were not controlled. Three startup observations do not establish startup tail latency. Each HTML report shows the full scope alongside measurements.

## Reports

- [linear-b1 HTML](macos-arm64/linear-b1/report.html) · [JSON](macos-arm64/linear-b1/report.json) · status `passed`
- [mlp-b1 HTML](macos-arm64/mlp-b1/report.html) · [JSON](macos-arm64/mlp-b1/report.json) · status `passed`
- [mlp-b16 HTML](macos-arm64/mlp-b16/report.html) · [JSON](macos-arm64/mlp-b16/report.json) · status `passed`
- [mlp-b64 HTML](macos-arm64/mlp-b64/report.html) · [JSON](macos-arm64/mlp-b64/report.json) · status `passed`
- [residual-b16 HTML](macos-arm64/residual-b16/report.html) · [JSON](macos-arm64/residual-b16/report.json) · status `passed`

Download/open HTML locally; GitHub does not render standalone HTML reports as a hosted application.

Regenerate with `python tools/measure_suite.py --output results/macos-arm64 --include-coreml` on an appropriately configured Mac, then `python tools/render_results.py`. New runs will differ. Review changes and verify with `python tools/check_project.py`; do not expect exact timing reproduction.

[Full methodology](../docs/METHODOLOGY.md) · [Generator](../tools/make_examples.py) · [Suite runner](../tools/measure_suite.py)
