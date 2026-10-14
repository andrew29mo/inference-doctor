"""Render the results overview from recorded evidence, without editing measurements."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def render():
    index = json.loads((ROOT / "results/macos-arm64/index.json").read_text())
    from suite_policy import REPEATS, WARMUP, ITERATIONS
    reports = {c["name"]: json.loads((ROOT / "results/macos-arm64" / c["report"]).read_text()) for c in index["cases"]}
    first = reports[index["cases"][0]["name"]]
    env = first["providers"][0]["runs"][0]["environment"]
    host = first["host"]
    lines = ["# Recorded deployment measurements", "",
             "These are descriptive local observations, not a general benchmark ranking or a speedup claim.", "",
             f'- Platform: {host["os"]} {host["os_version"]}, {host["machine"]}; reported CPU {host["cpu_model"] or "unknown"}; Python {env["python"]}, ONNX Runtime {env["onnxruntime"]}, NumPy {env["numpy"]}.',
             f"- Each provider/case: {REPEATS} fresh timing processes; {WARMUP} warmups and {ITERATIONS} timed calls per process; one ORT intra-op thread, sequential execution.",
             "- Provider order alternates each round (CPU/CoreML, then CoreML/CPU); profiling starts only after all timing trials within that invocation. Even trial count balances first/second position.",
             f"- Every first, warmup and timed output checked against an independent NumPy reference ({REPEATS * (1 + WARMUP + ITERATIONS):,} calls per provider/case), plus one separately checked profiling inference.",
             "- Same MLP weights across batches; CPU/CoreML batch-16 use identical model/input/reference hashes. Models are synthetic, not trained; no task accuracy is measured.",
             f"- Latencies below are per call/batch, not per example. Warm quantiles pool {REPEATS * ITERATIONS:,} calls; trial-level distributions and every raw sample are retained.",
             "- Process RSS includes Python/runtime/data/reference/validation and excludes controller, profiling workers, CoreML helper processes and device memory.",
             "- Measurements ran with system access needed by CoreML. A restricted sandbox initially prevented compilation; that failed diagnostic was not a performance measurement.", "",
             "| Case | Requested primary | Session median ms | First call median ms | Warm median µs | Warm p95 µs | Peak worker RSS MiB |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for case in index["cases"]:
        report = json.loads((ROOT / "results/macos-arm64" / case["report"]).read_text())
        for provider in report["providers"]:
            s = provider.get("summary")
            link = f'[JSON](macos-arm64/{case["report"]})'
            if s:
                lines.append(f'| {case["name"]} ({link}) | {provider["requested"]} | '
                             f'{s["session_creation_ms"]["median"]:.3f} | {s["first_run_ms"]["median"]:.3f} | '
                             f'{s["pooled_warm_latency_ms"]["median"]*1000:.3f} | '
                             f'{s["pooled_warm_latency_ms"]["p95"]*1000:.3f} | '
                             f'{s["peak_process_rss_bytes"]/1024**2:.2f} |')
            else:
                lines.append(f'| {case["name"]} ({link}) | {provider["requested"]} | Failed | — | — | — | — |')
    lines += ["", "## Interpretation", "",
              "These are validated synchronous Python calls on tiny synthetic graphs, including dispatch, transfers and output allocation. Comparisons share exact model/input/reference artifacts and use balanced trial order. They do not establish general CPU/accelerator superiority or isolate dispatch, compute and transfer costs.", "",
              "The separate CoreML profile attributed a fused subgraph kernel event to CoreML. This is provider evidence for that profile session, not proof of Neural Engine/GPU execution or assignment in every timing trial. The CPU profile showed CPU events. No unsupported physical-device claim is made.", "",
              "Session creation and first-use results include visible trial variability; caches and thermal state were not controlled. A few startup observations do not establish startup tail latency. Each HTML report shows the full scope alongside measurements.", "",
              "## Reports", ""]
    for case in index["cases"]:
        lines.append(f'- [{case["name"]} HTML](macos-arm64/{case["name"]}/report.html) · '
                     f'[JSON](macos-arm64/{case["name"]}/report.json) · '
                     f'[original reference](macos-arm64/{case["name"]}/reference.json) · status `{case["status"]}`')
    lines += ["", "Download/open HTML locally; GitHub does not render standalone HTML reports as a hosted application.", "",
              "Original synthetic reference tensors are retained losslessly as JSON and must reconstruct the measured NPZ hash exactly. Cross-platform NumPy recomputation is checked numerically at fixed independently enforced suite tolerances (rtol=1e-4, atol=1e-5), because BLAS rounding can differ. Reports cannot relax these gates. Model/input hashes still require exact equality.", "",
              "Regenerate with `python tools/measure_suite.py --output results/macos-arm64 --include-coreml` on an appropriately configured Mac, then `python tools/render_results.py`. New runs will differ. Review changes and verify with `python tools/check_project.py`; do not expect exact timing reproduction.", "",
              "[Full methodology](../docs/METHODOLOGY.md) · [Generator](../tools/make_examples.py) · [Suite runner](../tools/measure_suite.py)", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "results/README.md"
    text = render()
    if args.check:
        if path.read_text() != text:
            raise SystemExit("Results overview differs from recorded evidence")
    else:
        path.write_text(text)
