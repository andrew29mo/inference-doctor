"""Offline HTML; all data is escaped, with no external assets or scripts."""

import html
import json
from pathlib import Path


LIMITATIONS = [
    "Model and data are snapshotted once by the controller; all child processes use the same snapshot. Copying and ONNX preflight are outside worker time/RSS measurements.",
    "Latency is synchronous session.run wall time including Python dispatch, tensor transfers and output allocation. File loading and validation are outside each call timer.",
    "All measured outputs are checked outside each call timer when a reference is supplied. These checks influence cache state and memory; this is a validated-call benchmark, not a raw kernel benchmark.",
    "Session creation excludes Python/ORT imports and NPZ loading. First run is first use of a fresh session, not a cold OS/filesystem/device cache.",
    "RSS is whole worker-process resident memory: interpreter, libraries, inputs, references, validation and runtime. It is not model-only allocation or total system/device memory. CoreML/helper-process memory is excluded.",
    "OS high-water RSS covers process lifetime through the final checkpoint; parent 2 ms sampling also includes startup/exit but can miss short spikes. Stage RSS differences are observations, not causal allocations.",
    "Profiling runs in a separate fresh session and is excluded from timing/RSS summaries. EP assignment may differ between sessions and does not identify physical CPU/GPU/ANE use.",
    "Timing trials alternate provider order each round; all profiling runs follow all timing trials. Fixed inputs, OS scheduling, thermal state and system caches remain uncontrolled. No statistical speedup or cross-machine performance claim is made.",
    "Numeric finite dense tensors and self-contained ONNX models on macOS/Linux only. No dataset accuracy claim, service-load test, device-memory accounting, or sandbox for untrusted models.",
]


def write_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report["measurement_scope"] = LIMITATIONS
    raw = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (directory / "report.json").write_text(raw)
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for provider in report["providers"]:
        s = provider.get("summary")
        if s:
            rows.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in (
                provider["requested"], f'{s["session_creation_ms"]["median"]:.3f}',
                f'{s["first_run_ms"]["median"]:.3f}',
                f'{s["pooled_warm_latency_ms"]["median"]:.4f}',
                f'{s["pooled_warm_latency_ms"]["p95"]:.4f}',
                f'{s["peak_process_rss_bytes"] / 1024**2:.2f}',
                ", ".join(r["correctness"]["status"] for r in provider["runs"])
            )) + "</tr>")
    issues = "".join(f'<li><b>{esc(d["severity"])} · {esc(d["code"])}</b> '
                     f'{esc(d.get("provider") or "")} — {esc(d["message"])}</li>'
                     for d in report["diagnostics"]) or "<li>No diagnostic issues observed within the stated scope.</li>"
    scope = "".join(f"<li>{esc(item)}</li>" for item in LIMITATIONS)
    failure_details = "".join(
        f'<h3>{esc(p["requested"])}</h3><pre>{esc(json.dumps(p["error"], indent=2))}</pre>'
        for p in report["providers"] if "error" in p)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inference Doctor report</title><style>
body{{font:16px/1.6 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#172b3a;background:#f5f8fa}}
h1,h2{{line-height:1.25}}section{{background:white;padding:24px;margin:20px 0;border:1px solid #dce4e8;border-radius:12px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #dde5eb}}
.scroll{{overflow:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}li{{margin:10px 0}}.status{{font-size:22px;font-weight:650}}
</style><h1>Inference Doctor</h1><p>Measured deployment evidence · v{esc(report["tool_version"])}</p>
<section><div class="status">{esc(report["status"]).upper()}</div><p>{esc(report["generated_utc"])}</p>
<p>Model: {esc(report["artifacts"].get("model", {}).get("name", "unreadable"))} · Trials per provider: {esc(report["repeats"])}</p>
<p>Reported CPU: {esc(report.get("host", {}).get("cpu_model") or "unknown")} · Architecture: {esc(report.get("host", {}).get("machine", "unknown"))}</p>
<p>Passed means supplied output references and configured budgets passed in these trials; it is not deployment certification.</p></section>
<section><h2>Measurements</h2><p>Requested providers are not physical-device identities. Peak RSS covers the validated worker only; CoreML helper processes and device memory are excluded. These numbers cannot compare total deployment memory.</p><div class="scroll"><table><thead><tr><th>Requested primary</th><th>Session median ms</th><th>First run median ms</th><th>Warm median ms</th><th>Warm p95 ms</th><th>Peak worker RSS MiB</th><th>Reference</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div><p>Warm percentiles pool raw calls. Per-trial p95, every sample, memory stages and EP events are in the evidence below.</p></section>
<section><h2>Diagnosis</h2><ul>{issues}</ul>{failure_details}</section><section><h2>Interpretation and limits</h2><ul>{scope}</ul></section>
<section><details><summary>Complete evidence and artifact hashes</summary><pre>{esc(raw)}</pre></details></section></html>'''
    (directory / "report.html").write_text(page)
