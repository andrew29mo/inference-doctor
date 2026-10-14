import argparse
import math
from pathlib import Path

from .report import write_report
from .runner import diagnose


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def finite_nonnegative(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measured ONNX deployment diagnostics; local trusted models only.")
    parser.add_argument("model")
    parser.add_argument("--inputs", required=True, help="NPZ keyed by exact model input names")
    parser.add_argument("--reference", help="Independent NPZ keyed by exact model output names")
    parser.add_argument("--provider", action="append", help="Exact ORT provider name; repeat to compare (default CPUExecutionProvider)")
    parser.add_argument("--output", default="results/local/report")
    parser.add_argument("--iterations", type=positive_int, default=200)
    parser.add_argument("--warmup", type=nonnegative_int, default=20)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--timeout", type=positive_int, default=120, help="Seconds per child process")
    parser.add_argument("--rtol", type=finite_nonnegative, default=1e-4)
    parser.add_argument("--atol", type=finite_nonnegative, default=1e-5)
    parser.add_argument("--max-peak-mib", type=finite_nonnegative)
    parser.add_argument("--max-p95-ms", type=finite_nonnegative)
    args = parser.parse_args(argv)
    config = {k: getattr(args, k) for k in ("iterations", "warmup", "threads", "rtol", "atol")}
    config.update({k: str(Path(getattr(args, k)).resolve()) if getattr(args, k) else None
                   for k in ("model", "inputs", "reference")})
    # Output must not overwrite source artifacts, even on an error-report run.
    targets = {Path(args.output).resolve() / name for name in ("report.json", "report.html")}
    if any(Path(v) in targets for k, v in config.items() if k in ("model", "inputs", "reference") and v):
        parser.error("report output would overwrite an input artifact")
    providers = list(dict.fromkeys(args.provider or ["CPUExecutionProvider"]))
    report = diagnose(config, providers, args.repeats, args.timeout, args.max_peak_mib, args.max_p95_ms)
    write_report(report, args.output)
    print(f'{report["status"].upper()}: {len(report["providers"])} provider(s); report.json and report.html written.')
    for item in report["diagnostics"]:
        print(f'{item["severity"]}: {item["code"]}: {item["message"]}')
    return 2 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
