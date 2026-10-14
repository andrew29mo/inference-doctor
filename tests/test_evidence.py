import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_project", ROOT / "tools/check_project.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class EvidenceTests(unittest.TestCase):
    def test_percentile_verifier_rejects_changed_summary(self):
        with self.assertRaises(ValueError):
            checker.verify_summary({"count": 4, "min": 1, "max": 4, "median": 2.5, "p95": 3, "p99": 3.97}, [1, 2, 3, 4])

    def test_current_report_and_deliberate_evidence_corruption(self):
        path = ROOT / "results/macos-arm64/mlp-b16/report.json"
        if not path.exists():
            self.skipTest("Recorded suite not yet generated")
        report = json.loads(path.read_text())
        with tempfile.TemporaryDirectory() as directory:
            fixture = checker.generator.generate(directory, "mlp", 16)
            checker.verify_report(report, fixture)
            mutations = [
                lambda r: r.update(source_sha256="wrong"),
                lambda r: r["artifacts"]["inputs"].update(sha256="wrong"),
                lambda r: r["providers"][0]["runs"][0]["correctness"].update(checked_calls=1),
                lambda r: r["providers"][0]["runs"][0]["latency_samples_ms"].pop(),
                lambda r: r["providers"][0]["summary"]["pooled_warm_latency_ms"].update(p95=999),
                lambda r: r["providers"][0]["summary"].update(peak_process_rss_bytes=1),
                lambda r: r.update(status="failed" if r["status"] != "failed" else "passed"),
            ]
            for mutation in mutations:
                changed = copy.deepcopy(report)
                mutation(changed)
                with self.assertRaises(ValueError):
                    checker.verify_report(changed, fixture)


if __name__ == "__main__":
    unittest.main()
