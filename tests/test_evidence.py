import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

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
            canonical_path = path.parent / "reference.json"
            checker.verify_report(report, fixture, canonical_path)
            mutations = [
                lambda r: r.update(source_sha256="wrong"),
                lambda r: r["artifacts"]["inputs"].update(sha256="wrong"),
                lambda r: r["artifacts"]["reference"].update(sha256="wrong"),
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
                    checker.verify_report(changed, fixture, canonical_path)

    def test_cross_platform_roundoff_keeps_exact_recorded_reference_integrity(self):
        path = ROOT / "results/macos-arm64/mlp-b16/report.json"
        report = json.loads(path.read_text())
        with tempfile.TemporaryDirectory() as directory:
            fixture = checker.generator.generate(directory, "mlp", 16)
            with np.load(fixture / "reference.npz") as archive:
                arrays = {k: archive[k] for k in archive.files}
            arrays["Y"][0, 0] = np.nextafter(arrays["Y"][0, 0], np.float32(np.inf))
            np.savez(fixture / "reference.npz", **arrays)
            checker.verify_report(report, fixture, path.parent / "reference.json")
            arrays["Y"][0, 0] += 1
            np.savez(fixture / "reference.npz", **arrays)
            with self.assertRaises(ValueError):
                checker.verify_report(report, fixture, path.parent / "reference.json")

    def test_retained_reference_tampering_is_rejected(self):
        path = ROOT / "results/macos-arm64/mlp-b16/report.json"
        report = json.loads(path.read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = checker.generator.generate(root / "fixture", "mlp", 16)
            reference = json.loads((path.parent / "reference.json").read_text())
            reference["tensors"][0]["values"][-1][-1] += 1
            tampered = root / "reference.json"
            tampered.write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError, "integrity"):
                checker.verify_report(report, fixture, tampered)


if __name__ == "__main__":
    unittest.main()
