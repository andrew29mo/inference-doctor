import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort

from inference_doctor.cli import main
from inference_doctor.model import inspect_model
from inference_doctor.report import write_report
from inference_doctor.runner import diagnose, percentile, run_worker
from inference_doctor.validation import DiagnosticError, compare_outputs, load_arrays, validate_inputs
from inference_doctor.worker import measure

spec = importlib.util.spec_from_file_location("make_examples", Path(__file__).resolve().parents[1] / "tools/make_examples.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class DoctorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = generator.generate(self.root / "fixture", "mlp", 17)
        self.config = {"model": str(self.fixture / "model.onnx"), "inputs": str(self.fixture / "inputs.npz"),
                       "reference": str(self.fixture / "reference.npz"), "threads": 1,
                       "warmup": 2, "iterations": 5, "rtol": 1e-4, "atol": 1e-5}

    def cli(self, *extra):
        args = [self.config["model"], "--inputs", self.config["inputs"], "--reference", self.config["reference"],
                "--output", str(self.root / "report"), "--repeats", "1", "--iterations", "4", "--warmup", "1", *extra]
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(args)
        return code, json.loads((self.root / "report/report.json").read_text())

    def test_three_real_onnx_graphs_and_dynamic_batches(self):
        for kind, batch in [("linear", 1), ("mlp", 17), ("residual", 64)]:
            with self.subTest(kind=kind, batch=batch):
                fixture = generator.generate(self.root / kind, kind, batch)
                config = dict(self.config, **{k: str(fixture / f) for k, f in (
                    ("model", "model.onnx"), ("inputs", "inputs.npz"), ("reference", "reference.npz"))})
                report = diagnose(config, ["CPUExecutionProvider"], 2, 30)
                self.assertEqual(report["status"], "passed", report["diagnostics"])
                entry = report["providers"][0]
                self.assertGreater(entry["profile"]["provider_kernel_events"]["CPUExecutionProvider"], 0)
                self.assertEqual(entry["summary"]["pooled_warm_latency_ms"]["count"], 10)
                for run in entry["runs"]:
                    self.assertEqual(run["inputs"][0]["shape"][0], batch)
                    self.assertEqual(run["correctness"]["checked_calls"], 8)
                    self.assertGreater(run["memory"]["process_high_water_rss_bytes"], 1024**2)
                    self.assertGreater(run["memory"]["parent_rss_samples"], 0)
                    self.assertTrue(all(t > 0 for t in run["latency_samples_ms"]))

    def test_model_weights_are_identical_across_batch_sizes(self):
        models = []
        inputs = []
        for batch in (1, 16, 64):
            fixture = generator.generate(self.root / f"batch-{batch}", "mlp", batch)
            models.append((fixture / "model.onnx").read_bytes())
            with np.load(fixture / "inputs.npz") as archive:
                inputs.append(archive["X"])
        self.assertEqual(models[0], models[1])
        self.assertEqual(models[1], models[2])
        np.testing.assert_array_equal(inputs[0], inputs[1][:1])
        np.testing.assert_array_equal(inputs[1], inputs[2][:16])

    def test_installed_entrypoint_and_html_json(self):
        result = subprocess.run([sys.executable, "-m", "inference_doctor", self.config["model"],
                                 "--inputs", self.config["inputs"], "--reference", self.config["reference"],
                                 "--output", str(self.root / "installed"), "--repeats", "1",
                                 "--iterations", "2", "--warmup", "0"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.root / "installed/report.json").read_text())
        self.assertEqual(report["status"], "passed")
        html = (self.root / "installed/report.html").read_text()
        self.assertIn("RSS is whole worker-process", html)
        self.assertIn("CPUExecutionProvider", html)
        self.assertNotIn(str(self.root), json.dumps(report))

    def test_wrong_last_row_and_secondary_output_are_detected(self):
        for name in ("Y", "H"):
            with self.subTest(output=name):
                with np.load(self.config["reference"]) as archive:
                    arrays = {k: archive[k] for k in archive.files}
                arrays[name][-1, -1] += 1
                wrong = self.root / f"wrong-{name}.npz"
                np.savez(wrong, **arrays)
                code, report = self.cli("--reference", str(wrong))
                self.assertEqual(code, 2)
                self.assertIn("output_mismatch", [d["code"] for d in report["diagnostics"]])
                self.assertNotIn("summary", report["providers"][0])
                outputs = report["providers"][0]["error"]["details"]["outputs"]
                self.assertFalse(next(o for o in outputs if o["name"] == name)["passed"])

    def test_timed_only_fault_is_rejected_by_production_worker(self):
        factory = ort.InferenceSession
        config = dict(self.config, provider="CPUExecutionProvider", mode="measure", workdir=str(self.root))

        class FaultSession:
            def __init__(self, *args, **kwargs):
                self.inner = factory(*args, **kwargs)
                self.calls = 0

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def run(self, *args):
                self.calls += 1
                outputs = self.inner.run(*args)
                if self.calls == 4:  # First timed call after first inference + 2 warmups.
                    outputs[-1][-1, -1] += 1
                return outputs

        with patch.object(ort, "InferenceSession", FaultSession):
            with self.assertRaisesRegex(DiagnosticError, "differs"):
                measure(config)

    def test_input_mistakes_rejected_through_cli(self):
        with np.load(self.config["inputs"]) as archive:
            x = archive["X"]
        for code, arrays in [
            ("input_dtype", {"X": x.astype(np.float64)}),
            ("input_rank", {"X": x.reshape(-1)}),
            ("input_shape", {"X": x[:, :-1]}),
            ("input_names", {"wrong": x}),
            ("input_names", {"X": x, "extra": x}),
            ("nonfinite_data", {"X": x * np.nan}),
        ]:
            with self.subTest(code=code):
                path = self.root / "bad-input.npz"
                np.savez(path, **arrays)
                exit_code, report = self.cli("--inputs", str(path))
                self.assertEqual(exit_code, 2)
                self.assertIn(code, [d["code"] for d in report["diagnostics"]])

    def test_missing_provider_is_not_silent_cpu_success(self):
        code, report = self.cli("--provider", "MissingExecutionProvider")
        self.assertEqual(code, 2)
        self.assertEqual(report["diagnostics"][0]["code"], "provider_unavailable")
        self.assertIn("CPUExecutionProvider", report["providers"][0]["error"]["details"]["available_providers"])

    def test_profile_failure_is_not_false_assignment_evidence(self):
        from inference_doctor import runner
        actual = runner.run_worker

        def with_profile_failure(config, timeout):
            if config["mode"] == "profile":
                return {"ok": False, "error": {"code": "runtime_error", "message": "test"}}
            return actual(config, timeout)

        with patch.object(runner, "run_worker", with_profile_failure):
            report = diagnose(self.config, ["CPUExecutionProvider"], 1, 30)
        self.assertEqual(report["status"], "warning")
        self.assertIsNone(report["providers"][0]["profile"])
        self.assertIn("profiling_failed", [d["code"] for d in report["diagnostics"]])

    def test_artifact_snapshot_is_shared_and_original_mutation_is_irrelevant(self):
        from inference_doctor import runner
        actual = runner.run_worker
        paths = []

        def mutate_original(config, timeout):
            paths.append(config["inputs"])
            self.assertNotEqual(config["inputs"], self.config["inputs"])
            Path(self.config["inputs"]).write_bytes(b"changed after snapshot")
            return actual(config, timeout)

        with patch.object(runner, "run_worker", mutate_original):
            report = diagnose(self.config, ["CPUExecutionProvider"], 2, 30)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(set(paths)), 1)
        self.assertEqual(len(paths), 3)

    def test_reference_optional_does_not_claim_correctness(self):
        config = dict(self.config, reference=None)
        report = diagnose(config, ["CPUExecutionProvider"], 1, 30)
        self.assertEqual(report["status"], "warning")
        self.assertEqual(report["providers"][0]["runs"][0]["correctness"]["status"], "not_checked")

    def test_real_memory_and_latency_budgets_fail(self):
        for option, expected in [("--max-peak-mib", "memory_budget"), ("--max-p95-ms", "latency_budget")]:
            with self.subTest(option=option):
                code, report = self.cli(option, "0")
                self.assertEqual(code, 2)
                self.assertIn(expected, [d["code"] for d in report["diagnostics"]])

    def test_worker_timeout(self):
        config = dict(self.config, provider="CPUExecutionProvider", mode="measure", iterations=10000000)
        result = run_worker(config, 1)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "worker_timeout")

    def test_external_weights_rejected(self):
        model = onnx.load(self.config["model"])
        path = self.root / "external.onnx"
        onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                        location="weights.bin", size_threshold=0)
        with self.assertRaisesRegex(DiagnosticError, "External"):
            inspect_model(path)

    def test_missing_and_corrupt_model_get_failure_reports(self):
        for path in (self.root / "missing.onnx", self.root / "invalid.onnx"):
            if path.name == "invalid.onnx":
                path.write_bytes(b"not a model")
            with contextlib.redirect_stdout(io.StringIO()):
                code = main([str(path), "--inputs", self.config["inputs"], "--output", str(self.root / "error")])
            self.assertEqual(code, 2)
            report = json.loads((self.root / "error/report.json").read_text())
            self.assertEqual(report["status"], "failed")

    def test_html_escapes_model_names_and_messages(self):
        report = {"tool_version": "test", "status": "failed", "generated_utc": "now", "repeats": 1,
                  "artifacts": {"model": {"name": '<img src=x onerror="alert(1)">'}}, "providers": [],
                  "diagnostics": [{"severity": "error", "code": "test", "message": "<script>alert(1)</script>"}]}
        write_report(report, self.root / "html")
        page = (self.root / "html/report.html").read_text()
        self.assertNotIn("<script>", page)
        self.assertNotIn("<img", page)
        self.assertIn("&lt;script&gt;", page)

    def test_reference_shape_dtype_names_nonfinite_and_integer_exactness(self):
        actual = np.ones((2, 3), dtype=np.float32)
        for reference in ({"Y": actual[0]}, {"Y": actual.astype(np.float64)}, {"Z": actual}):
            with self.assertRaises(DiagnosticError):
                compare_outputs(["Y"], [actual], reference, 1e-4, 1e-5)
        with self.assertRaises(DiagnosticError):
            compare_outputs(["Y"], [actual * np.nan], {"Y": actual}, 1e-4, 1e-5)
        integer = np.array([2**63 - 1], np.int64)
        with self.assertRaises(DiagnosticError):
            compare_outputs(["Y"], [integer], {"Y": integer - 1}, 1, 1)

    def test_shared_symbol_and_empty_dimensions(self):
        metadata = [SimpleNamespace(name=k, type="tensor(float)", shape=["N", 3]) for k in ("A", "B")]
        with self.assertRaises(DiagnosticError):
            validate_inputs(metadata, {"A": np.ones((2, 3), np.float32), "B": np.ones((3, 3), np.float32)})
        with self.assertRaises(DiagnosticError):
            validate_inputs(metadata, {"A": np.ones((0, 3), np.float32), "B": np.ones((0, 3), np.float32)})

    def test_npz_object_arrays_forbidden(self):
        path = self.root / "objects.npz"
        np.savez(path, X=np.array([{}], dtype=object))
        with self.assertRaises(DiagnosticError):
            load_arrays(path)

    def test_invalid_numeric_options_rejected(self):
        for option, value in [("--rtol", "nan"), ("--atol", "inf"), ("--iterations", "0"), ("--warmup", "-1")]:
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main([self.config["model"], "--inputs", self.config["inputs"], option, value])

    def test_percentile_interpolation(self):
        self.assertEqual(percentile([4, 1, 3, 2], .5), 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], .95), 3.85)


if __name__ == "__main__":
    unittest.main()
