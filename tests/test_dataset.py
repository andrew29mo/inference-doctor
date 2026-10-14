import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import onnxruntime as ort

from inference_doctor.cli import main
from inference_doctor.runner import diagnose, checked_worker, run_worker
from inference_doctor.dataset import ClassificationDataset
from inference_doctor.validation import DiagnosticError, load_arrays, compare_outputs
from inference_doctor.worker import measure

ROOT = Path(__file__).resolve().parents[1]
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
if HAS_SKLEARN:
    spec = importlib.util.spec_from_file_location("digits_prepare", ROOT / "examples/digits/prepare.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)


@unittest.skipUnless(HAS_SKLEARN, "Install requirements-real-model.txt for real-model tests")
class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = prepare.prepare(self.root)
        self.config = {"model": str(self.root / "model.onnx"), "inputs": str(self.root / "inputs.npz"),
                       "reference": str(self.root / "reference.npz"), "labels": str(self.root / "labels.npz"),
                       "batch_size": 16, "classification_output": "probabilities", "threads": 1,
                       "warmup": 1, "iterations": 1, "rtol": 1e-4, "atol": 1e-5}

    def test_actual_worker_scores_full_dataset_at_all_batch_sizes(self):
        for batch in (1, 16, 64, 128):
            with self.subTest(batch=batch):
                report = diagnose(dict(self.config, batch_size=batch), ["CPUExecutionProvider"], 1, 60)
                self.assertEqual(report["status"], "passed", report["diagnostics"])
                metadata = report["dataset"]
                batches = (450 + batch - 1) // batch
                self.assertEqual(metadata["rows"], 450)
                self.assertEqual(metadata["last_batch_rows"], 450 - (batches - 1) * batch)
                run = report["providers"][0]["runs"][0]
                self.assertEqual(run["correctness"]["checked_calls"], 2 + 2 * batches)
                self.assertEqual(sum(run["latency_sample_rows"]), 450)
                self.assertEqual(len(run["latency_samples_ms"]), batches)
                for evidence in [run["dataset"]["evaluation"], *run["dataset"]["timed_passes"]]:
                    self.assertEqual(evidence["valid_rows"], 450)
                    self.assertEqual(evidence["correct"], self.manifest["correct"])
                    self.assertEqual(evidence["sample_ids_sha256"], self.manifest["sample_ids_int64_sha256"])
                    self.assertEqual(evidence["predictions_sha256"], self.manifest["predictions_int64_sha256"])
                summary = report["providers"][0]["summary"]["dataset"]
                self.assertAlmostEqual(summary["per_trial_inference_us_per_example"][0], sum(run["latency_samples_ms"]) * 1000 / 450)

    def test_cli_dataset_report_is_real_accuracy(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = main([self.config["model"], "--inputs", self.config["inputs"], "--reference", self.config["reference"],
                         "--labels", self.config["labels"], "--batch-size", "64", "--classification-output", "probabilities",
                         "--iterations", "1", "--repeats", "1", "--warmup", "0", "--output", str(self.root / "report")])
        self.assertEqual(code, 0)
        report = json.loads((self.root / "report/report.json").read_text())
        self.assertEqual(report["providers"][0]["summary"]["dataset"]["correct"], 436)
        page = (self.root / "report/report.html").read_text()
        self.assertIn("436/450", page)
        self.assertIn("no padding", page)

    def test_timed_tail_only_corruption_is_detected_by_formal_worker(self):
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
                if self.calls == 60:  # first + 29 evaluation + 1 warmup + 29 timed.
                    outputs[0][-1, -1] += 1
                return outputs

        with patch.object(ort, "InferenceSession", FaultSession), self.assertRaises(DiagnosticError) as context:
            measure(config)
        self.assertEqual(context.exception.code, "output_mismatch")

    def test_last_ground_truth_changes_actual_accuracy(self):
        with np.load(self.config["labels"]) as archive:
            arrays = {k: archive[k] for k in archive.files}
        with np.load(self.config["reference"]) as archive:
            predicted = int(archive["probabilities"][-1].argmax())
        was_correct = arrays["labels"][-1] == predicted
        arrays["labels"][-1] = (predicted + 1) % 10 if was_correct else predicted
        np.savez(self.config["labels"], **arrays)
        report = diagnose(self.config, ["CPUExecutionProvider"], 1, 60)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["providers"][0]["summary"]["dataset"]["correct"], self.manifest["correct"] + (-1 if was_correct else 1))

    def test_controller_rejects_contradictory_actual_worker_evidence(self):
        dataset = ClassificationDataset(load_arrays(self.config["inputs"]), load_arrays(self.config["reference"]),
                                        self.config["labels"], 16, "probabilities")
        config = dict(self.config, provider="CPUExecutionProvider", mode="measure", dataset_metadata=dataset.metadata())
        original = run_worker(config, 60)
        self.assertTrue(original["ok"], original)
        with patch("inference_doctor.runner.run_worker", return_value=original):
            self.assertTrue(checked_worker(config, 60, ["logits", "probabilities"])["ok"])
        mutations = [
            lambda r: r["dataset"]["evaluation"].update(accuracy=1.0),
            lambda r: r["dataset"]["timed_passes"][0].update(accuracy=1.0),
            lambda r: r["dataset"]["timed_passes"][0].update(validated_wall_ms=-1),
            lambda r: r["dataset"]["timed_passes"][0].update(inference_ms=0),
            lambda r: r["dataset"]["evaluation"].update(inference_ms=0),
            lambda r: r["memory"]["rss_bytes_by_stage"].update(after_session=-1),
            lambda r: r["memory"].update(parent_sampled_peak_rss_bytes=float("nan")),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(fault=index):
                changed = copy.deepcopy(original)
                mutation(changed["result"])
                with patch("inference_doctor.runner.run_worker", return_value=changed):
                    outcome = checked_worker(config, 60, ["logits", "probabilities"])
                self.assertFalse(outcome["ok"])
                self.assertEqual(outcome["error"]["code"], "worker_protocol")

    def test_actual_pass_rejects_skipped_repeated_and_reordered_batches(self):
        inputs, reference = load_arrays(self.config["inputs"]), load_arrays(self.config["reference"])
        dataset = ClassificationDataset(inputs, reference, self.config["labels"], 128, "probabilities")
        session = ort.InferenceSession(self.config["model"], providers=["CPUExecutionProvider"])
        names = [o.name for o in session.get_outputs()]
        subsets = list(dataset.slices())
        def validate(outputs, expected):
            compare_outputs(names, outputs, expected, 1e-4, 1e-5)
        # Real ORT calls and per-output comparisons still pass for each subset;
        # coverage must independently reject omission, duplication and reordering.
        for changed in (subsets[:-1], subsets + subsets[-1:], list(reversed(subsets))):
            with patch.object(dataset, "slices", return_value=iter(changed)):
                with self.assertRaises(DiagnosticError) as context:
                    dataset.run_pass(session, names, lambda stage, call: call(), validate, timed=True)
                self.assertEqual(context.exception.code, "dataset_coverage")

    def test_fit_receives_only_the_fixed_training_split(self):
        x, y, train, test, provenance = prepare.dataset()

        class Captured(Exception):
            pass

        with patch.object(prepare, "fit_classifier", side_effect=Captured) as fit:
            with self.assertRaises(Captured):
                prepare.train_state()
        np.testing.assert_array_equal(fit.call_args.args[0], x[train])
        np.testing.assert_array_equal(fit.call_args.args[1], y[train])
        self.assertEqual(len(train), 1347)
        self.assertEqual(len(test), 450)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(len({row.tobytes() for row in x}), len(x))
        self.assertEqual(provenance["train_ids"], train.tolist())

    def test_float32_reference_preserves_original_sklearn_classes(self):
        state = json.loads(prepare.STATE.read_text())
        x, y, _, test, _ = prepare.dataset()
        original = prepare.restore(state).predict_proba(x[test])
        with np.load(self.config["reference"]) as archive:
            converted = archive["probabilities"]
        np.testing.assert_array_equal(original.argmax(axis=1), converted.argmax(axis=1))
        self.assertEqual(int(np.sum(original.argmax(axis=1) == y[test])), self.manifest["correct"])

    def test_fresh_inference_process_does_not_load_training_libraries(self):
        config = dict(self.config, provider="CPUExecutionProvider", mode="measure", workdir=str(self.root))
        script = ('import json,sys; from inference_doctor.worker import measure; '
                  'r=measure(json.loads(sys.argv[1])); '
                  'print(json.dumps({"sklearn": "sklearn" in sys.modules, "scipy": "scipy" in sys.modules, '
                  '"correct": r["dataset"]["evaluation"]["correct"]}))')
        completed = subprocess.run([sys.executable, "-c", script, json.dumps(config)],
                                   capture_output=True, text=True, env=dict(os.environ, PYTHONOPTIMIZE=str(sys.flags.optimize)))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), {"sklearn": False, "scipy": False, "correct": 436})

    def test_duplicate_ids_and_missing_labels_rejected(self):
        with np.load(self.config["labels"]) as archive:
            arrays = {k: archive[k] for k in archive.files}
        arrays["sample_ids"][-1] = arrays["sample_ids"][0]
        np.savez(self.config["labels"], **arrays)
        report = diagnose(self.config, ["CPUExecutionProvider"], 1, 60)
        self.assertEqual(report["status"], "failed")
        self.assertIn("dataset_ids", [d["code"] for d in report["diagnostics"]])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main([self.config["model"], "--inputs", self.config["inputs"], "--batch-size", "16"])

    def test_recorded_evidence_rejects_missing_rows_and_inflated_tolerances(self):
        path = ROOT / "results/digits-macos/b16/report.json"
        if not path.exists():
            self.skipTest("Real-model measurement not generated yet")
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        from check_digits import verify_case
        from reference_evidence import load_reference
        canonical = load_reference(path.parents[1] / "reference.json")
        report = json.loads(path.read_text())
        verify_case(report, self.root, canonical, 16)
        mutations = [
            lambda r: r["settings"].update(atol=1),
            lambda r: r["providers"].pop(),
            lambda r: r["providers"][0]["runs"][0]["latency_sample_rows"].pop(),
            lambda r: r["providers"][0]["runs"][0]["dataset"]["timed_passes"][0].update(valid_rows=448),
            lambda r: r["providers"][0]["runs"][0]["dataset"]["timed_passes"][0].update(predictions_sha256="wrong"),
            lambda r: r["providers"][0]["summary"]["dataset"].update(correct=450),
            lambda r: r["providers"][0]["summary"]["dataset"]["per_trial_inference_us_per_example"].__setitem__(0, 0),
            lambda r: r["providers"][0]["runs"][0]["memory"]["rss_bytes_by_stage"].update(after_session=-1),
            lambda r: r["providers"][0]["runs"][0]["dataset"]["evaluation"].update(validated_wall_ms=-1),
        ]
        for mutation in mutations:
            changed = copy.deepcopy(report)
            mutation(changed)
            with self.assertRaises(ValueError):
                verify_case(changed, self.root, canonical, 16)


if __name__ == "__main__":
    unittest.main()
