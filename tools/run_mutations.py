"""Challenge real implementation guards using isolated source copies and targeted tests.

Only assertion failures count as detected faults; import/runtime errors do not.
Package mutations deliberately avoid recorded-source-hash tests, which would
otherwise reject every change without exercising the changed behavior.
"""

import os
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
D = "test_doctor.DoctorTests."
E = "test_evidence.EvidenceTests."
MUTATIONS = [
    ("unchecked_timed_output", "src/inference_doctor/worker.py", "            validate(outputs)\n", "            pass\n", D + "test_timed_only_fault_is_rejected_by_production_worker"),
    ("unchecked_secondary_output", "src/inference_doctor/validation.py", "for name, actual in zip(names, outputs):", "for name, actual in zip(names[:1], outputs[:1]):", D + "test_wrong_last_row_and_secondary_output_are_detected"),
    ("unchecked_final_row", "src/inference_doctor/validation.py", '        if actual.dtype.kind in "biu":', '        actual, expected = actual[:-1], expected[:-1]\n        if actual.dtype.kind in "biu":', D + "test_wrong_last_row_and_secondary_output_are_detected"),
    ("broadcast_reference", "src/inference_doctor/validation.py", "or actual.shape != expected.shape:", "or False:", D + "test_reference_shape_dtype_names_nonfinite_and_integer_exactness"),
    ("half_timed_calls", "src/inference_doctor/worker.py", 'range(config["iterations"]):', 'range(config["iterations"] // 2):', D + "test_three_real_onnx_graphs_and_dynamic_batches"),
    ("report_chooses_tolerance", "tools/check_project.py", '    require(report["settings"] == policy, "Recorded settings/tolerances differ from fixed suite policy")', '    pass  # fault: report chooses its own settings', E + "test_current_report_and_deliberate_evidence_corruption"),
    ("empty_output_evidence", "tools/check_project.py", '        require(sorted(o["name"] for o in outputs) == sorted(canonical), "Incomplete output evidence")', '        pass  # fault: no inventory check', E + "test_current_report_and_deliberate_evidence_corruption"),
    ("profile_error_downgraded", "src/inference_doctor/runner.py", 'issue("error", error["code"],', 'issue("warning", error["code"],', D + "test_profile_numerical_failure_is_an_error"),
    ("grouped_provider_order", "src/inference_doctor/runner.py", 'order = report["providers"] if trial % 2 == 0 else list(reversed(report["providers"]))', 'order = report["providers"]', D + "test_alternating_order_profiles_last_and_fallback_warnings"),
    ("ignored_stage_rss", "src/inference_doctor/runner.py", '*memory["rss_bytes_by_stage"].values())', '0)', D + "test_memory_budget_includes_every_observed_stage"),
]
REAL_MUTATIONS = [
    ("wrong_export_softmax_axis", "examples/digits/prepare.py", 'name="softmax", axis=1)', 'name="softmax", axis=0)', 'test_dataset.DatasetTests.test_actual_worker_scores_full_dataset_at_all_batch_sizes'),
    ("unchecked_dataset_timed_outputs", "src/inference_doctor/dataset.py", '            validate(outputs, expected)', '            if not timed:\n                validate(outputs, expected)', 'test_dataset.DatasetTests.test_timed_tail_only_corruption_is_detected_by_formal_worker'),
    ("omitted_final_dataset_row", "src/inference_doctor/dataset.py", '            predictions.append(predicted)', '            predictions.append(predicted[:-1] if timed and subset.stop == self.rows else predicted)', 'test_dataset.DatasetTests.test_actual_worker_scores_full_dataset_at_all_batch_sizes'),
    ("heldout_rows_reach_fit", "examples/digits/prepare.py", 'fit_classifier(x[train], y[train])', 'fit_classifier(x, y)', 'test_dataset.DatasetTests.test_fit_receives_only_the_fixed_training_split'),
]


def run_case(mutation):
    with tempfile.TemporaryDirectory(prefix="inference-doctor-mutation-") as directory:
        root = Path(directory)
        for name in ("src", "tools", "tests"):
            shutil.copytree(ROOT / name, root / name, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        shutil.copytree(ROOT / "examples/digits", root / "examples/digits", ignore=shutil.ignore_patterns("__pycache__"))
        (root / "results").symlink_to(ROOT / "results", target_is_directory=True)
        if mutation is not None:
            _, filename, before, after, test = mutation
            path = root / filename
            source = path.read_text()
            if source.count(before) != 1:
                raise ValueError(f"Mutation target is not unique: {mutation[0]}")
            path.write_text(source.replace(before, after))
            targets = [test]
        else:
            targets = sorted({m[-1] for m in MUTATIONS})
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / "src"), str(root / "tests")]),
                   PYTHONOPTIMIZE=str(sys.flags.optimize))
        completed = subprocess.run([sys.executable, "-m", "unittest", "-v", *targets], cwd=root,
                                   env=env, capture_output=True, text=True, timeout=120)
        if mutation is None:
            if completed.returncode:
                raise RuntimeError("Unmodified baseline failed:\n" + completed.stdout + completed.stderr)
        elif completed.returncode != 1 or "FAIL:" not in completed.stderr or "ERROR:" in completed.stderr:
            raise RuntimeError(f"Fault survived or did not fail by assertion: {mutation[0]}\n" + completed.stdout + completed.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-model", action="store_true", help="Include dataset/fit faults; requires requirements-real-model.txt")
    args = parser.parse_args()
    if args.real_model:
        MUTATIONS += REAL_MUTATIONS
    run_case(None)
    for mutation in MUTATIONS:
        run_case(mutation)
        print(f"Detected: {mutation[0]}", flush=True)
    print(f"Baseline passed; all {len(MUTATIONS)} deliberate faults were rejected by behavioral assertions.")
