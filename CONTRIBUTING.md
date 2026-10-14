# Development

Use Python 3.12 on macOS or Linux. Create an isolated environment and install `requirements-dev.txt` as described in the README. Direct and transitive runtime versions for the recorded suite are pinned in `requirements-lock.txt`; dependencies retain their upstream licenses.

Run:

```sh
python -m unittest discover -s tests -v
python -O -m unittest discover -s tests
python tools/check_project.py
python tools/run_mutations.py
python tools/measure_suite.py --iterations 10 --repeats 1
```

Tests invoke the installed production CLI and worker with actual ONNX Runtime sessions, plus independent NumPy references. They reject wrong final-row/secondary-output values, a timed-call-only corruption, invalid tensor names/dtypes/shapes, missing providers, nonfinite values, tiny resource budgets, external weights, worker timeout and tampered recorded evidence. No accelerator is required in CI. CI timings are smoke checks and never performance gates.

If package source changes, its hash changes. Regenerate the recorded suite with the revised source before the evidence tests and publication check can pass. Do not edit old numbers to match a claim. Keep generated models/NPZ files, local results, environments, credentials, personal instructions, raw logs and machine paths out of Git. The publication scanner is a basic heuristic, not a guarantee of secret detection.

The suite deliberately retains original synthetic reference values as JSON. Exact NPZ reconstruction verifies the measured reference hash; cross-platform NumPy regeneration is compared numerically under fixed suite-policy tolerances, rather than demanding bit-identical BLAS results. Tests reject both reference tampering and recomputation errors beyond tolerance.

The repository has no project license selected yet. Do not assume this code is licensed under the licenses of its dependencies or sibling projects.

Source mutation checks run targeted behavioral tests against isolated copies of the actual package and tools. The unchanged baseline must pass, and each deliberately faulty version must fail by assertion rather than import error or stale-source-hash rejection. Policy fixtures that relabel CPU computations test controller behavior only; they are not accelerator measurements.

Real-model validation requires `requirements-real-model.txt`. Run `python tools/check_digits.py` and `python tools/run_mutations.py --real-model`; CI installs these dependencies and exercises all real-model tests. The four additional source faults test an incorrect exported Softmax axis, skipped timed dataset output checks, an omitted final row and held-out rows reaching fit. CoreML measurements are local actual backend runs; Linux/macOS CI timing smoke uses CPU and checks the committed CoreML evidence without claiming to reproduce accelerator timings.
