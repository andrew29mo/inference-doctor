# Inference Doctor

A small ONNX deployment diagnostic tool: verify outputs, measure actual inference calls and process memory, and explain which execution-provider evidence is available.

It answers a focused question: **Does this model run correctly with these inputs, on this runtime/provider, within my measured latency and process-RSS budgets?**

## MVP features

- Exact input names, tensor dtypes, fixed shapes and shared symbolic dimensions are checked before inference. No silent conversion or broadcasting.
- Every first, warmup and timed output is verified against an optional independent NPZ reference. Missing references produce a warning, never a correctness claim.
- Fresh worker processes measure session construction, first inference, warm-call distributions and whole-process RSS. Raw samples and trial variability are retained.
- Requested providers must be available and register successfully. A separate profiling session reports observed kernel-provider assignments, CPU partitions and missing evidence.
- Offline HTML and JSON reports include diagnoses, source/artifact hashes, settings, software versions, memory stages and explicit measurement limits.
- Optional per-trial p95 latency and peak-process-RSS budgets produce machine-readable failures.

This wraps ONNX Runtime; it is not a new inference engine. A provider being selected does not prove GPU/NPU/Neural Engine execution.

## Quick start

Python 3.12 on macOS or Linux:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python tools/make_examples.py --kind mlp --batch 16
inference-doctor examples/generated/mlp-b16/model.onnx \
  --inputs examples/generated/mlp-b16/inputs.npz \
  --reference examples/generated/mlp-b16/reference.npz \
  --output results/local/mlp-b16
```

Open `results/local/mlp-b16/report.html`. The matching `report.json` retains all evidence. Generated models and input/reference data stay ignored; the committed generator reproduces them.

For your own model, input NPZ keys must equal the model input names, and reference NPZ keys must equal **all** output names. References must have matching shapes/dtypes and come from an implementation you trust. This tool cannot establish whether an arbitrary supplied reference is independent. Only trusted local, self-contained ONNX models and finite numeric/boolean dense tensors are supported.

Compare CPU and CoreML on an installation that provides both:

```sh
inference-doctor examples/generated/mlp-b16/model.onnx \
  --inputs examples/generated/mlp-b16/inputs.npz \
  --reference examples/generated/mlp-b16/reference.npz \
  --provider CPUExecutionProvider --provider CoreMLExecutionProvider \
  --repeats 4 --warmup 50 --iterations 500 \
  --output results/local/providers
```

An unavailable or failed requested backend produces a failed report. It is not silently presented as a successful CPU comparison. CoreML needs access to its system temporary directories and services; sandbox restrictions can prevent compilation. Diagnose in the same environment where you intend to deploy.

To enforce budgets, add `--max-p95-ms 1 --max-peak-mib 128` with thresholds appropriate to your application. The memory budget covers the **validated worker process**, including Python, libraries, inputs/references and checks; it is not model-only RAM or total accelerator memory.

## Actual measurements

[Recorded measurements and interpretation](results/README.md) cover five workloads: a linear graph, an MLP at batches 1/16/64, and a residual graph. CPU runs all five; CoreML is also measured on the identical MLP batch-16 model, inputs and reference.

These are actual ONNX Runtime executions with independent NumPy output checks, not simulated estimates. These five baseline examples are original seeded **synthetic models**, not trained task models, and make no dataset-accuracy claim. Tiny graphs emphasize dispatch costs. Neither an overall accelerator speedup nor model-only memory savings is claimed.

Every report contains session and first-run timing separately from warm inference. Provider order alternates across four trials in the recorded suite; profiling follows all timing trials in that invocation and is excluded from timing/RSS summaries. OS caches are not cleared. Process RSS excludes CoreML helper processes and device memory, so it cannot establish total CPU-versus-CoreML memory use. See the full [measurement contract](docs/METHODOLOGY.md).

Reproduce the suite:

```sh
python tools/measure_suite.py --output results/local/suite
# On a Mac with CoreML available and necessary system access:
python tools/measure_suite.py --output results/local/suite-coreml --include-coreml
```

The suite writes every outcome, including failures, into its index. Its exit status is nonzero if any case failed; warnings remain zero. Inspect individual reports for diagnoses.

## Real model and complete dataset

[The trained Digits example](examples/digits/README.md) adds task accuracy and full-dataset batching to the same production worker. It retains an original 64→32→10 MLP, fixed 1,347/450 train/test IDs and independent sklearn/NumPy references. [Measured CPU/CoreML comparisons](results/digits-macos/README.md) use identical model/data/label artifacts at maximum batches 1, 16, 64 and 128, including actual short final batches.

```sh
python -m pip install -r requirements-real-model.txt
python examples/digits/prepare.py
python tools/measure_digits.py --output results/local/digits --include-coreml
```

Dataset mode requires `--labels`, `--batch-size`, `--classification-output` and `--reference`. The label NPZ contains 1D int64 `labels`, unique `sample_ids`, and ordered `classes`. All inputs/references share the same leading row dimension. `--iterations` means complete timed dataset passes in this mode. Every output and every class prediction is checked; each pass records coverage and prediction fingerprints. No padding is performed. Call percentiles mix full/short batches; use the dataset per-example table to compare normalized inference costs. A separate wall-time figure includes validation and audit work.

Worker RSS includes the whole loaded test/reference dataset and validation allocations. It excludes training/sklearn, controller work, CoreML helper processes and device memory. The accuracy is one local held-out estimate; it does not establish a writer-independent benchmark or production generalization. [Dataset attribution and dependency notices](THIRD_PARTY_NOTICES.md).

## Verification and architecture

```sh
python -m unittest discover -s tests -v
python -O -m unittest discover -s tests
python tools/check_project.py
python tools/run_mutations.py
# Full real-model validation (requires optional dependencies):
python tools/check_digits.py
python tools/run_mutations.py --real-model
```

Tests exercise the installed CLI and production worker with real ORT sessions. They challenge all outputs and the final batch row, deliberate timed-only corruption, invalid inputs, missing providers, budgets, timeouts and report consistency. Ten core deliberately faulty source variants must fail behavioral assertions; a source-hash mismatch alone does not count as detection. CI runs on Linux and macOS; its timing smoke tests are not performance gates.

The [real-model review](docs/REAL_MODEL_REVIEW.md) covers training isolation, full/tail batch coverage and CPU/CoreML comparison limits. The optional full mutation suite adds six dataset/export/training faults, including inconsistent accuracy evidence and reordered batches.

The controller snapshots inputs, performs ONNX preflight and launches independent measurement/profile workers. The worker validates inputs, runs ORT and checks outputs. The controller computes summaries and conservative diagnoses, then writes escaped static HTML and JSON. [Development and verification details](CONTRIBUTING.md).

## Limits and license

The MVP measures labelled classification accuracy in dataset mode; it does not measure concurrent service load, sustained thermal behavior, physical accelerator assignment, device/helper-process memory, or production readiness. It supports a fixed input set or a finite labelled dataset per invocation, with no external ONNX weight files, custom provider options, string/sequence/map tensors, I/O binding or Windows RSS accounting.

The original code, documentation and trained weight state are licensed under the [MIT License](LICENSE). Third-party materials are excluded from this grant: the Digits source data and dataset-derived fixture values retain CC BY 4.0 terms where applicable, and dependencies retain their own licenses. See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution, changes and license links.

The [pre-publication review](docs/PUBLICATION_REVIEW.md) records verification scope and remaining limits. Repository visibility is a separate owner decision.
