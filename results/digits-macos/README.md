# Real-model dataset measurements

Trained 64→32 ReLU→10 Softmax MLP on 1347 digits; all 450 held-out rows are scored. Independent sklearn reference: **436/450 (96.89%)**.

All cases share identical ONNX model, input, reference and label artifacts. Four fresh trials per requested provider alternate CPU/CoreML order; profiling follows all timing trials within each case. Each trial performs a first batch, one untimed complete evaluation, five warmup calls on the first batch, and five timed complete passes.

**Inference µs/example** is the median of the four trial means: sum of actual synchronous call times divided by 450×5 valid examples. **Validated wall** also includes slicing, output validation, class scoring and coverage hashes. Neither is raw kernel throughput. Short final batches are computed and checked, with no padding.

| Maximum batch | Calls/pass | Final rows | Requested primary | Correct | Inference µs/example | Validated wall µs/example | Peak worker RSS MiB |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 450 | 1 | CPUExecutionProvider | 436/450 | 4.162 | 31.702 | 62.05 |
| 1 | 450 | 1 | CoreMLExecutionProvider | 436/450 | 17.931 | 42.742 | 70.08 |
| 16 | 29 | 2 | CPUExecutionProvider | 436/450 | 0.328 | 2.188 | 61.44 |
| 16 | 29 | 2 | CoreMLExecutionProvider | 436/450 | 3.075 | 5.336 | 69.91 |
| 64 | 8 | 2 | CPUExecutionProvider | 436/450 | 0.117 | 0.754 | 61.67 |
| 64 | 8 | 2 | CoreMLExecutionProvider | 436/450 | 1.210 | 1.938 | 70.30 |
| 128 | 4 | 66 | CPUExecutionProvider | 436/450 | 0.079 | 0.446 | 61.61 |
| 128 | 4 | 66 | CoreMLExecutionProvider | 436/450 | 0.723 | 1.175 | 70.30 |

Reported host: Apple M5 Pro, Darwin 27.0.1, arm64.

Each report retains raw call times paired with actual row counts, every pass's correctness/coverage/prediction fingerprint, startup/first-call timing, per-trial normalized performance and RSS stages. These are descriptive measurements on one machine, without thermal control or statistical ranking claims.

**RSS is the whole validated worker, not model-only RAM or total deployment memory.** It includes the full loaded test/reference arrays and checks, and excludes controller preflight, profiling processes, CoreML helper services and device memory. Differences across batches can be allocator/system noise. EP attribution is not proof of Neural Engine/GPU use or identical assignment in every timing trial.

The split is local to sklearn's bundled 1,797-row subset, not the original UCI evaluation protocol or a verified writer-independent split. There is no hyperparameter search on the held-out set. Pixel normalization is fixed division by 16, with no fitted preprocessing. Exact image duplicates and ID overlaps across the split are rejected.

## Evidence

- [Training/test provenance and reference accuracy](manifest.json)
- [Retained reference tensors](reference.json)
- [Retained test labels, row IDs and class mapping](labels.json)
- [Frozen numeric sklearn state](../../examples/digits/model.json)
- [Preparation and export](../../examples/digits/prepare.py)
- Batch 1: [HTML](b1/report.html) · [JSON](b1/report.json) · `passed`
- Batch 16: [HTML](b16/report.html) · [JSON](b16/report.json) · `passed`
- Batch 64: [HTML](b64/report.html) · [JSON](b64/report.json) · `passed`
- Batch 128: [HTML](b128/report.html) · [JSON](b128/report.json) · `passed`

Download/open HTML locally; GitHub does not host the report UI.

Reproduce with `python tools/measure_digits.py --output results/local/digits --include-coreml`; install `requirements-real-model.txt` first. Check recorded evidence with `python tools/check_digits.py`.

[Dataset attribution](../../THIRD_PARTY_NOTICES.md) · [Methodology](../../docs/METHODOLOGY.md)
