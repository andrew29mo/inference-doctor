# Real-model and batch comparison review

Reviewed on 2026-10-14, starting from `2ded1e6`. This review concerns the trained Digits experiment; the [older MVP review](REVIEW.md) remains historical.

## Findings and fixes

The production controller previously accepted a worker's accuracy of 1.0 alongside 436 correct predictions out of 450, a negative validated-pass wall time, or a zero inference total alongside positive raw call times. These faults were reproduced by modifying a real worker result at the process boundary. No such inconsistency was found in the retained reports. The controller now rejects these contradictions as `worker_protocol`, checks untimed evaluation fields, and verifies positive integer RSS observations and consistent profile evaluation evidence.

The retained-dataset checker previously used the maximum RSS observation without validating every observation. A negative stage value could be hidden by other valid values. It now checks every observation and nonempty stage evidence, plus untimed evaluation wall-time and timing-field semantics.

Regression tests start from actual production-worker results, confirm the original is accepted, and require rejection after each targeted alteration. They are protocol tests; the mocked process boundary does not constitute accelerator evidence.

## Training and reference evidence

- The fixed split contains 1,347 training rows and 450 test rows, with disjoint IDs and no exact image duplicates across partitions. Normalization divides by 16 without fitted statistics. The training entrypoint passes only training arrays to the real fitting helper; a targeted test captures those arguments, and a source mutation that passes the complete dataset is rejected.
- The fixed recipe has no held-out tuning, early stopping or calibration. A fresh local retraining reproduced every retained coefficient and intercept exactly (maximum absolute difference 0.0; 180 optimizer iterations). This strengthens reproducibility on this environment; cross-platform optimizer equality is not required or promised.
- Probability references come from the frozen float64 sklearn model and preserve its classes after float32 conversion. Logit references come from NumPy using the exported float32 weights. Both outputs are checked under fixed tolerances; task accuracy uses public ground-truth labels rather than reference agreement alone.
- The source cannot prove undocumented historical human decisions. This is one local split of sklearn's bundled subset, not an independently held-out writer evaluation or the original UCI evaluation protocol.

## Production-path and coverage evidence

The tests invoke the production CLI and fresh workers with actual ORT sessions. All four maximum batch sizes (1, 16, 64, 128) score the complete 450-row dataset. Their calls per pass are 450, 29, 8 and 4; final batch sizes are 1, 2, 2 and 66. No padding is used.

A corruption introduced only in the last row of the final timed batch is detected by the production output validator. Additional tests execute real ORT calls while omitting, duplicating or reversing batches: valid individual output comparisons cannot mask incorrect dataset coverage. Changing only the final ground-truth label changes measured accuracy as expected.

Ordinary and optimized Python each pass all 41 tests. All 16 deliberately faulty source variants are rejected by behavioral assertion failures, including disabled timed validation, changed Softmax axis, omitted final predictions, training on held-out rows, accepted inconsistent accuracy and unchecked batch order. Import errors and source-hash mismatches do not count as mutation detection.

## Remeasurement and interpretation

All synthetic and real-model reports were rerun after the production-source change. The [generated dataset table](../results/digits-macos/README.md) is derived from raw reports. Each batch case uses identical model/input/reference/label artifacts for both requested providers, four alternating fresh-worker trials, five warmup calls and five complete timed passes. Profiling is separate and follows timing.

All CPU/CoreML runs and profiling evaluations reproduce the same 436/450 correct predictions (96.89%). In this rerun, median inference cost ranges from 4.163 to 0.076 microseconds/example for CPU and from 18.118 to 0.694 for CoreML as maximum batch increases from 1 to 128. CPU is faster for these small workloads on this machine; this does not establish a general CPU/CoreML ranking. Larger batches amortize dispatch in this experiment, without establishing a universal preferred batch size.

Timing measures synchronous Python runtime calls, including output materialization and dispatch. Validation is outside the call timer but affects inter-call spacing/cache state; separate validated-wall results include that work. The initial full evaluation warms both full and tail shapes. These are warm descriptive measurements, not cold-start, service-load or statistically controlled performance claims. Cases run in fixed batch order; thermal state and OS caches are uncontrolled. Configured ORT thread limits do not prove equal internal CoreML resource use.

Peak worker RSS is approximately 61–62 MiB for CPU and 70 MiB for CoreML in this run. It includes Python, runtime libraries, all test/reference arrays and validation, and excludes the controller, separate profiling workers, CoreML helper services and device memory. It supports only this worker-process comparison. It cannot establish total deployment RAM, model-only allocation or accelerator memory efficiency. Profile attribution does not establish physical Neural Engine/GPU use or assignment in every timing trial.

Linux/macOS CI checks the actual CPU path and recorded report consistency; CI success alone is not evidence of actual CoreML execution. The committed CoreML measurements were run locally on the recorded Mac with the real provider available. The repository remains Private.
