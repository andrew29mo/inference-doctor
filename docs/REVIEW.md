# Historical MVP review (0.1.1)

Reviewed baseline: `adda9f0`. Reviewed package: `0.1.1`, report schema 2. Its measurements superseded the earlier schema-1 results. Current 0.2.0 reports add dataset mode and were remeasured; historical observations remain in Git and are not mixed into current summaries.

## Findings and corrections

1. **Evidence validator accepted incomplete records and report-selected tolerances.** Reproductions accepted an empty provider list, empty output evidence, or `rtol=atol=1e6`. The suite now has a separate fixed acceptance policy (`rtol=1e-4`, `atol=1e-5`, four trials, exact cases/providers). It requires all input/output metadata, validation counts, profiling correctness, node attribution, execution order and summaries. HTML must match the verified JSON. Schema/version/source/model metadata must agree.
2. **Provider order and profiling placement introduced avoidable ordering effects.** Timing now alternates CPU/CoreML and CoreML/CPU across four rounds. All timing ends before profiling starts within each invocation. OS caches, accelerator caches and thermal state are still uncontrolled; this is not a cold-start or statistical ranking experiment.
3. **Profiling numerical failures were downgraded to attribution warnings.** A wrong output or incomplete correctness evidence in the profiling session now fails the diagnostic. Ordinary profiling unavailability still warns when all timing trials succeeded.
4. **Memory aggregation omitted stage snapshots as an independent observation.** Peak/budget calculations now use the maximum of OS high-water RSS, sampled RSS and all stage snapshots. Previous output arrays are released outside the call timer before the next call. Reference comparison allocations remain included in worker memory, explicitly documented.
5. **Presentation could imply total accelerator memory or physical device identity.** The table now labels the requested primary provider and states its exclusions immediately above the measurements. CPU model/architecture are reported when observable; no hostname or serial identifier is queried. Int64 error diagnostics also preserve exact integer differences rather than rounding large values through float64.
6. **Output alias protection missed symlinks to input files.** Resolved report targets are checked before writing. A regression test confirms the original input bytes survive rejection.

## Evidence of behavioral coverage

- 30 tests passed under ordinary Python and `python -O`. Tests launch the production CLI/worker and execute real ORT sessions with independent NumPy references. Optimized entrypoint tests propagate the optimization setting to subprocesses and verify it in worker evidence.
- The source-mutation baseline passed; all ten deliberate faults failed by behavioral assertions: unchecked timed outputs, unchecked secondary outputs, ignored final rows, broadcast references, half the timed calls, report-selected tolerances, missing output evidence, downgraded profiling errors, unbalanced provider order, and ignored stage RSS.
- Mutations use isolated copies of the real package/tools. Runtime-package mutations are tested without recorded-source-hash checks, avoiding the misleading result where every mutant is rejected solely because its hash changed. Import errors do not count as detection.
- Controller policy fixtures that relabel CPU results exercise scheduling/fallback-warning logic only. They are explicitly not evidence of a real accelerator or real mixed-provider execution. The committed CPU/CoreML measurements use the actual requested backends.
- Exact reconstruction of the retained reference NPZ must match its recorded SHA-256. Independently regenerated NumPy values must match within fixed policy tolerances; actual cross-platform drift is printed per case. Tests accept a one-ULP change, reject absolute/relative errors beyond tolerance, reject inflated tolerances, and reject modified retained references.

## What the measurements establish

The five recorded workloads use original synthetic models. MLP batch 1/16/64 have identical model bytes; CPU/CoreML batch 16 use the same model, inputs and reference snapshot. Each requested provider has four fresh timing processes, 50 warmups and 500 measured calls per process; every call is checked outside its timer. Profiling uses another process and one independently checked inference.

The results support descriptive synchronous-call latency and validated-worker RSS for this machine/workload. The CPU/CoreML latency paths use matching Python interfaces and data, including dispatch/transfers/output allocation; provider internals may use different thread/device resources. No general accelerator speedup, sustained throughput or causal explanation is established.

**The memory numbers cannot compare total deployment memory.** They exclude controller preflight, profiling workers, CoreML helper services and device memory, and include interpreter/libraries/input/reference/validation allocations. A CoreML EP event proves attribution only in the separate profile session, not Neural Engine/GPU execution or attribution in every timing trial. The MVP does not establish dataset accuracy or production readiness.
