# Pre-publication review

Reviewed on 2026-10-14, starting from `347dfaa`. The repository remains Private; this review does not authorize changing visibility or assigning a license.

## Corrections

- The contributor guide described four additional real-model source faults while the suite implements six. It now lists all six, consistent with the README and mutation runner.
- The README now explicitly distinguishes making source visible from granting an open-source license. No license was assigned or copied from another project.
- CI's official checkout/setup-python actions were updated to pinned Node.js 24 versions after verifying their upstream action manifests. This removes reliance on the runner's forced migration of older Node.js actions. Job permissions remain `contents: read`.

No new inference defect or numerical inconsistency was found in this review. The prior production fixes and coverage challenges are documented in the [real-model review](REAL_MODEL_REVIEW.md).

## Verification and performance claims

All 41 tests passed under normal and optimized Python. Tests exercise the production CLI/worker and real ORT calls, with deliberate input/output/coverage faults. All 16 source mutations must fail behavioral assertions; a hash mismatch or import failure alone does not count.

An isolated build using the pinned build dependency produced the 0.2.0 wheel successfully. Its nine package Python files match the reviewed source byte-for-byte; the remaining five files are package metadata. It contains no model/data fixtures, local paths, personal settings or environment files, and declares no project license. Build artifacts were kept outside tracked files.

Both evidence checkers pass. The five synthetic cases and four trained-model batch cases retain matching package-source hashes, exact model/input/label artifact checks, fixed numerical tolerances, complete validation/coverage counts, recomputed timings/RSS summaries and HTML generated from the verified JSON. Generated overview tables match their raw reports. The trained-model reference and every recorded CPU/CoreML pass agree on 436/450 correct predictions; training isolation and short-tail tests remain intact.

No inference package source changed in this review, so the preceding real hardware measurements remain applicable; timings were not edited or rerun merely to obtain different numbers. The reported local observations do not establish a universal provider ranking, writer-independent model accuracy, physical Neural Engine/GPU execution or total deployment memory. Worker RSS excludes the controller, profiling workers, CoreML helper services and device memory. CI executes CPU smoke measurements and checks retained CoreML evidence; it does not certify CoreML hardware execution. See [methodology](METHODOLOGY.md) and the generated [synthetic](../results/README.md) and [trained-model](../results/digits-macos/README.md) result tables.

## Publication hygiene

The review scanned all tracked working-tree files and all five reachable commits, including 153 historical file blobs at the starting revision. No credential patterns, private home paths, local-service URLs, unexpected binaries or personal instruction files were found. All author/committer addresses use GitHub noreply addresses. The GitHub username, CPU model, architecture, runtime versions and measurement timestamps are intentional provenance, not anonymized data.

Tracked numeric model state, test labels/IDs, references and generated reports are intentional reproducibility evidence. Data originates from the attributed public Digits dataset. The largest historical blob is approximately 1.13 MB of report HTML. Generated ONNX/NPZ files, local results, environments and build output remain ignored. Personal workspace instructions were not staged or uploaded. This was a heuristic scan and manual review, not a guarantee that every possible secret representation can be detected; unreachable/deleted Git objects, other repositories, GitHub issues and external services were outside scope.

## License and visibility

No project `LICENSE` exists, package metadata declares no license, and GitHub reports no detected project license. README, contributor guidance and third-party notices agree that the original code and trained weight state have no selected project license. This is consistent, but it is not an open-source license grant. The owner must choose a license before describing the original code as open source; public visibility itself is a separate decision. [GitHub licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository) explains the distinction and GitHub's viewing/forking rights.

The [third-party notices](../THIRD_PARTY_NOTICES.md) retain dataset authors, source/DOI, CC BY 4.0 attribution and experiment modifications. The [UCI source](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) confirms the dataset license. Dependencies retain their own licenses and are not vendored. The dataset terms do not assign a project-code license.
