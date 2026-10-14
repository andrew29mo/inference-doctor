# Security policy

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/andrew29mo/inference-doctor/security/advisories/new) to report a suspected vulnerability privately. Include the affected commit/version, reproduction steps and impact. Do not post credentials, private datasets or sensitive model details in public issues or pull requests. No response-time guarantee is promised.

## Maintenance and dependency updates

Only the latest `main` is maintained; there is no supported release/backport policy. These experimental tools are not security-certified inference sandboxes. Run models, datasets and build scripts only when you trust their provenance.

Dependabot checks GitHub Actions and Python dependencies weekly and can open security-update PRs for vulnerable dependencies. Updates are not automatically merged. Fixed dependency versions and retained measurement evidence can require deliberate reference regeneration or remeasurement; a green CI run does not prove that a dependency is free of vulnerabilities.

## Solo-maintainer contribution workflow

1. Create a branch and pull request; direct changes to `main` are restricted.
2. All configured CI checks must succeed against an up-to-date base before merging. Checks are tied to the GitHub Actions integration.
3. The maintainer may merge their own PR without another person's approval; required approving reviews are zero. Review conversations must be resolved.
4. Main-branch deletion and force pushes are blocked. There is no administrator bypass in the ruleset.

Actions use read-only repository tokens, pinned and explicitly allowed action revisions, and checkout without persisted credentials. All external fork contributors require maintainer approval before PR workflows run. Review untrusted changes before approving; approval permits their code to execute on a hosted runner. No self-hosted runners or privileged `pull_request_target` execution are used by the current workflows.

Only the explicitly allowed official action repositories may run, and their workflow references must use full commit SHAs. Dependabot may propose a new pinned revision within those allowed repositories without a separate policy edit; adding a different action repository requires an explicit allowlist change. Review dependency and action changes before merging. SHA pinning alone does not establish that an action is safe.

Repository protections are server-side settings and can change independently of Git history. This policy is not a guarantee that no vulnerabilities or credential leaks exist. Source-data and dependency licenses remain governed by their respective notices.

The security hardening update pins setuptools 83.0.0 to address [GHSA-h35f-9h28-mq5c](https://github.com/advisories/GHSA-h35f-9h28-mq5c), which concerns file exclusions while building source distributions on normalization-sensitive filesystems. Historical measurement records retain their original dependency versions; they were not rewritten as measurements of the patched environment.
