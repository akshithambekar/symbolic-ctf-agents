# Research archive

This directory preserves the dated experiment outputs that were stored beside the working project before this repository branch was assembled. The experiment folders were copied without editing their run data, configurations, trained policy files, evaluation traces, or source snapshots. Each archived phase remains a separate record; exploratory results are not mixed with confirmatory results.

Included suites and artifacts:

- `matched-ablation-20260918-v1` — frozen matched 3-vs-3 five-method ablation and its records.
- `matched-ablation-20260918` — earlier matched-ablation artifact set.
- `matched-ablation-preflight` — preflight outputs.
- `offensive-phase-20260924-v1` — diagnosis, oracle validation, and staged exploratory Phase 3 work.
- `phase3-confirmatory-20260927-v1` — staged confirmatory study outputs.
- `chase-recovery-20261002-v1` — focused chase-recovery diagnostic.
- `translator-benchmark-20260925-v1` and `translator-benchmark-20260925-v2` — translator benchmark results and human-review artifacts.
- `viewer-*` folders — live viewer, oracle, guidance, and historical replay outputs.
- `openrouter-smoke-*` folders — translator provider smoke-test results, including the earlier failed model request.
- `export-bundles/` — previously created project ZIP exports.

`MANIFEST.sha256` records the archived files for integrity checks. The root `MANIFEST.sha256` covers the project's original source inventory; its listed hashes were refreshed only where repository packaging required documentation or ignore-rule updates. Frozen run directories and their internal manifests were not changed.

On Windows, enable Git long-path support before cloning because the preserved upstream PyQuaticus mission snapshot contains deeply nested filenames: `git config --global core.longpaths true`.

Local API credentials, `.env` files, Python virtual environments, caches, and the installed `vendor/pyquaticus` working copy are excluded. Dr. Dasgupta's unreleased paper, meeting notes, and private transcript are not included in this public repository branch.
