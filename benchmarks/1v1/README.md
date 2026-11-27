# PyQuaticus 1v1 checkpoint benchmark

This benchmark evaluates a saved PPO checkpoint against the 50 fixed scenarios in [`v1.json`](v1.json). It reports simulator-measured captures and milestone outcomes.

Run the full suite into a new output directory:

```sh
MPLCONFIGDIR=/tmp/mplcache PYTHONPATH=src .venv/bin/python benchmarks/run_1v1.py \
  runs/fixed-start-orbit-ppo-20261119/training/policy.zip \
  --output runs/1v1-benchmark-v1/run-1
```

The checkpoint path is a required positional argument. Run again with a different `--output` path to compare repeatability. Use `--specification` to select another compatible specification, or add `--case-id 50` to run one case.

Each output directory contains:

- `summary.json`: capture and milestone totals overall and by scenario family, plus case 50.
- `cases.jsonl`: per-case milestone times, pickups, flag losses, tags, out-of-bounds events, end reason, and sampled Red and Blue paths.
- `provenance.json`: checkpoint and suite hashes, saved specification, and effective environment settings.
