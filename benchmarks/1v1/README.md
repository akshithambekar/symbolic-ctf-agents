# PyQuaticus 1v1 checkpoint benchmark, v1

`v1.json` is the frozen 50-case scenario roster. Its IDs, seeds, explicit Red spawns, and route definitions are the benchmark identity. Case 43 moves Blue from the proposed (21, 38) to (23, 38): the original point is rejected by PyQuaticus as colliding with the Blue flag; the 2 m east shift is the smallest integer-coordinate adjustment tested that resets successfully. The runner validates the roster before loading a policy and verifies each reset against the resolved Blue and Red positions.

Run the suite twice into fresh output directories:

```sh
MPLCONFIGDIR=/tmp/mplcache PYTHONPATH=src .venv/bin/python benchmarks/run_1v1.py \
  --output runs/1v1-benchmark-v1/run-1
MPLCONFIGDIR=/tmp/mplcache PYTHONPATH=src .venv/bin/python benchmarks/run_1v1.py \
  --output runs/1v1-benchmark-v1/run-2
```

The default checkpoint is `runs/fixed-start-orbit-ppo-20261119/training/policy.zip`. Use `--checkpoint` and `--specification` to select another compatible artifact. The runner loads PPO once on CPU and uses deterministic predictions. `--case-id 50 --output <new-directory>` runs a single-case smoke check.

Each output contains `provenance.json`, newline-delimited per-case rows in `cases.jsonl`, and `summary.json`. The case rows include simulator-observed milestone and loss events plus one-second Red and Blue trajectory samples. Capture is counted from the simulator capture counter. Simulator exceptions stop the run with an error instead of adding an ordinary failure row. Compare `cases.jsonl` in the two output directories by case ID and milestone timestamps to check determinism.

The fixed-start-orbit policy has no standard config/specification sidecars. Its saved `config.json`, `experiment.py`, `launch_training.py`, and run summary document the 19-value `MaritimeEnv` observation, seven waypoint actions, unguided PPO, 5-second repeat, 240-second horizon, gamma 0.99, Heron dynamics, and the custom orbit opponent used during training. The benchmark loads the stock `ppo-nov` specification because guidance is disabled; formula stages and shaped guidance rewards do not affect policy inputs or benchmark reward. The result provenance includes the checkpoint and source hashes used for each run.

The fixed patrol controllers command the case's stated fraction of Red's 2 m/s maximum speed. Cases 45–49 use a 16 m trigger, 8-second timeout, 18 m distance from route leash, and 12-second cooldown. Case 50 has no trigger distance, timeout, leash, or cooldown and pursues any untagged Blue boat on Red territory. The case 50 route uses the requested idle line at x=112 between y=18 and y=62.
