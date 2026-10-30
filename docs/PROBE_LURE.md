# Probe, lure, and switch experiments

The experiment measures the complete sequence: upper probe, committed pursuit,
withdrawal, lower attack, flag pickup, and capture. `G(in_bounds)` applies
throughout the episode. A qualified completion requires both LTL satisfaction
and a simulator capture. Raw captures and wins are reported separately.

The current formula permits tagging and recovery between milestones. Pursuit
commitment is a geometric proxy based on position, speed, and heading. It does
not prove that the defender inferred or fell for a deception.

## Experiment settings

The standard arena is 160 by 80, with a 240-second deadline. Blue has full
simulator-state observations and nine waypoint actions. Red uses the scripted
reactive chase policy. Defender speed 1.0 is the primary target. Slower speeds
are curriculum conditions and are labelled in each result.

The study retains the full formula and observation shape during curriculum
training. Phases 2 through 6 end an episode when the corresponding prefix is
complete. Phase 0 runs the full task. Promotion requires the configured success
rate on development starts.

Training uses a tag penalty of 0.5, a one-time reward of 2 for each new milestone,
and a reward of 24 for mission acceptance. Raw capture reward is zero in this
study, so a capture that skips the strategy does not receive the main task
reward. Potential-based guidance retains zero terminal potential. Completing
phase 6 pays the acceptance reward once.

## Train PPO

Run commands from the repository root with the installed environment:

```bash
MPLCONFIGDIR=/tmp/mctf-mpl .venv/bin/python scripts/train_probe_lure_full.py \
  --output runs/probe-lure-study \
  --chunk-steps 4096 --stage-steps 32768 \
  --validation-episodes 20 --eval-episodes 100 --variants
```

The default curriculum is `2, 3, 4, 5, 6, 0`. `--start-phase 0` trains directly
on the full task. `--opponent-speed 0.6` selects an easier training opponent.
The primary final evaluation always uses speed 1.0. With `--variants`, final
evaluation also covers the training speed, speed 0.8, and wide and narrow arenas.

`--max-chunks 1` stops after one saved chunk. To continue, repeat the original
command with `--resume`. Training settings and source hashes must match the
manifest. Use the run's `source_snapshot` with its `src` directory on
`PYTHONPATH` if the working source has changed.

Resume restores policy and optimizer state and restarts simulator episodes with
an explicit derived seed. It discards logs from an interrupted, uncheckpointed
chunk. This is a checkpointed continuation, rather than exact restoration of
the interrupted simulator state.

## Initialize from demonstrations

The teacher uses the same nine actions. It has verified full completions against
a speed-0.6 defender. At full speed it waits for the defender to climb before
the probe, then reaches the lower attack. The flag grab remains out of reach:
from the on-side release, the equal-speed defender is closer to the flag.

```bash
MPLCONFIGDIR=/tmp/mctf-mpl .venv/bin/python scripts/bootstrap_probe_lure.py \
  --output runs/probe-lure-bootstrap \
  --demonstrations 50 --epochs 200 --validation-episodes 20
```

The script collects successful full-mission demonstrations and trains the policy
actor by imitation. It saves the best development checkpoint under `selected`.
Use `--initialize-from runs/probe-lure-bootstrap/selected/policy.zip` with the
PPO runner to continue training. Report this method as imitation-assisted PPO.
The bootstrap itself performs supervised updates and zero PPO environment steps.

## Read results

- `manifest.json` records settings, package versions, source hashes, seed ranges,
  and initialization provenance.
- `state.json` records completed chunks, stage progress, and selected checkpoints.
- `training_episodes.jsonl` records each completed training episode.
- Each checkpoint contains development trajectories and a validation summary.
- `summary.json` contains final results for the selected saved checkpoint.
- `test-*` directories contain final per-episode trajectories and summaries.

Demonstration seeds start at 1000, development seeds at 20000, and final test
seeds at 30000. Checkpoint selection uses development results. Final tests do
not update checkpoint selection. Use fresh test seeds for a later study if prior
test results influence its design.

`status: stage_budget_exhausted` means the current stage failed to meet the
promotion threshold within its step cap. A previously selected checkpoint can
still have successful full-mission results. Read training status and measured
performance separately.

Each selected directory is compatible with the viewer:

```bash
uv run --locked mctf watch runs/probe-lure-study/selected/step-000004096/policy.zip \
  --seed 30000 --episodes 1 --speed 1
```

Choose an actual selected directory from `state.json`. A selected bundle uses
the full-speed evaluation configuration, even when training used a slower
opponent. Pass `--config` with that run's slower `arena_config.json` to inspect
the slower condition.

## Verify the implementation

```bash
MPLCONFIGDIR=/tmp/mctf-mpl .venv/bin/python -m pytest \
  tests/test_probe_lure.py tests/test_probe_lure_runner.py \
  tests/test_research.py tests/test_pipeline.py tests/test_viewer.py -q
```

These checks cover ordered mission progression, one-time rewards, later
curriculum stages, the successful slower-opponent teacher, checkpoint
continuation, development selection, and rendering. Translation benchmark
integration tests have separate dependencies, including Spot.
