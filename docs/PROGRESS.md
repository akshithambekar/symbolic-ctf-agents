# Imported scope and progress

Imported selectively from `yahya-akshith_rl` at `663e9b0`.

## Included

- 1v1 and shared-policy 2v2/3v3 environment wrappers, static/patrol/chase opponents.
- PPO training, checkpoint continuation, paired evaluation, metrics and plots.
- Controlled-English/LLM translation, finite-trace validation, reward guidance,
  specification provenance, and a predictive bounds shield.
- Scripted capture oracle, native watch/replay viewer, relevant core tests.

## Local verification (2026-10-13)

The selective import passed **66 tests on macOS ARM64**, including simulator
pickup/capture, reset reproducibility, logic semantics, specification provenance,
1v1/3v3 PPO train-save-reload-evaluate, and headless recording/replay. Dependency
lock consistency also passed. These short integration runs verify execution.

PyQuaticus emitted 24 divide-by-zero warnings in parallel-line geometry during
tests. The suite passed; observations and rewards are checked for finiteness in
the relevant integration tests. Live desktop viewing was not exercised.

## Gantt status

Baseline control, RL interfaces, symbolic integration, and all three opponent
scenarios are implemented for research. Reestablish static and patrol baselines
before testing chase and team play.

## Next work

1. Resolve the competition-rule differences in [the RL contract](RL_PIPELINE.md).
2. Decide whether full-state observations and waypoint actions fit our project.
3. Review rewards for opponent captures, collisions, defense, and task progress.
4. Reproduce a small local baseline/guided comparison before larger experiments.
5. Diagnose chase approach/pickup failures and evaluate changes with matched seeds.

Use common task metrics across methods. Keep training-seed variation separate from
episode variation, and report shield deadlocks alongside violations and captures.
