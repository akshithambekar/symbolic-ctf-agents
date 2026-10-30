# Imported scope and progress

Imported selectively from `yahya-akshith_rl` at `663e9b0`; original studies remain
on that branch. These historical results describe different experimental protocols.

## Included

- 1v1 and shared-policy 2v2/3v3 environment wrappers, static/patrol/chase opponents.
- PPO training, checkpoint continuation, paired evaluation, metrics and plots.
- Controlled-English/LLM translation, finite-trace validation, reward guidance,
  specification provenance, and a predictive bounds shield.
- Scripted capture oracle, native watch/replay viewer, relevant core tests.

The [historical 1v1 archive](../research_archive/README.md) now contains five
patrol checkpoints and selected chase-study records. The other experiment
archives, manuscripts, translation benchmarks, tactical-role studies, imitation
training, and chase/shield follow-up runners remain on the source branch.

## Recorded outcomes

| Study | Outcome | Limit |
| --- | --- | --- |
| Original guided 1v1 | Static 20/20; patrol 18/20; chase 0/20 captures | One training seed |
| Later patrol curriculum | 138/150 captures across five seeds | Selected softened-guidance/no-tag-penalty configuration |
| Subsequent chase stage | 0/150 captures across five seeds | Full-speed chase acquisition unresolved |
| Offensive bounds-shield replication | Bounds episodes 75/75 → 0/75; captures 46/75 → 50/75 | Selected boundary stress condition; failures in other conditions remain recorded |

The source branch records 95 passing tests on Windows. Historical outcomes do not
show that guidance works better in general. The source branch contains the results
in `RESULTS.md`, the Phase 3 stage summaries, and
`experiments/offense5-20261012-v1/REPORT.md`.

## Local verification (2026-10-13)

The selective import passed **66 tests on macOS ARM64**, including simulator
pickup/capture, reset reproducibility, logic semantics, specification provenance,
1v1/3v3 PPO train-save-reload-evaluate, and headless recording/replay. Dependency
lock consistency also passed. These short integration runs verify execution; they
do not reproduce the historical policy-performance studies above.

PyQuaticus emitted 24 divide-by-zero warnings in parallel-line geometry during
tests. The suite passed; observations and rewards are checked for finiteness in
the relevant integration tests. Live desktop viewing was not exercised.

## Gantt status

Baseline control, RL interfaces, symbolic integration, and all three opponent
scenarios are implemented for research. Static and patrol successes are recorded.
Reactive-opponent training and evaluation exist, with reliable chase capture still
open. Team play exists; learned coordination needs further evidence.

## Next work

1. Resolve the competition-rule differences in [the RL contract](RL_PIPELINE.md).
2. Decide whether full-state observations and waypoint actions fit our project.
3. Review rewards for opponent captures, collisions, defense, and task progress.
4. Reproduce a small local baseline/guided comparison before larger experiments.
5. Diagnose chase approach/pickup failures and evaluate changes with matched seeds.

Use common task metrics across methods. Keep training-seed variation separate from
episode variation, and report shield deadlocks alongside violations and captures.
