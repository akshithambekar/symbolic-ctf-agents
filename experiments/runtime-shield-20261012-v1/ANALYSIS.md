# Post-run analysis: paired runtime-only bounds shield

Completed 360 episodes, 180 matched pairs, 2 independently trained checkpoints. Both arms use the same saved patrol-trained neural policies, deterministic action selection, seeds, maps, opponents, and evaluation horizon; only the runtime gate differs. No retraining or guidance features were introduced.

| Map | Opponent | Arm | Captures / episodes | Pickup % | Territory % | Tag % | OOB % | Deadlock % | Common reward |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| standard | static | unshielded | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| standard | static | runtime_bounds_shield | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| standard | patrol | unshielded | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| standard | patrol | runtime_bounds_shield | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| standard | chase | unshielded | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| standard | chase | runtime_bounds_shield | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| wide | static | unshielded | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| wide | static | runtime_bounds_shield | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| wide | patrol | unshielded | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| wide | patrol | runtime_bounds_shield | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| wide | chase | unshielded | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| wide | chase | runtime_bounds_shield | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| narrow | static | unshielded | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| narrow | static | runtime_bounds_shield | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| narrow | patrol | unshielded | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| narrow | patrol | runtime_bounds_shield | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| narrow | chase | unshielded | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |
| narrow | chase | runtime_bounds_shield | 0/20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -2.400 |

## Intervention evidence

The shield performed 43200 successor checks, substituted 0 actions, and recorded 0 interventions in enemy-territory/carrier states, 0 near the flag, and 0 while carrying. 180/180 pairs had no shield events and identical recorded action/state trajectories.

The shield was dormant throughout this sample. Both checkpoints already avoided the checked bounds failures. Consequently this study does not estimate how shielding changes behavior when an unsafe proposal occurs. It establishes no safety advantage, no offensive benefit, and no shield-caused suppression of offensive actions. If a policy fails to approach the flag before filtering, a safety filter alone supplies no task-learning signal.

## Limits and next useful test

This is exploratory with two training seeds. Held-out episodes are paired observations within training replicates, not extra independent training seeds. These policies were trained against patrol; static/chase and wider/narrower arenas are transfer evaluations. There are no successful capture times to compare when capture counts are zero. Deadlocks and numerical termination are explicitly recorded because shorter exposure can improve common reward without improving control.

Before spending more evaluation budget, obtain a conventionally trained checkpoint that demonstrates offensive progress and inspect whether it ever proposes unsafe boundary actions. Reuse that same checkpoint and held-out initial conditions with the gate off/on. A separate, clearly labeled boundary-stress evaluation could assess intervention effectiveness; it would not replace task-level performance testing.

## Integrity

All expected pairs and configurations were checked. Frozen source and simulator hashes, saved model weights and training configurations, raw metrics, trajectories, and intervention events accompany this analysis. Earlier Phase 2/3 results are unchanged. Timing is diagnostic wall-clock measurement, not a controlled performance benchmark.
