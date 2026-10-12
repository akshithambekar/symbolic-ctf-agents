# Post-run analysis: paired runtime-only bounds shield

Completed 180 episodes, 90 matched pairs, 2 independently trained checkpoints. Both arms use the same saved unshielded neural checkpoints (see input config.json for the training method), deterministic action selection, seeds, maps, opponents, and evaluation horizon; only the runtime gate differs. No retraining or guidance features were introduced.

| Map | Opponent | Starts | Arm | Captures / episodes | Pickup % | Territory % | Tag % | OOB % | Deadlock % | Common reward |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| standard | static | standard | unshielded | 10/10 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 20.838 |
| standard | static | standard | runtime_bounds_shield | 10/10 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 20.838 |
| standard | patrol | standard | unshielded | 4/10 | 100.0 | 100.0 | 60.0 | 0.0 | 0.0 | 5.892 |
| standard | patrol | standard | runtime_bounds_shield | 4/10 | 100.0 | 100.0 | 60.0 | 0.0 | 0.0 | 5.892 |
| standard | chase | standard | unshielded | 0/10 | 0.0 | 100.0 | 100.0 | 0.0 | 0.0 | -6.400 |
| standard | chase | standard | runtime_bounds_shield | 0/10 | 0.0 | 100.0 | 100.0 | 0.0 | 0.0 | -6.400 |
| wide | static | standard | unshielded | 10/10 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 20.548 |
| wide | static | standard | runtime_bounds_shield | 10/10 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 20.548 |
| wide | patrol | standard | unshielded | 8/10 | 100.0 | 100.0 | 20.0 | 0.0 | 0.0 | 15.950 |
| wide | patrol | standard | runtime_bounds_shield | 8/10 | 100.0 | 100.0 | 20.0 | 0.0 | 0.0 | 15.950 |
| wide | chase | standard | unshielded | 0/10 | 0.0 | 100.0 | 100.0 | 0.0 | 0.0 | -6.400 |
| wide | chase | standard | runtime_bounds_shield | 0/10 | 0.0 | 100.0 | 100.0 | 0.0 | 0.0 | -6.400 |
| narrow | static | standard | unshielded | 10/10 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 20.898 |
| narrow | static | standard | runtime_bounds_shield | 10/10 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 20.898 |
| narrow | patrol | standard | unshielded | 0/10 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | -2.800 |
| narrow | patrol | standard | runtime_bounds_shield | 0/10 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | -2.800 |
| narrow | chase | standard | unshielded | 0/10 | 0.0 | 100.0 | 100.0 | 0.0 | 0.0 | -8.400 |
| narrow | chase | standard | runtime_bounds_shield | 0/10 | 0.0 | 100.0 | 100.0 | 0.0 | 0.0 | -8.400 |

## Intervention evidence

The shield performed 16874 successor checks, substituted 0 actions, and recorded 0 interventions in enemy-territory/carrier states, 0 near the flag, and 0 while carrying. 90/90 pairs had no shield events and identical recorded action/state trajectories.

The shield was dormant throughout this sample. The checkpoints already avoided the checked bounds failures. Consequently this study does not estimate how shielding changes behavior when an unsafe proposal occurs. It establishes no safety advantage, no offensive benefit, and no shield-caused suppression of offensive actions.

## Limits and next useful test

This is exploratory with two training seeds. Held-out episodes are paired observations within training replicates, not extra independent training seeds. The training method and opponent are preserved in inputs/. Any new map, opponent, or boundary start is a transfer/stress condition; boundary starts are engineered exposure probes and are not pooled with ordinary starts for task-performance claims. Capture time is computed only among successful episodes. Deadlocks and numerical termination are explicitly recorded because shorter exposure can improve common reward without improving control.

Before spending more evaluation budget, verify offensive capability in the saved checkpoint and inspect whether it ever proposes unsafe boundary actions. Reuse that same checkpoint and held-out initial conditions with the gate off/on. A separate, clearly labeled boundary-stress evaluation could assess intervention effectiveness; it would not replace task-level performance testing.

## Integrity

All expected pairs and configurations were checked. Frozen source and simulator hashes, saved model weights and training configurations, raw metrics, trajectories, and intervention events accompany this analysis. Earlier Phase 2/3 results are unchanged. Timing is diagnostic wall-clock measurement, not a controlled performance benchmark.
