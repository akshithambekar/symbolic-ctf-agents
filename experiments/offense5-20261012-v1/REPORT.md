# Five-seed matched offensive-exposure replication

The original one-step gate was selected from the earlier two-seed screen. This freezes a narrow engineered enemy-territory approach condition: 2.01 m lower-edge margin, 1 m/s outward speed, standard arena. Seeds 0/1 reuse selection checkpoints; seeds 2/3/4 are new independently trained replicas of the same imitation-initialized PPO protocol. All 150 held-out evaluation episodes use fresh seeds and exact same-checkpoint off/on pairing.

| Arm | Captures / episodes | Pickups | Bounds episodes | Deadlocks | Offensive substitutions | Common reward | Mean tags |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 46/75 | 50 | 75 | 0 | 0 | 8.156 | 0.707 |
| original | 50/75 | 50 | 0 | 0 | 75 | 11.953 | 0.733 |

## Independent training units

| Seed | Arm | Captures / 15 | Bounds episodes | Deadlocks |
|---|---|---:|---:|---:|
| 0 | off | 10 | 15 | 0 |
| 0 | original | 10 | 0 | 0 |
| 1 | off | 8 | 15 | 0 |
| 1 | original | 10 | 0 | 0 |
| 2 | off | 10 | 15 | 0 |
| 2 | original | 10 | 0 | 0 |
| 3 | off | 9 | 15 | 0 |
| 3 | original | 10 | 0 | 0 |
| 4 | off | 9 | 15 | 0 |
| 4 | original | 10 | 0 | 0 |

Seed-level paired deltas and separate points are retained; evaluation episodes are not treated as independent training seeds. Zero sample variance in a seed metric is not certainty about other distributions. No pooled-episode confidence interval is claimed as seed-level uncertainty.

## Opponent breakdown

| Opponent | Arm | Captures / 25 | Pickups | Bounds episodes |
|---|---|---:|---:|---:|
| static | off | 25 | 25 | 25 |
| static | original | 25 | 25 | 0 |
| patrol | off | 21 | 25 | 25 |
| patrol | original | 25 | 25 | 0 |
| chase | off | 0 | 0 | 25 |
| chase | original | 0 | 0 | 0 |

## Scope

This measures deployment-only filtering of identical task-capable checkpoints, not neuro-symbolic training superiority. Any bounds benefit applies to this selected condition and exact simulator model. It is not an unbiased comparison across all game states: the stress condition was chosen after calibration and deliberately exposes unsafe proposals. Five seeds provide replication of that condition, not a general invariant. Original high-speed deadlocks and rejected proactive/braking variants remain in the separate screen. Chase acquisition in ordinary full missions remains a separate question.

All 75 paired configurations and initial states were verified. Raw metrics, exact models/configurations/specifications, source hashes and full state/control traces are retained. Earlier results are unchanged.
