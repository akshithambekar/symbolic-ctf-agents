# Tactical strategies and capable-attacker study

This is a new exploratory phase. Earlier Phase 2/3 and runtime-only results are preserved unchanged. The goal was to supply player tactics and a task-capable unchanged neural checkpoint for an actual runtime-shield intervention test.

## Attacker construction

Two independently trained seeds each used 24 CaptureOracle demonstrations, 160 behavioral-cloning epochs, and 4,096 PPO decisions. Training used static opponents in the standard arena, the existing eight waypoint actions, conventional task reward, and no temporal observation, formal reward shaping, or shield. Each final policy passed its separate 10-episode static capability screen with 10 captures. This is imitation-initialized PPO, not the earlier scratch-trained neural baseline. Teacher demonstrations, curves, PPO logs, checkpoints, configurations and source snapshots are retained under attackers/.

## Same-checkpoint shield comparison

| Evaluation context | Episodes per arm | Captures off/on | Bounds-violation episodes off/on | Substitutions | Shield deadlocks |
|---|---:|---:|---:|---:|---:|
| Ordinary starts, 3 maps and 3 opponents | 90 | 42 / 42 | 0 / 0 | 0 | 0 |
| Stopped boundary starts, standard map | 36 | 18 / 18 | 0 / 0 | 0 | 0 |
| Moving boundary starts, selected moderate stress | 36 | 16 / 16 | 18 / 0 | 18 | 0 |
| Severe moving start, one seed, lower edge | 3 | 1 / 0 | 3 / 0 | 0 | 3 |

Off/on arms use identical checkpoints and paired evaluation seeds. Ordinary-start static captures were 30/30, patrol 12/30, and chase 0/30; chase also had zero pickups. All ordinary pairs had identical trajectories because the gate was dormant.

The moderate stress uses a valid 2.01 m vessel-center boundary margin and an outward initial speed of 1 m/s. Both training seeds separately had nine bounds-violation episodes out of 18 without the shield and zero with it, with unchanged capture counts. All 18 interventions occurred at the lower home boundary; upper-edge pairs were dormant. Substitutions did not occur near the enemy flag or while carrying. Mean tags increased from 0.889 to 1.056, so this is a bounds-safety result, not tagging protection. Mission satisfaction increased from 10/36 to 16/36. Common reward increased from 5.503 to 6.220; successful capture time was 136.75 versus 125.5 seconds, conditional on success.

This supplies evidence that the implemented gate can prevent tested unsafe proposals while preserving captures in a controlled exposure condition. It does not establish general safety guarantees, improved chase performance, or superiority of neuro-symbolic learning. Evaluation episodes are not independent training replicates. The two seeds produced equal aggregate results; this small sample is not a seed-uncertainty estimate.

The condition was selected after an explicit nine-cell calibration with one checkpoint (stress-probes/selection.json). The final paired run uses both checkpoints and fresh evaluation seeds. Selection is exploratory, not preregistration or confirmatory testing. At higher outward speeds (1.5–2 m/s), the gate found no feasible candidate and terminated episodes. The severe screen's zero violations therefore reflect early termination, not a working safe attacker; its higher short-horizon reward must not be interpreted as improved performance.

## Per-player tactical library

The additive TacticalArena provides attacking flank routes, carrier return, defender interception/home patrol, and support/escort waypoints. It supports no defender, fixed Agent 0 defender, conditional defense on opponent intrusion, and dynamic nearest noncarrier assignment. Flag carriers retain return duties. Human-authored role constraints are assembled with the mission and freshly validated; the conditional clause is G(opponent_push -> defender_behind_center). Role-specific grounding is explicit and separate from the original Agent-0 predicate. Contracts are monitored, not guaranteed or enforced by the bounds gate.

The 3v3 scripted screen covers four role modes against static, patrol, chase and an explicitly raiding opponent: 48 episodes, three per cell. Against raid, each defensive mode achieved 3/3 blue captures and zero red captures; no-defender achieved zero blue captures and three red captures. Chase produced zero captures in all modes. No role-contract violations were observed in this screen, but conditional defense can violate its instantaneous contract if the selected defender is outside home territory when intrusion starts.

These are scripted feasibility probes, not learned team coordination or a strategy ranking. Dynamic mode also uses escort attack style while the other modes use split flanks, so it is not an isolated role ablation. tactics/REPORT.md and raw role-labelled traces contain every cell, including unsuccessful ones.

## Integrity and next study

Exact run configurations, specifications, source hashes, raw metrics, trajectories, proposed/substituted actions and checkpoints accompany each completed study. The failed long-path preflight and invalid sub-radius initial-position trial are preserved as setup failures and excluded from measured results. Viewers were checked headlessly for both tactics and the new PPO checkpoint. Tests include actual moving-start shield physics, valid initial vessel footprints, conditional grounding and carrier-aware role assignment.

Next: retain the capable static attacker for shield checks, improve chase acquisition separately, and test proactive bounds recovery before reaching states with no feasible candidate. Promote only successful candidates to a larger matched seed-level study. Learned 3v3 tactics should follow demonstrated chase capability; this phase does not claim to have solved that stage.
