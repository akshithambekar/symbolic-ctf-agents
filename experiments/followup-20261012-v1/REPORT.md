# Offensive shielding, role control and staged chase follow-up

Exploratory screens only. Previous studies are unchanged. Exact source snapshots, simulator hashes, policies, configurations and raw traces accompany the runs. Two independently trained source checkpoints are reused unchanged for the shield study. Each exposure cell has one held-out episode per training seed; repetitions are not additional independent seeds.

## Offensive exposure and deadlock recovery

| Condition | Arm | Captures / episodes | Pickup episodes | Bounds episodes | Deadlocks | Substitutions | Offensive / carrier substitutions |
|---|---|---:|---:|---:|---:|---:|---:|
| ordinary | off | 3/6 | 4 | 0 | 0 | 0 | 0 / 0 |
| ordinary | original | 3/6 | 4 | 0 | 0 | 0 | 0 / 0 |
| ordinary | brake | 3/6 | 4 | 0 | 0 | 0 | 0 / 0 |
| ordinary | proactive | 3/6 | 4 | 0 | 0 | 0 | 0 / 0 |
| approach_tight | off | 4/6 | 4 | 6 | 0 | 0 | 0 / 0 |
| approach_tight | original | 4/6 | 4 | 0 | 0 | 6 | 6 / 0 |
| approach_tight | brake | 4/6 | 4 | 0 | 0 | 6 | 6 / 0 |
| approach_tight | proactive | 0/6 | 0 | 0 | 6 | 0 | 0 / 0 |
| approach_reserve | off | 4/6 | 4 | 6 | 0 | 0 | 0 / 0 |
| approach_reserve | original | 0/6 | 0 | 0 | 6 | 0 | 0 / 0 |
| approach_reserve | brake | 0/6 | 0 | 0 | 6 | 0 | 0 / 0 |
| approach_reserve | proactive | 0/6 | 0 | 0 | 6 | 0 | 0 / 0 |
| return_tight | off | 3/6 | 0 | 0 | 0 | 0 | 0 / 0 |
| return_tight | original | 3/6 | 0 | 0 | 0 | 0 | 0 / 0 |
| return_tight | brake | 3/6 | 0 | 0 | 0 | 0 | 0 / 0 |
| return_tight | proactive | 0/6 | 0 | 0 | 6 | 0 | 0 / 0 |
| return_reserve | off | 0/6 | 0 | 6 | 0 | 0 | 0 / 0 |
| return_reserve | original | 0/6 | 0 | 0 | 6 | 0 | 0 / 0 |
| return_reserve | brake | 0/6 | 0 | 0 | 6 | 0 | 0 / 0 |
| return_reserve | proactive | 0/6 | 0 | 0 | 6 | 0 | 0 / 0 |
| return_severe | off | 0/6 | 0 | 6 | 0 | 0 | 0 / 0 |
| return_severe | original | 0/6 | 0 | 0 | 6 | 0 | 0 / 0 |
| return_severe | brake | 0/6 | 0 | 0 | 6 | 0 | 0 / 0 |
| return_severe | proactive | 0/6 | 0 | 0 | 6 | 0 | 0 / 0 |

The original gate enforces predicted G(in_bounds). Brake adds zero-speed/turn candidate controls; proactive additionally rejects successors whose current velocity projects outside center bounds in two seconds. That reserve is a heuristic strengthening, not a certified braking model. The learned action space is unchanged. Raw substituted controls are authoritative: emergency controls are labelled hold (7) in legacy waypoint counts.

Approach/return starts are engineered enemy-territory edge conditions, not ordinary full missions. Return starts initialize possession with PyQuaticus init_dict; their captures test return behavior, not acquisition. All paired initial states were checked. Deadlocks terminate early; reduced time penalties or tags in short episodes are not evidence of improved control. No result supports a general invariant or superiority claim.

## Role strategies with attack style controlled

| Attack style | Roles | Opponent | Blue captures / 3 | Red captures | Mission satisfied / 3 |
|---|---|---|---:|---:|---:|
| split_flank | none | static | 2 | 0 | 2 |
| split_flank | none | patrol | 1 | 0 | 1 |
| split_flank | none | chase | 0 | 0 | 0 |
| split_flank | none | raid | 0 | 3 | 0 |
| split_flank | fixed | static | 0 | 0 | 0 |
| split_flank | fixed | patrol | 0 | 0 | 0 |
| split_flank | fixed | chase | 0 | 0 | 0 |
| split_flank | fixed | raid | 3 | 0 | 3 |
| split_flank | conditional | static | 0 | 0 | 0 |
| split_flank | conditional | patrol | 3 | 0 | 3 |
| split_flank | conditional | chase | 0 | 0 | 0 |
| split_flank | conditional | raid | 3 | 0 | 3 |
| split_flank | dynamic | static | 1 | 0 | 1 |
| split_flank | dynamic | patrol | 0 | 0 | 0 |
| split_flank | dynamic | chase | 0 | 0 | 0 |
| split_flank | dynamic | raid | 3 | 0 | 3 |
| escort | none | static | 2 | 0 | 2 |
| escort | none | patrol | 1 | 0 | 1 |
| escort | none | chase | 0 | 0 | 0 |
| escort | none | raid | 0 | 3 | 0 |
| escort | fixed | static | 0 | 0 | 0 |
| escort | fixed | patrol | 0 | 0 | 0 |
| escort | fixed | chase | 0 | 0 | 0 |
| escort | fixed | raid | 3 | 0 | 3 |
| escort | conditional | static | 0 | 0 | 0 |
| escort | conditional | patrol | 0 | 0 | 0 |
| escort | conditional | chase | 0 | 0 | 0 |
| escort | conditional | raid | 3 | 0 | 3 |
| escort | dynamic | static | 1 | 0 | 1 |
| escort | dynamic | patrol | 0 | 0 | 0 |
| escort | dynamic | chase | 0 | 0 | 0 |
| escort | dynamic | raid | 3 | 0 | 3 |

These are scripted tactical policies, not learned coordination. Attack style is identical within each four-role comparison, resolving the earlier role/style confounding. Three episodes per cell are feasibility evidence, not a robust ranking.

## Carrier/defender conflict

| Mode | Reassign carrier duty | Initial roles | Captures | Mission satisfied |
|---|---|---|---:|---:|
| fixed | False | ['carrier', 'support', 'support'] | 1 | False |
| fixed | True | ['carrier', 'defender', 'support'] | 1 | True |
| conditional | False | ['carrier', 'support', 'support'] | 1 | False |
| conditional | True | ['carrier', 'defender', 'support'] | 1 | True |
| dynamic | False | ['carrier', 'defender', 'support'] | 1 | True |
| dynamic | True | ['carrier', 'defender', 'support'] | 1 | True |

Carrier priority is preserved in every variant. Optional fallback assigns another available player to defense when fixed Agent 0 carries; dynamic mode already excludes carriers. These intentionally supplied carrier/intrusion states test contract conflicts, not learned pickups. The role constraint remains instantaneous; reassignment does not guarantee a defender is in position.

## Staged chase acquisition

| Variant | Seed | Added decisions | Completed curriculum | Full-speed pickup % | Full-speed capture % | Static capture % | Candidate |
|---|---:|---:|---|---:|---:|---:|---|
| unmodified | 0 | 0 | False | 0 | 0 | 100 | False |
| unmodified | 1 | 0 | False | 0 | 0 | 100 | False |
| task | 0 | 4096 | False | 0 | 0 | 100 | False |
| progress | 0 | 4096 | False | 0 | 0 | 100 | False |
| directional | 0 | 4096 | False | 0 | 0 | 100 | False |
| task | 1 | 4096 | False | 0 | 0 | 100 | False |
| progress | 1 | 4096 | False | 0 | 0 | 100 | False |
| directional | 1 | 4096 | False | 0 | 0 | 100 | False |

Each training stage adds 2,048 decisions at chase speeds 0.65, 0.85, then 1.0. Advancement requires pickups in at least 20% of five held-out episodes and static captures in at least 60% of five separate episodes. Failed candidates stop; they are still evaluated at full chase speed as transfer diagnostics. Different total budgets after stopping must not be described as matched full-curriculum training.

Task, flag/home progress, and progress-gated local evasion use identical observations/actions and initial checkpoints. These are task-reward continuation variants without formal training guidance. Common task reward is separate from training return. A candidate flag is a screening recommendation, not a five-seed conclusion.

## Next decision

Reject any recovery variant that reduces captures through early termination. Promote an offensive shield condition only when it has nonzero offensive substitutions, fewer bounds episodes, unchanged or better captures, and no deadlocks; freeze a separate held-out protocol before a larger comparison. Chase and role results remain exploratory.

## Post-hoc easier-speed reference

The unmodified source policies were subsequently evaluated on the same lower-speed screen seeds. This reference was not used to alter promotion. A passed capability screen does not establish learning gains.

| Seed | Chase speed | Unmodified pickups / 5 |
|---|---:|---:|
| 0 | 0.65 | 2 |
| 0 | 0.85 | 0 |
| 1 | 0.65 | 2 |
| 1 | 0.85 | 0 |

Compare stage screen files against these references before attributing any lower-speed success to continuation training. The full-speed baseline and continued variants also use matched evaluation seeds.

## Completed selected-condition replication

The original gate passed the offensive screen. A separate five-seed replication with fresh evaluation seeds is recorded in ../offense5-20261012-v1/REPORT.md: bounds violations 75/75 off versus 0/75 on; captures 46/75 versus 50/75, zero deadlocks, and 75 offensive substitutions. Three checkpoints are new; two were reused from selection. See chase/REWARD_AUDIT.md for the reward/task mismatch. None of the six chase variants was promoted.
