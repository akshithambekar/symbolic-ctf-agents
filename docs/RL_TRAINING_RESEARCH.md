# Training PPO for reactive maritime capture the flag

Research report · November 3, 2026

Code reviewed at `main` commit `999e83e`. The historical 1v1 `TeamArena` study is pinned to `663e9b0`.

## Goal

The project has two deliverables, and one training pipeline has to produce both.

- **A competition-ready blue team for MCTF 2026.** It must play the official game in [`GAME_RULES.md`](../GAME_RULES.md): 3v3, 10-minute matches, scored as team captures minus opponent captures, against opponents that both defend and attack. Ties go to the team with more grabs, then to the team with fewer collisions. A collision is coming within 4 m of any boat.
- **A paper that answers the research question in the [README](../README.md).** Can validated LTL specifications derived from natural-language strategies improve rule compliance and capture performance, compared with PPO that does not use symbolic guidance?

These two deliverables put four requirements on the plan:

- **Captures against a full-speed reactive defender are required.** Competition opponents chase at full speed, so a team that captures only at 60% speed is not ready.
- **Every training stage runs a guided arm and an unguided arm.** Guidance means the LTL reward shaping, the monitor state in the observation, and the shield where it is used. Any other change, such as new actions, curricula, start states, demonstrations, or architectures, goes into both arms. If only one arm gets a change, the paper comparison stops measuring guidance.
- **Both arms report captures and rule violations at every stage.** The reduced-speed stages also produce paper results. The comparison is most informative where unguided PPO is partly successful, neither stuck at 0% nor near 100%.
- **The research environment must move to the competition rules.** [`RL_PIPELINE.md`](RL_PIPELINE.md#where-the-research-setup-differs-from-competition-play) lists the gaps: capture geometry, boundary exit, Powerplay, 10-minute scoring, tagging, and collisions. The research environments also give agents the full simulator state. Confirm which observations competition agents receive before you settle the observation interface.

## TL;DR

- **PPO almost never sees a successful attack, so it has nothing to learn from.** Against a full-speed chasing defender, no run has produced a single flag pickup.
  - Historical 1v1: in 10,480 training episodes, the learner crossed into enemy territory 9,285 times and never got within pickup range.
  - 2v2 curriculum: 2 pickups and 0 captures in 2,500 patrol episodes, then 0 pickups in 2,500 chase episodes.
- **The first question is whether a full-speed capture is possible with the current controls.** The competition requires one. The scripted teacher captures 9/10 when the defender moves at 60% speed, and nothing we have tried captures at full speed.
- **More training under the current settings is unlikely to help.** First find out what can reach the flag, then train on that.
- **There may already be a first paper result.** Against 1v1 patrol, task-only PPO captured 0/150 and softened LTL guidance captured 29/150. Section 1 has the details.

### Recommended plan

1. **Check feasibility before training.** Use a script and a bounded planner to test whether the current actions can capture at each defender speed. Compare them with faster decisions and with a small set of finer maneuvers.
2. **Fix the competition rule gaps early.** Rule changes such as capture geometry and boundary exits change what is feasible. Paper results gathered before the fix might not carry over.
3. **Fix the curriculum.** Train on a mix of carry starts, near-flag starts, and normal starts. Promote to a harder stage only after the policy captures on held-out starts.
4. **Train guided and unguided arms side by side** with identical actions, curricula, and seeds.
5. **Keep the flat MLP as the control.** Try a frame-stacked MLP and a small LSTM only after the pipeline produces useful attack experience. Use an entity encoder for questions about teammates, opponents, or team size.
6. **Build up from 2v2 to 3v3, one role at a time.** Train the attacker with a frozen teammate, then train defense against opponents that score. Finish with full 10-minute 3v3 matches.

## Terms used in this report

| Term | Meaning |
| --- | --- |
| Pickup, capture | A pickup grabs the enemy flag. A capture brings it back to your own side. |
| Static, patrol, chase | Red defender behaviors. Chase pursues the attacker. "Speed 0.6" means chase at 60% of full speed. |
| Native step | One simulator step, about one simulated second. |
| Repeat | The number of native steps that one high-level action lasts. The default is 5. |
| Joint decision | One 2v2 decision point. It produces two agent transitions, one for each blue boat. |
| Teacher | The scripted probe-and-lure policy in [`probe_lure.py`](../src/maritime_nsrl/probe_lure.py). |
| Role mission | The 2v2 LTL formula: agent 0 stays on its half, and agent 1 must cross midfield. |
| Carry start | An episode that starts with the attacker already holding the flag. |

## 1. What the runs show

| Setup | Result | Takeaway |
| --- | --- | --- |
| Legacy 1v1 waypoint PPO | The checkpoint trained on static, then patrol, captured 10/10 vs static, 10/10 vs patrol, and 0/10 vs chase. Continuing that checkpoint directly against chase dropped to 0/10 in all three conditions. | PPO can learn approach and return. A sudden switch to chase failed and also erased the easier skills. |
| Historical 1v1 on `TeamArena` | The best patrol arm (softened guidance, no tag penalty) captured 138/150. Five fresh PPO models with the same settings vs chase entered enemy territory 150/150 and picked up 0/150. | Removing the tag penalty and reaching midfield were not enough at full speed. |
| 1v1 teacher | Speed 0.6: 9/10 captures. Speed 0.8: 1/10 pickups, 0 captures. Speed 1.0: reached the attack zone 10/10, 0 pickups. A policy cloned from the teacher captured 19/20 at speed 0.6 and 0/20 at full speed. | A capture is possible at reduced speed, and imitation works there. These runs do not measure what PPO fine-tuning adds. |
| 2v2 shared PPO, base mission | 3,000 chase episodes. 1,536 entered the enemy half, with 0 pickups. Final evaluation: 0/90. | Training against chase from the start produced no successful attacks. |
| 2v2 shared PPO, role mission | 2,500 patrol episodes: 922 entries, 2 pickups, 0 captures. Then 2,500 chase episodes: 678 entries, 0 pickups. Final evaluation: 0/90. | The curriculum moved to chase before the policy had captured once against patrol. |

Notes on these results:

- **The patrol arms show that the tag cost matters.** Task reward alone captured 0/150, and the temporal-state features alone also captured 0/150. Softened guidance captured 29/150, and softened guidance with no tag cost captured 138/150. The five 30-episode tests of the best arm scored 24, 29, 30, 26, and 29.
- **The 1v1 chase runs never came close.** Each of the five training logs has 100,608 decisions and 2,096 episodes. The closest approach to the flag was about 20.3 m, and the pickup radius is 10 m.
- **The two 2v2 pickups did not convert.** They came at 179 and 231 simulated seconds. One carrier was tagged, and the other ran out of time.
- **Training and deployment fail differently.** During training the 2v2 policy often crosses midfield. In the final chase evaluation, all 30 chase cases ended without crossing midfield or taking a tag.
- **Use the current teacher rerun for comparisons.** I reran the teacher on seeds 0–9 with repeat 5 and a 240-second deadline, and it reproduced 9/10 at speed 0.6. An earlier trial at speed 0.8 reported 8/10 pickups. The current rerun gets 1/10.
- **Each 2v2 evaluation covers one trained policy per arm.** The evaluation grid is 3 opponent types × 3 layouts × 10 seeds = 90 cases, and both arms use the same seeds.
- **Some records are only partly available.** The 1v1 demo and teacher results come from earlier run records. The original teacher aggregate and policy are not in the current checkout. The 1v1 patrol and chase CSVs live in the historical revision.

Sources: [progress record](PROGRESS.md) · [historical patrol CSV](https://github.com/akshithambekar/symbolic-ctf-agents/blob/663e9b0922e450bec0c9233017919877112072bf/research_archive/phase3-confirmatory-20260927-v1/stages/02-1v1-patrol/per_seed_summary.csv) · [historical chase CSV](https://github.com/akshithambekar/symbolic-ctf-agents/blob/663e9b0922e450bec0c9233017919877112072bf/research_archive/phase3-confirmatory-20260927-v1/stages/03-1v1-chase/per_seed_summary.csv) · [historical chase run directory](https://github.com/akshithambekar/symbolic-ctf-agents/tree/663e9b0922e450bec0c9233017919877112072bf/research_archive/phase3-confirmatory-20260927-v1/stages/03-1v1-chase/softened_guidance_no_tag_penalty) · [teacher study protocol](PROBE_LURE.md) · [teacher code](../src/maritime_nsrl/probe_lure.py#L85-L133) · [3,000-episode evaluation](../runs/team-2v2-3000/evaluation.csv) · [5,000-episode evaluation](../runs/team-2v2-role-curriculum-5000/evaluation.csv) · [2v2 patrol episode log](../runs/team-2v2-role-curriculum-5000/patrol-stage/training_episode_metrics.jsonl) · [2v2 chase episode log](../runs/team-2v2-role-curriculum-5000/chase-stage/training_episode_metrics.jsonl)

### A two-rule script beats both trained 2v2 checkpoints

I replayed both role-curriculum checkpoints and compared them with a simple script: agent 0 holds, and agent 1 goes for the flag and heads home once it has it. All three policies used the saved 2v2 role config, the standard arena, and seeds 31000–31002. Nothing was trained in this check.

| Red policy | Patrol checkpoint | Chase checkpoint | Hold-and-attack script |
| --- | ---: | ---: | ---: |
| Static | 0/3 captures | 0/3 | 3/3 |
| Patrol | 0/3 | 0/3 | 1/3 |
| Chase, speed 1.0 | 0/3 | 0/3 | 0/3 |

- All four of the script's captures satisfied the role mission.
- Against patrol, the script picked up the flag 3 times and took 6 tags. Against chase, it crossed every time, took 9 tags, and picked up 0 times.
- The patrol checkpoint never crossed midfield. The chase checkpoint crossed in all 3 static episodes, 1 patrol episode, and 0 chase episodes.
- In a separate three-seed screen, five simple scripts, including two-attacker routes, picked up 0 times against full-speed chase.

**Conclusion:** the current actions and role mission allow 2v2 captures against static and some patrol starts. The script is a reproducible attacker baseline. Whether anything can capture against full-speed chase is still open. To rerun this check, see [Reproduce the 2v2 check](#reproduce-the-2v2-check).

## 2. Why the learning signal disappears

There are four likely causes. Each subsection describes the problem, then the next test.

### Cause 1: the actions may not allow an escape

**Problem.**

- The legacy `MaritimeEnv` gives PPO 19 scaled features and 7 actions. The team wrapper has 8 actions, including flank, return lane, and evading the nearest opponent. The teacher study adds a 9th.
- Each high-level action lasts 5 native steps, about 5 seconds. The steering controller updates within that window, but the choice of action stays fixed.
- A chasing defender retargets every native step. An escape may need a sideways turn, a speed change, or a new target before the attacker's next decision.
- The team "evade" action steers straight away from the nearest opponent and is clipped to the arena edge.
- A capture happens as soon as the carrier reaches its own side. The "home" waypoint is just a steering target deeper inside that side.

The cause is a hypothesis. The outcomes so far cannot tell us the best result achievable with these actions, so test it with a controller.

**Next test: a feasibility screen.**

- Defender speeds: 0.60, 0.70, 0.80, 0.90, and 1.00.
- Action interfaces: the current actions at repeat 5, the same actions at repeat 1 or 2, and a small maneuver set. The maneuver set has heading offsets relative to the goal or the pursuer, plus a few speed levels.
- Controllers: direct flag pursuit, a scripted controller that accounts for interception, and a bounded lookahead planner.
- Starts: normal starts, a reachable state near the flag, and a valid carry state. Use the same snapshots and opponent policy for every interface.
- Records: time remaining, minimum tag margin, pickup, and return.

Before you run the screen:

- Keep red's pre-chase patrol speed fixed. The team chase multiplier also changes red's patrol speed, which shifts where pursuit starts.
- Change the teacher's progress detector from a distance per decision to a rate per second before you vary repeat.

A planner success proves that the interface allows a capture. A planner failure only describes the search budget and coverage that were tested.

**If the screen finds a full-speed route:** keep the waypoint controller for basic navigation, and give PPO the smallest set of maneuvers that works. Treat fully continuous speed and heading as a later experiment. A continuous action space makes the search harder, so it needs a scripted continuous baseline, bounded inputs, and logging of saturated actions. The closest related study found that its direct-velocity PPO and TD3 agents played defensively and rarely grabbed the flag. Its pretrained behavior options learned useful behavior sooner, though they also leaned defensive.

Sources: [legacy action space and repeat](../src/maritime_nsrl/single_agent_env.py#L41-L66) · [legacy step loop](../src/maritime_nsrl/single_agent_env.py#L140-L161) · [team actions and chase](../src/maritime_nsrl/team_env.py#L349-L415) · [pinned simulator capture rule](https://github.com/mit-ll-trusted-autonomy/pyquaticus/blob/475d670650ec5d49691ba707cd1054c4e45d8053/pyquaticus/envs/pyquaticus.py#L1758-L1780) · [Beason et al., maritime CTF competition, sections IV–V](https://arxiv.org/abs/2404.17038)

### Cause 2: the reward makes staying home look safe

**Problem.**

- Default rewards are +20 for a capture, +2 for a pickup, −2 per tag, and −0.01 per native step. A 240-second episode with no events costs 2.4.
- One tag cancels one pickup, and crossing midfield risks tags long before any capture bonus. If the policy rarely reaches the flag, holding back or retreating is a reasonable thing for it to learn.
- The patrol results in section 1 show that removing the tag cost mattered. The chase run with no tag cost still never picked up the flag, so test the tag cost together with an action interface that can actually attack.
- Shaped return can be misleading. In the archived chase reward audit, some heuristic arms had positive training return in all 42 episodes of a stage, for both seeds. Those arms had zero pickups and a mean task reward of −6.4.
- Potential-based shaping gives local feedback without changing which policy is optimal. With per-decision discount `gamma` and zero potential at terminal states, the shaping terms add up to a constant that depends only on the start state. Acceptance bonuses, safety penalties, and heuristic progress terms do not have this property, so they can change which behavior the objective favors.

**Next test.**

- Score every arm on the same scoreboard. Compare task reward only, task reward plus potential shaping, and task reward plus potential shaping plus one-time pickup or tactical milestones.
- Keep the temporal features the same in every arm.
- Test the tag cost as its own factor after you choose the reward signal.
- Log tag costs before and after pickup, net return, and every reward component.
- Give return credit based on who holds the flag. Test flag loss, re-pickup, repeated boundary crossings, and timeouts on purpose.
- Measure what the automaton observation adds in a separate ablation.
- Select policies by captures, return success after a pickup, and mission completion. Use shaped return only to debug the critic and the reward terms.

Reward-machine methods use subtask state well. Their counterfactual replay needs an off-policy learner, so it would be a separate comparison.

Sources: [single-agent reward](../src/maritime_nsrl/single_agent_env.py#L162-L193) · [team reward](../src/maritime_nsrl/team_env.py#L521-L535) · [team shaping](../src/maritime_nsrl/team_env.py#L618-L660) · [historical reward audit](https://github.com/akshithambekar/symbolic-ctf-agents/blob/663e9b0922e450bec0c9233017919877112072bf/experiments/followup-20261012-v1/chase/REWARD_AUDIT.md) · [Ng, Harada, and Russell, potential-based shaping](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf) · [Icarte et al., Reward Machines](https://arxiv.org/abs/2010.03950)

### Cause 3: the curriculum moves on before the policy succeeds

**Problem.** Static and patrol teach navigation and flag return. Chase adds interception as soon as the attacker crosses midfield. The timing window the teacher uses at speed 0.6 closes at full speed. A switch after a fixed number of episodes can take away the useful starts before the policy has learned either half of the mission.

**Next test: train the mission backward.**

- Build a bank of valid states from replays or supported simulator resets. Each saved state must keep who holds the flag, the flag position, cooldowns, time, and mission phase consistent.
- Train the return first against the relevant defender. Then work back to pickup, then to approach.
- Keep some normal starts at every stage, and keep sampling earlier opponents.
- A starting mix to measure: 30% carry starts, 30% near-flag starts, and 40% normal starts. Shift toward normal starts as held-out performance on normal starts improves.
- Score carry starts only as a return diagnostic. Require captures from normal starts before promotion.

The pieces mostly exist. The legacy wrapper has a carry-start hook. The teacher-study runner has development-based promotion, separate test seeds, and resumable chunks.

**Use demonstrations where the teacher succeeds.**

- Use demonstrations only where the teacher succeeds with the actions being trained.
- Include successful attacks, recoveries, interrupted approaches, and flag-loss states.
- Benchmark the teacher and the cloned checkpoint before any PPO update.
- If the teacher can recover from states the learner reaches, label those states DAgger-style.
- At equal environment budgets, compare behavior cloning alone, PPO from scratch, and cloning followed by PPO. Report the cost of collecting demonstrations and the cost of the teacher search separately.
- Keep the step-zero checkpoint in the comparison. Otherwise a run that keeps its starting checkpoint can look like a PPO gain.

Sources: [Reverse Curriculum Generation](https://proceedings.mlr.press/v78/florensa17a.html) · [legacy carry-start hook](../src/maritime_nsrl/single_agent_env.py#L92-L121) · [teacher-study runner promotion](../scripts/train_probe_lure_full.py#L349-L379) · [teacher-study checkpoint selection](../scripts/train_probe_lure_full.py#L291-L310) · [Ross, Gordon, and Bagnell, DAgger](https://proceedings.mlr.press/v15/ross11a.html)

### Cause 4: the 2v2 task barely tests defense, and the reward hides who made a mistake

**Problem.**

- None of the red policies (static, patrol, or chase) ever attacks. A blue defender meets its constraint by holding.
- Against chase, both red boats can pursue the same attacker. The strict stay-back rule rules out a decoy or a second attacker in that situation.
- Before a pickup, the shaping potential uses whichever blue boat is closest to the enemy flag. It does not credit approach progress to a specific role.
- After the role mission becomes impossible, its automaton can charge −1 per native step. One early violation can outweigh every later event reward for both boats.
- Both blue boats share one reward. A defender error and an attacker error produce the same return for both boats.
- Agent identity is already in the shared actor's observation. Separate actors might help the roles specialize, but this is not measured yet.

**Next test: one role at a time.**

- **Attacker:** train with the guard frozen. Measure pickup and return.
- **Defender:** train against a real scripted attacker. Measure conceded pickups, conceded captures, and legal intercepts.
- Give each role a scripted baseline.
- Compare a fixed-role pair with a two-attacker or decoy pair against the same red lineups.
- Once red can score, add opponent capture costs or a score differential. Audit collisions and boundary outcomes separately.
- After each role works on its own, compare three setups: a shared actor with explicit roles, two separate actors, and a centralized critic.

A centralized critic sees the joint state during training, while each actor uses only its own observation. Our shared actor already sees the full boat state. To make the critic comparison meaningful, define the critic's state input clearly or use a role-specific value baseline.

Sources: [role predicates](../src/maritime_nsrl/team_env.py#L135-L161) · [team observations](../src/maritime_nsrl/team_env.py#L290-L332) · [red targeting](../src/maritime_nsrl/team_env.py#L377-L415) · [team potential](../src/maritime_nsrl/team_env.py#L418-L428) · [shared reward and rollout lanes](../src/maritime_nsrl/team_env.py#L705-L745) · [Yu et al., MAPPO](https://arxiv.org/abs/2103.01955)

## 3. What to borrow from other PPO pipelines

| Reference | What to take from it |
| --- | --- |
| [PPO paper](https://arxiv.org/abs/1707.06347), [SB3 PPO 2.7.0](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/ppo.html) | Pin the version. Count rollouts and updates. Log approximate KL, clip fraction, entropy, value loss, and explained variance next to game events. PPO learns only from fresh rollouts, so it can learn to capture only if the rollouts contain captures. |
| [Engstrom et al.](https://arxiv.org/abs/2005.12729), [Andrychowicz et al.](https://arxiv.org/abs/2006.05990), [CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/) | Audit normalization, advantages, initialization, rollout length, minibatches, and clipping against a readable reference. Change one setting at a time. These implementation details changed results a lot in their benchmarks. |
| [Maritime 2v2 field study](https://arxiv.org/abs/2404.17038), [MCTF'26 PPO example](https://github.com/lmckane/MCTF2026/blob/master/examples/train.py) | Use the field study's rule-based options as baseline ideas. The study found that low-level policies played conservatively and that behavior-based coordination did better. The MCTF example is a reference for PyQuaticus integration and rollout accounting. It uses a different simulator branch, 10 workers, 6,000 world transitions per rollout, and 15 million world steps. |
| [MAPPO code](https://github.com/marlbenchmark/on-policy), [RLlib multi-agent](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html) | Compare local and centralized value estimates with the same actor inputs. Use explicit policy mapping and frozen policies. MAPPO's ablations of value inputs, normalization, epochs, and clipping transfer better than its benchmark win rates. |
| [DeepMind Quake CTF](https://arxiv.org/abs/1807.01281), [OpenAI Five](https://arxiv.org/abs/1912.06680) | After the agents can beat the scripts, train against a mix of frozen opponents and earlier snapshots, and report cross-play on held-out opponents and teammates. Both projects combined many techniques at very large scale, so neither isolates a single one. |

**The 2024 maritime CTF competition report favors a dedicated return controller.** One decentralized RLlib PPO entry trained for 17,000 episodes per agent and finished with 5 grabs and 0 captures (scored without collision penalties). Another team used QMIX for general play and switched to a league-trained PPO policy after a pickup, and it recorded 15 captures. The competition was 3v3 with different scoring. Its results support testing a dedicated return controller and a collision-aware scoreboard. They say little about how large a training budget to use. ([Dasgupta et al., Tables 3–4](https://openreview.net/pdf/fb0192e3a03bffead074253dc40624ed8d8cbf97.pdf))

**Choosing a framework:**

- Start with SB3 and one trainable actor, for example an attacker paired with a scripted defender. This needs a Gym wrapper that exposes one learner inside the joint arena.
- Keep the existing shared-team wrapper as the comparison for policy sharing.
- Move to RLlib or the MAPPO reference when you need several trainable role policies at once or a centralized critic. The 3v3 competition team will probably need this. The [upstream PyQuaticus 2v2 RLlib example](https://github.com/mit-ll-trusted-autonomy/pyquaticus/blob/72b50e067ab311929390ecd4e59131452be15c6d/rl_test/train_2v2.py#L71-L127) shows PettingZoo wrapping, separate blue policy IDs, rollout workers, and checkpointing.
- Keep the same PyQuaticus game rules in every comparison.

### Architecture experiments

Try these in order, and only after the pipeline produces useful attack experience.

| Model | Question | How to test it |
| --- | --- | --- |
| Flat `MlpPolicy` (control) | Can a plain feedforward policy learn once it sees enough pickups and returns? | Start with the existing networks: 64×64 legacy or 128×128 team. On the new action interface, compare the current features with agent-relative bearings, relative velocities, flag and midline distances, cooldown, and time remaining. Test feature changes separately from capacity changes. |
| `MultiInputPolicy`, default extractor | Does a named observation schema help organize and validate the data? | Group the inputs into a Gymnasium `Dict` with self, teammate, opponent, arena, and mission entries. SB3 flattens non-image inputs, so the model capacity is about the same. Use this only as a parity check if you need the schema. |
| `MultiInputPolicy`, entity encoder | Does encoding every boat with shared weights help when opponent order or team size changes? | Use [Deep Sets](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)-style pooling over teammates and opponents with masks, then add self and mission features. Compare it with a flat MLP that gets the same inputs. Test mirrored starts and swapped opponent slots. With only two boats, 1v1 gains little from this model. |
| Frame-stacked `MlpPolicy` | Does recent motion, or red's hidden patrol phase, matter? | Stack the last 3–4 decisions, optionally with the previous action. Reset the stack at episode boundaries. Keep the same span in simulated seconds when repeat changes. Compare it with a single-frame model that has explicit motion features. |
| `RecurrentPPO`, `MlpLstmPolicy` | Does memory help with feint timing, recovery, or an opponent whose mode is hidden? | Use one small LSTM with 64 or 128 units on the flat features. Pin `sb3-contrib`. Keep the hidden state per arena and per actor, and pass episode-start masks during training, evaluation, and viewing. Compare it with the stacked MLP at equal simulation budget, and report wall time. |

**About memory.** The observation already includes headings, speed, cooldown, and mission phase. A hidden patrol waypoint, controller history, or a mixed opponent could still leave the observation incomplete. To test whether memory helps, first compare runs with and without an explicit opponent-mode or controller-state feature. Then compare frame stacking and the LSTM on the version without that feature. An LSTM evaluation must keep the hidden state across decisions and clear it on reset. SB3-Contrib implements recurrent PPO and action-masked PPO as separate classes, so you cannot use both features in one model.

Sources: [Ni et al., recurrent model-free RL](https://proceedings.mlr.press/v162/ni22a.html) · [SB3 policy networks](https://stable-baselines3.readthedocs.io/en/v2.7.0/guide/custom_policy.html) · [SB3-Contrib RecurrentPPO](https://sb3-contrib.readthedocs.io/en/v2.7.0/modules/ppo_recurrent.html) · [SB3-Contrib MaskablePPO](https://sb3-contrib.readthedocs.io/en/v2.7.0/modules/ppo_mask.html)

## 4. How much training to run

### Count experience consistently

Report five counts: **agent transitions, joint decisions, native simulator steps, completed episodes, and PPO updates.** For example, the 5,000-episode 2v2 run collected:

- 480,000 agent transitions from 240,000 joint decisions
- about 1.2 million native steps
- about 936 completed PPO rollouts across both stages

The two transitions in a joint decision come from one game state and share its events and rewards. Treat them as two views of one sample, not as two independent samples.

Two counting problems in the current code:

- The episode-stop callback cuts off the last partial rollout.
- The configured upper bound from episodes to steps assumes a 0.1-second step. With simulator speedup, the real step is about 1 second.

Compare runs by measured counts. In the new runner, record both collected and optimized transitions. On resume, SB3 restores the PPO optimizer, but the arena, RNG state, and rollout collection start over.

### Budget curve

- Evaluate each arm at 50k, 150k, and 500k joint decisions. Run the 500k tier only for arms that show progress on pickup or return. At repeat 5, 500k joint decisions is up to 2.5 million native steps.
- Screen with 3 training seeds. Confirm the two best arms with at least 5.
- When repeat changes, match native steps as the main budget. Also report decisions and wall time.
- **Scale up** if pickups rise and captures after a pickup also rise. **Do not scale up** if pickups stay flat and neither flag distance nor tag margin improves. In that case, revisit the action interface or the start distribution.

### Throughput

- Benchmark 1, 4, and 8 parallel arenas for simulator throughput.
- For the first 1v1 comparison, try 1×512 against 4×128 transitions per batch. The sample count per update is the same, but shorter fragments change bootstrapping. Report this comparison separately and inspect how episode boundaries are handled.
- When the rollout mix is stable, a candidate setup is 8 arenas × 256 decisions with minibatches of 128 or 256.
- In 2v2, group each pair of lanes by arena and keep both agents in step.
- Profile simulator time, Python time, and learner time before you change hardware.

### Hyperparameters

- Reference settings: LR `3e-4`, 5 epochs, clip `0.2`, entropy coefficient `0.01`.
- Try LR `1e-4` or a smaller update target if KL and clip fraction show large updates, or if performance on the easy task drops.
- Track action entropy by role and stage before you change the entropy coefficient.
- The inputs are already scaled. If you add `VecNormalize`, compare training statistics with frozen evaluation statistics. Keep reward normalization separate from observation scaling.

### Discounting and time limits

- If the decision interval changes, retune `gamma` in physical time. A `gamma` of 0.99 per 5 seconds is about 0.998 per second.
- Make the shaping potential use the same per-decision discount as the learner. Decide how to sum reward within a repeated action and how to handle early termination.
- If 240 seconds is the task deadline, expose the remaining time and end the episode there. If the cutoff only limits data collection, bootstrap from the final observation.
- Check both wrappers and every vectorized adapter for this before you compare them.

Sources: [team rollout wrapper](../src/maritime_nsrl/team_env.py#L716-L745) · [episode budget and training](../src/maritime_nsrl/experiments.py#L398-L482) · [SB3 vectorized environments](https://stable-baselines3.readthedocs.io/en/v2.7.0/guide/vec_envs.html) · [SB3 PPO parameters](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/ppo.html) · [on-policy ablations](https://arxiv.org/abs/2006.05990) · [Pardo et al., Time Limits in RL](https://proceedings.mlr.press/v80/pardo18a.html) · [legacy deadline handling](../src/maritime_nsrl/single_agent_env.py#L182-L212) · [team terminal observations](../src/maritime_nsrl/team_env.py#L737-L745)

## 5. Experiment plan

**Every training step below runs as a guided and unguided pair.** Both arms share actions, curriculum, start states, demonstrations, architecture, seeds, and budget. They differ only in the symbolic guidance. Each step therefore gives a competition result (the better policy moves on) and a paper result (the gap between the arms in captures and violations).

**Fix the competition rule gaps before the main paper runs.** Match capture and untag geometry, boundary-exit recovery, tagging, and collision counting to [`GAME_RULES.md`](../GAME_RULES.md) early, because these rules change what is feasible. Short first-capture episodes can stay as a training device. Evaluate on full 10-minute matches scored by capture difference. Add Powerplay once its scheduling and agent-selection rules are known.

**Use the basic capture mission first.** Keep the English instruction fixed, and check that its formula matches the simulator events. Test richer missions only after the basic controller is competent. Compare task reward, monitor state, and monitor state plus guidance on the same controller. For the teacher's probe-and-lure mission, score raw captures and mission-qualified captures separately. The gap shows when following the strategy costs performance. A policy trained on one fixed DFA answers a question about one mission only. To train one policy for many instructions, condition the policy on the current goal or the remaining formula, and hold out whole missions ([LTL2Action](https://proceedings.mlr.press/v139/vaezipoor21a.html)).

1. **1v1 feasibility.** Run the script and the bounded planner on the base mission. Compare the current waypoint decisions, faster decisions, and a small maneuver set.
   - Use 20 paired starts per speed and interface. Count valid pickups and returns separately. Replay any winning route from scratch to confirm it.
   - Pick the smallest action set that captures repeatedly at the target speed.
   - If the planner reaches pickup but not return, work on the return leg in isolation.
   - If nothing reaches pickup at the target speed, record the gap. Train at the highest speed that works reproducibly, and keep developing attack controls.
2. **1v1 curriculum.** Use a flat MLP on the chosen interface.
   - Compare direct training at the target speed with the mixed-start, mixed-speed curriculum. Match native steps and seeds.
   - Provisional promotion rule: at least 12/20 captures on normal development starts at two checkpoints in a row. The policy must also stay within 2/20 of its earlier-level score on the same starts.
   - Use fresh seeds for final claims.
3. **Reward signal and initialization.** Use the best start mix from step 2.
   - Compare task reward only, task reward plus potential shaping, and task reward plus potential shaping plus milestones.
   - If a teacher works at the target speed, compare behavior cloning, PPO from scratch, and cloning followed by PPO.
   - Keep the actions, start pool, and scoring the same.
4. **Architecture.** Compare a flat MLP, a frame-stacked MLP, and a small LSTM.
   - Keep the reward, actions, held-out cases, and curriculum schedule the same. Freeze or replay the schedule so that promotion timing does not differ between arms.
   - Test the entity encoder separately on 2v2, layout transfer, or team-size transfer.
   - Report capture gains per simulated step and per hour.
5. **2v2 attacker.** Freeze the blue guard and train the attacker.
   - Use the hold-and-attack script as the smoke test. Require repeated static and patrol captures before promotion.
   - In a separate arm, pair the attacker with a scripted decoy teammate. Test whether drawing pursuit, or using a defender's tag cooldown, opens a pickup window for the second attacker.
   - Label whether the fixed-role LTL mission applies, and evaluate the learned policies against both setups.
6. **2v2 defense and coordination.** Add a scripted red attacker and train defense on its own. Then pair the two learned roles.
   - Using the same match seeds and joint-decision budget, compare four setups: a scripted pair, fixed learned roles, a shared role-conditioned PPO, and separate actors.
   - Try a centralized critic if joint value estimates look weak.
   - Add historical opponents once the team wins against more than one opponent. Sample the opponent at reset and keep it fixed for the whole match.
7. **3v3 competition matches.** Move the role controllers to 3v3 under the aligned rules.
   - Play full 10-minute matches against red teams that both attack and defend at full speed.
   - Score by capture difference, then grabs, then collisions, as the competition does. Track Powerplay use once it is implemented.
   - Train against the opponent pool below, and keep scripted opponents in it.
   - Run the guided and unguided comparison here too. The 3v3 gap is the strongest paper result, because the competition game is the setting the paper claims to cover.

**Opponent pool.** A starting mix is 50% scripted opponents, 30% recent frozen snapshots, and 20% older snapshots. Tune the mix on development cross-play. Report a results matrix against unseen scripts and checkpoints, from both sides where the code supports it. Keep scripted opponents in the pool to catch policies that become vulnerable to simple attacks.

### Milestones

- **First implementation milestone: a feasibility runner.** It needs a pinned environment, a switch for the action interval, and a replayable result per seed. Build it on `TeamArena` so one task contract covers 1v1, 2v2, and 3v3. Keep the legacy 1v1 setup as a separate control, because its action and observation meanings differ.
- **First training milestone: a learned attacker that captures.** It must capture against scripted patrol in 2v2 with a frozen guard. It must also capture against at least one chase speed in 1v1 from normal starts.
- **First paper milestone: a guided and unguided comparison.** Use at least 5 training seeds per arm, report captures and violations, and cover at least two missions. Run it at a setting where unguided PPO is partly successful, then repeat it at full speed once full-speed captures work.
- **Capability target:** captures from normal starts against full-speed chase.
- **Competition milestone: a 3v3 team that wins 10-minute matches on capture difference.** It must win under the aligned rules against scripted attacking opponents and against held-out frozen snapshots.

On resume, the standard team CLI keeps the saved arena parameters except the opponent kind. A speed or repeat curriculum therefore needs an explicit runner config and compatibility checks. ([team resume](../src/maritime_nsrl/experiments.py#L369-L407))

### What to measure and how to select

- **Per-episode outcomes.** Record the game outcome, pickup, time to pickup, time left at pickup, tags before and after pickup, and the minimum flag and opponent distances. Also record rule violations, collisions, mission completion, and requested versus executed actions when shielding is on.
- **Capture after pickup.** Compute this rate only over episodes that have a pickup, and show the denominator. If there are no pickups, report the rate as undefined.
- **Seeds and splits.** Use independent training seeds and separate development and test starts, with fixed evaluation cases across arms. Start with 20 development episodes per important condition. If the budget allows, run at least 50 fresh test episodes per confirmed target condition and per trained seed.
- **Uncertainty.** Report per-seed results and the spread across seeds. A Wilson interval for one saved policy describes only its episodes. For claims across seeds, treat each trained policy as one replicate ([Agarwal et al.](https://arxiv.org/abs/2108.13264)). Show rates by opponent and layout before any aggregate.
- **Deterministic and sampled evaluation.** Run deterministic and sampled evaluations as separate, labeled runs. Deterministic is the deployment baseline. Sampled runs show whether the policy sometimes attacks even when its most likely action is to stay home.
- **Checkpoints.** At every development checkpoint, save the easy-opponent retention score and the target-speed score. Run the final test only on the selected policy.
- **Provenance.** Store the simulator and code revisions, environment and reset parameters, seed schedule, reward terms, observation and action schemas, normalization statistics, learning settings, wall time, and checkpoint ancestry. Record why every episode ended, including the deadline, a numerical termination, or a shield deadlock.
- **Two evaluation tasks.** Use short first-capture episodes for diagnostics during development. A red team that never attacks tests offense only. Judge competition readiness on full 10-minute matches under the aligned rules, against opponents that can score. Report the capture difference, grabs, and collisions within 4 m, which is the competition's ranking order ([research rule alignment](RL_PIPELINE.md#where-the-research-setup-differs-from-competition-play)).

## Appendix: reproduce the runtime checks

Run these snippets from the repository root with the locked environment installed. The 2v2 check needs the two saved local checkpoints. The teacher check needs only the source and the simulator.

### Reproduce the 2v2 check

```bash
MPLCONFIGDIR=/tmp/mctf-mpl XDG_CACHE_HOME=/tmp/mctf-cache .venv/bin/python - <<'PY'
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from stable_baselines3 import PPO

from maritime_nsrl.team_env import ArenaConfig, TeamArena

root = Path("runs/team-2v2-role-curriculum-5000")
config = ArenaConfig(**json.loads((root / "chase-stage/arena_config.json").read_text()))
policies = {
    stage: PPO.load(root / stage / "policy.zip", device="cpu")
    for stage in ("patrol-stage", "chase-stage")
}

for opponent in ("static", "patrol", "chase"):
    for method in (*policies, "hold/direct"):
        arena = TeamArena(replace(config, opponent=opponent))
        totals = Counter()
        try:
            for seed in (31000, 31001, 31002):
                observation = arena.reset(seed=seed)
                done = False
                while not done:
                    if method in policies:
                        action, _ = policies[method].predict(observation, deterministic=True)
                    else:
                        carrying = arena.game.state["agent_has_flag"][1]
                        action = [7, 1 if carrying else 0]
                    observation, _, done, info = arena.step(action)
                metrics = info["metrics"]
                totals.update(
                    captures=metrics["captures"],
                    pickups=metrics["pickups"],
                    tags=metrics["tags_received"],
                    mission=int(metrics["ltlf_satisfied"]),
                    crossed=int(metrics["reached_enemy_territory"]),
                )
        finally:
            arena.close()
        print(opponent, method, dict(totals), flush=True)
PY
```

### Reproduce the 1v1 teacher check

```bash
MPLCONFIGDIR=/tmp/mctf-mpl XDG_CACHE_HOME=/tmp/mctf-cache .venv/bin/python - <<'PY'
from collections import Counter

from maritime_nsrl.probe_lure import FORMULA, ProbeLureTeacher
from maritime_nsrl.team_env import ArenaConfig, TeamArena

for speed in (0.6, 0.8, 1.0):
    arena = TeamArena(ArenaConfig(formula=FORMULA, probe_action=True, opponent_speed=speed))
    totals = Counter()
    phases = Counter()
    try:
        for seed in range(10):
            arena.reset(seed=seed)
            teacher = ProbeLureTeacher()
            done = False
            while not done:
                _, _, done, info = arena.step(teacher(arena))
            metrics = info["metrics"]
            totals.update(
                pickups=metrics["pickups"],
                captures=metrics["captures"],
                qualified=int(bool(metrics["captures"]) and metrics["ltlf_satisfied"]),
            )
            phases[metrics["furthest_phase"]] += 1
    finally:
        arena.close()
    print(speed, dict(totals), dict(phases), flush=True)
PY
```
