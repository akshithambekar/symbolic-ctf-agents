# How the RL pipeline works

This guide follows one training run from the game rules to the saved policy. It
explains what the simulator controls, what the learning algorithm sees and does,
how a mission enters training, and what evaluation can establish.

The short version is:

```text
PyQuaticus game → environment wrapper → observations and actions → PPO updates
                                      ↘ mission monitor and optional guidance
```

The wrapper is the connection between the game and PPO. It converts simulator
state into numbers the policy can use, converts policy choices into boat commands,
and turns game events into rewards. The LTL mission can describe desired behavior,
but it does not replace the game's rules or physics.

## Start with the game

PyQuaticus owns the game simulation. It advances boat dynamics and applies rules
for movement, tagging, flag pickup, capture, boundaries, and episode termination.
This repository pins the simulator revision in [`pyproject.toml`](../pyproject.toml).
The wrapper configures a research match around that simulator.

The current wrappers use a 240-second time limit, a one-capture score limit, and
a 30-second tagging cooldown. They set the simulator time step to `tau=0.1` and
the speedup factor to 10, so one simulator step represents one simulated second.
The wrappers stop an episode when a capture ends the match or the time limit is
reached. A time limit is treated as the end of the task, not as a pause in a
continuing game.

These settings define the research task. They do not reproduce every competition
rule. [`GAME_RULES.md`](../GAME_RULES.md) and the rule-alignment section below
record known differences and open questions.

## The wrapper defines the learning problem

PPO needs a repeated interface: at each decision, it receives an observation,
chooses an action, and receives a reward and a signal that the episode ended. The
Gymnasium wrappers implement this interface around PyQuaticus:

- `MaritimeEnv` in [`single_agent_env.py`](../src/maritime_nsrl/single_agent_env.py) wraps
  one controlled boat against one scripted opponent.
- `TeamArena` in [`team_env.py`](../src/maritime_nsrl/team_env.py) wraps
  one, two, or three controlled boats against a same-sized scripted team. A single
  PPO policy is shared by all controlled boats.

The wrapper does not expose camera-like or sensor-limited data. It reads
`game.state`, which includes the position and status of every boat, including the
opponents. The resulting policy has perfect simulator-state access. This is a
research assumption, not evidence that the policy can run from physical sensors.

### Observations describe the current situation

An observation is the numeric input to the policy. The 1v1 wrapper provides 19
values: both boats' positions, headings, speeds, flag possession, tagged status,
and tagging cooldowns, followed by elapsed time and two mission-monitor values.
The mission values are zero when temporal guidance is disabled.

The team wrapper provides eight values per boat, followed by elapsed time, the
identity of the controlled boat taking this turn, whether the team has ever
carried the flag, and a one-hot encoding of the mission automaton state. For a
team of size `N` and an automaton with `S` states, the observation has
`16N + 3 + S` values. The optional defender-distance feature adds one value.

Positions, speed, cooldown, and time are scaled before use. Values are stored as
`float32` and clipped to `[-2, 2]`. Team observations put the acting boat first,
then its teammates, then the opponents. The two wrappers have different
observation layouts, so their saved policies cannot be interchanged.

### Actions are movement choices, not game rules

PPO chooses a discrete movement primitive. The wrapper maps that choice to a
waypoint, then a steering controller turns the waypoint into the simulator's
continuous `[speed, relative_heading]` command. PyQuaticus applies the boat
dynamics. A choice such as “move to the enemy flag” is therefore a policy action;
it does not guarantee that the boat reaches the flag or can legally capture it.

The 1v1 actions target the enemy flag home, own flag home, two midfield flanks,
two forward flanks, or a hold position. The team wrapper uses flag and home
targets, offensive flanks, return lanes, evasion, and hold. A probe-lure
experiment can enable a ninth, probe action. Team targets adapt to the arena and
current positions. Tagged boats use the simulator's recovery controller instead
of the selected movement primitive.

One policy decision is held for up to five native simulator steps. The wrapper
stops repeating it early if the episode ends. Thus, PPO acts at a slower rate than
the simulator updates boat positions.

## Rewards tell PPO what to improve

The simulator reports events and state changes. The wrapper turns those changes
into a scalar reward for PPO. The default common task reward for each native step
is:

```text
20 × own captures + 2 × own pickups − 2 × newly tagged controlled boats − 0.01
```

These terms are event counts between the previous and new state. Starting an
episode already carrying a flag does not count as a pickup. The small time cost
encourages quicker completion. A tag penalty applies when a controlled boat newly
enters the tagged state.

This reward is a training signal, not the game's official scoring system. The
team wrapper can vary its reward coefficients and logs a fixed common reward for
comparing outcomes. Neither default reward includes a penalty for an opponent
capture or a collision. A study that needs balanced offense and defense must
choose and document those incentives.

## LTL adds a description of desired behavior

Linear temporal logic (LTL) expresses conditions over a sequence of states. The
repo's formulas use propositions grounded in simulator state, such as `in_bounds`,
`tagged`, `has_flag`, `captured`, and `home`. For example:

```text
G(in_bounds) & F(has_flag & F(captured))
```

`G(in_bounds)` says the boat or team must stay in bounds at every step.
`F(has_flag & F(captured))` says it must eventually carry the flag and later score.
The finite episode supplies the end of the trace. An eventuality left incomplete
at the deadline is unsatisfied.

The formula does not make a movement decision. A monitor checks which
propositions hold after each simulator step and tracks mission progress. The
environment can use that progress in two ways:

- **Reward guidance** adds progress rewards, milestone bonuses, or safety
  penalties to PPO's task reward.
- **Temporal state** adds the current monitor state to the policy observation,
  so the policy can condition its action on mission progress.

Guidance can change what PPO optimizes. For example, potential-based progress
reward in 1v1 uses `5 × (0.99 × next_potential − current_potential)`, with terminal
potential set to zero. Milestone and safety terms can also apply. Team guidance
uses automaton acceptance and failure terms. The full shaping package has no
established policy-invariance guarantee, so compare common task rewards and actual
captures as well as shaped training returns.

The repo has two formula engines. The 1v1 `fragment` accepts conjunctions of
`G` Boolean safety constraints and one nested `F` goal sequence. The team `dfa`
engine supports `!`, `&`, `|`, `->`, `X`, `U`, `F`, and `G`, and compiles the
formula into a deterministic finite automaton. Strong `X` requires a next state.
The validator checks syntax, proposition names, and satisfiability over possible
Boolean labels. It does not prove that a translation preserves the English intent
or that the game can physically satisfy the formula.

The optional predictive shield is separate from reward guidance. It checks
candidate next states using simulator copies and commits a candidate that meets
its configured safety formula. Its default formula is `G(in_bounds)`. If no safe
candidate exists, the episode ends as a shield deadlock. This one-step check does
not provide a real-world safety guarantee.

## PPO learns a policy through repeated episodes

At each decision, PPO's policy maps the observation to a distribution over
available actions. The environment executes the selected action, returns the next
observation and reward, and reports whether the episode ended. PPO stores these
transitions, estimates which actions led to better outcomes, and updates the
policy. It repeats this cycle across episodes until it reaches the requested
training budget.

The code uses Stable-Baselines3 PPO with an `MlpPolicy`. The 1v1 model has two
hidden layers of 64 units. The shared team model has two hidden layers of 128
units and treats each controlled boat's turn as an agent transition. Requested
budgets are rounded to complete PPO batches: 512 decisions for 1v1 and 256
decisions per controlled boat for team training. The saved `policy.zip` is the
learned policy; the run directory also records configuration, specification,
training logs, and metrics.

The 1v1 and team runs are different learning problems. They use different
observations, action layouts, and PPO batch sizes. Do not compare their raw
training rewards as if they came from the same interface or objective.

## Evaluation measures what the policy does

Evaluation loads a saved policy and runs it in fresh episodes with deterministic
action selection by default. The standard opponent set is static, patrol, and
chase. Evaluation records task outcomes and diagnostics such as captures, pickups,
tags, rule violations, episode duration, and mission satisfaction. It can also
save trajectories for plots or playback.

Use held-out evaluation seeds and report capture rate alongside common task
metrics. A high shaped return alone does not show that the agent captures the
flag. Episode-to-episode variation for one fixed policy also does not measure
variation across independently trained policies; use multiple training seeds for
that question.

The CLI provides the pipeline stages:

```text
mctf translate → mctf validate → mctf train or train-team
               → mctf evaluate or evaluate-team → mctf plot
```

Training and evaluation refuse to overwrite existing output paths. Resume loads
policy weights and optimizer state where supported, then starts fresh environments
and logs. It continues learning from a checkpoint; it does not recreate an
interrupted run exactly. See the [README](../README.md) for complete commands.

## Where the research setup differs from competition play

The wrappers preserve a reproducible research task, while some game rules remain
unmatched to the competition contract. The current follow-up list is:

| Topic | Research behavior | Work still needed |
| --- | --- | --- |
| Capture and recovery | Capture on return to own territory; untag near own flag home | Confirm corner-base geometry and implement it if needed |
| Boundary exit | Auto-tag, then simulator-controlled recovery | Match immediate return to the nearest base |
| Powerplay | Not implemented in the imported wrappers | Obtain scheduling and agent-selection rules, then implement them |
| Duration and scoring | 240 seconds; first capture; offensive reward | Add a ten-minute match and evaluate score difference |
| Tagging | 10 m threshold and 30-second cooldown | Confirm exact threshold and cooldown rules |
| Collisions | Upstream detection uses summed boat radii | Verify event counting against the competition's 4 m criterion |

The [official rules](https://www.mctf26.com/rules) were reviewed on 2026-10-13.
Rule alignment remains open. Treat results as evidence about the configured
research environment until these differences are resolved.
