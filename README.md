# Maritime neuro-symbolic reinforcement learning

The current [LaTeX manuscript](paper/manuscript.tex) distinguishes the original deployment-only shielding proposal from the implemented training-time guidance. The separate [runtime-only screen](experiments/runtime-shield-20261012-v1/ANALYSIS.md) evaluates the same two saved neural policies with the bounds gate disabled/enabled: 360 episodes, no captures or pickups, and no substitutions in 43,200 shield checks. All 180 pairs had identical recorded trajectories. This exploratory result tests a dormant gate and establishes no safety or offensive advantage.

Reproduce this screen into a fresh directory:

```powershell
.venv/Scripts/python.exe scripts/run_runtime_shield_study.py --output experiments/runtime-shield-new --episodes 10 --training-seeds 0 1
.venv/Scripts/python.exe scripts/analyze_runtime_shield_study.py experiments/runtime-shield-new
```

New additive tools: [translator benchmark](TRANSLATOR_BENCHMARK.md) compares the
existing translation/repair adapter with a pinned, CTF-adapted LTLCodeGen
translation-only adapter using 40 reference-labelled statements and DFA semantic
equivalence. [Watch/replay](VIEWER.md) displays saved policies or the oracle through
PyQuaticus's renderer, with formal-state overlays and optional PNG/JSONL recording.
Both write only to fresh output directories. The first benchmark's live calls were
blocked by socket permissions; no translator quality ranking is claimed.

## Phase 3: offensive recovery

The completed five-seed matched suite is preserved unchanged in
`research_archive/matched-ablation-20260918-v1`. Phase 3 uses a separately hashed source snapshot
in `research_archive/offensive-phase-20260924-v1`. It adds offensive-funnel metrics (enemy
territory, flag vicinity, pickup, return across midfield, capture), per-stage times,
action histograms, shield substitutions, intervention context, and optional JSONL
trajectory traces.

The first exploratory curriculum intentionally starts small: two seeds and 30,720
transitions in 1v1 static. A variant advances to patrol, chase, and then 3v3 stages
only after reaching a 10% held-out capture rate. This phase tests neural,
temporal-state-only, and softened neuro-symbolic guidance before considering
expensive confirmatory runs. See the phase directory's `DIAGNOSIS.md`, oracle raw
episodes, traces, protocol, plots, and exploratory run directories.

## Chase-recovery diagnostic

The completed Phase 3 confirmatory chase stage reached enemy territory in every
held-out episode but produced no flag pickups. The additive chase-recovery study
therefore keeps the successful softened, no-tag-penalty specification fixed and
uses a small, two-seed 1v1 chase diagnostic before any new five-seed study. It
compares the current configuration with dense pre-pickup flag-progress shaping and
with combined flag-progress, defender-evasion shaping, and a normalized nearest-
defender distance observation. The waypoint action set is deliberately unchanged
in this first test, so changes can be attributed to feedback rather than action
abstraction.

Run it from a fresh directory beside the existing results:

```powershell
.venv/Scripts/python.exe scripts/run_chase_recovery.py initialize --root results/chase-recovery-reproduction
.venv/Scripts/python.exe scripts/run_chase_recovery.py run --root results/chase-recovery-reproduction
```

Each run saves its frozen source snapshot, protocol, exact arena/PPO configuration,
raw training metrics, held-out evaluation CSV, trajectory JSONL, and selected
replay records. The pre-registered screen recommends, but does not automatically
launch, a five-seed comparison only when a non-baseline candidate has at least a
10% pooled pickup rate and reduces mean minimum enemy-flag distance by at least
10% versus the current no-tag-penalty baseline. This exploratory study does not
make a superiority claim.

## Specification provenance and role composition

`results/phase2-role-spec.json` now explicitly separates the original translated
mission from human-added no-tag and defender constraints. The complete original
LLM response, instruction, model, and validation remain under `mission_translation`.
The translated mission retains both `G(in_bounds)` and the ordered objective
`F(has_flag & F(captured))`; no new standalone mission is attributed to the LLM.
`combined_ltl`, its compatibility alias `ltl`, and the fresh validation report
refer to exactly the same formula. Reproduce it with:

```powershell
mctf compose --mission results/live-translation.json --constraint 'G(!tagged)' --constraint 'G(defender_behind_center)' --output results/phase2-role-spec.json
mctf validate results/phase2-role-spec.json
```

Training and evaluation reject conflicting provenance and independently revalidate
the formula. `--role-constraint` composes these human constraints idempotently and
saves the resulting artifact with the policy. It never replaces the mission.
These checks establish internal consistency and logical satisfiability, not
English semantic equivalence or physical realizability. They do not repair old
checkpoint metadata retroactively.

All five ablation commands, including shield-only, are in
[PHASE2_COMPARISON.md](PHASE2_COMPARISON.md). The CLI shield enforces arena bounds
only; role and no-tag constraints are monitored, not added to its hard gate.
The historical run labelled neural used guidance defaults and is not a valid
purely neural baseline. Corrected experiments must be rerun before comparison.

A working Python proof of concept: English strategy → validated finite-trace LTL → reward machine and shaping → Stable-Baselines3 PPO → real 1v1 PyQuaticus → recorded evaluation. No LLM is called during gameplay. This is a small experimental platform, not a competition-ready policy or a verified autonomous vessel controller.

## Quick start

Requires Git and [uv](https://docs.astral.sh/uv/). Run commands from this directory. The bootstrap creates a project-local Python 3.10 environment and installs a pinned simulator checkout and dependency lock. Windows is the tested platform; the simulator-only install excludes the unused native MOOS bridge on all platforms.

```powershell
python scripts/bootstrap.py
.venv/Scripts/Activate.ps1
python -m pytest -q
mctf experiment --strategy strategies/capture.txt --output runs/demo --steps 4096 --episodes 5
```

On Linux/macOS, replace activation with `source .venv/bin/activate`. Those platforms have not been exercised in this delivery. If activation is unavailable, use `.venv/Scripts/python.exe -m maritime_nsrl.cli` in place of `mctf` on Windows.

The bootstrap requires network access on the first run. It intentionally installs PyQuaticus separately with `--no-deps`, after installing its simulation dependencies from the lock, to exclude `pymoos`. Therefore a package dependency audit may report the unused MOOS dependency missing on Linux/macOS. Real-vehicle MOOS integration is outside this project.

`results/extended/` contains the main matched comparison, `results/pilot/` preserves the initial short run, and `results/curriculum/` explores mixed initial states. These include actual training logs, saved policies, and paired evaluation records. See [RESULTS.md](RESULTS.md) for measured outcomes and limitations. New experiment directories must not already exist, to protect previous results.

## Separate pipeline stages

```powershell
mctf translate strategies/capture.txt --output runs/capture-spec.json
mctf validate runs/capture-spec.json
mctf train --spec runs/capture-spec.json --output runs/baseline --steps 16384 --seed 0
mctf train --spec runs/capture-spec.json --output runs/guided --steps 16384 --seed 0 --guided
mctf evaluate runs/baseline/policy.zip --output runs/baseline/evaluation.csv --episodes 20
mctf evaluate runs/guided/policy.zip --output runs/guided/evaluation.csv --episodes 20
mctf plot runs

# Three training seeds, held-out opponents
mctf matrix --spec runs/capture-spec.json --output runs/chase-seeds --seeds 0 1 2 --steps 8192 --guided

# Shared-policy 2v2 matrix with predictive shielding and three maps
mctf matrix --spec runs/capture-spec.json --output runs/team-seeds --team-size 2 --shield --seeds 0 1 --steps 8192
```

For multiple independently trained policies:

```powershell
mctf experiment --output runs/multiseed --steps 65536 --seeds 0 1 2 --episodes 30
```

`matrix` is the compact research harness: single-agent mode trains each requested seed against the chase opponent, then evaluates static/patrol/chase; team mode trains a shared PPO policy and evaluates all three opponents on standard, wide, and narrow arenas. Every seed gets its own immutable directory and a combined `matrix.json`.

PPO collects complete batches of 512 decisions, so a nonmultiple requested budget rounds up. `training_complete.json` records the actual count. One decision holds a selected waypoint for up to five simulated seconds. A complete episode has a 240-second deadline, or ends on the first capture. Runs are CPU-only and set Torch to one thread.

## English translation

The default `offline` backend is an explicit **controlled-English substitute**, not an LLM. It maps complete supported sentences to formulas and rejects unrecognized text. It never pretends arbitrary English has been understood. Examples:

| Sentence | Formula |
|---|---|
| Always stay within the arena. | `G(in_bounds)` |
| Never get tagged. | `G(!tagged)` |
| Eventually capture the enemy flag. | `F(captured)` |
| Pick up the enemy flag and then capture it. | `F(has_flag & F(captured))` |
| Eventually return home. | `F(home)` |

Combine safety sentences with one mission sentence. Three example strategy files are provided.

The `llm` adapter accepts an OpenAI-compatible Chat Completions endpoint, including local servers. Choose a model your endpoint provides:

```powershell
$env:LTL_API_KEY = "your-key"  # omit for local endpoints that need no key
mctf translate strategies/cautious.txt --backend llm --model YOUR_MODEL --base-url https://api.openai.com/v1 --output runs/llm-spec.json
```

The endpoint receives the strategy and proposition definitions, returns JSON containing LTL, and gets at most two repair attempts after a validation failure. All successful/failed model responses in a successful translation are retained with the final artifact. HTTP errors propagate; the adapter never silently switches to offline translation. It parses data only and executes no model-generated code. A live `gpt-4o-mini-2024-07-18` call was also tested successfully: [recorded translation](results/live-translation.json). Its formula is identical to the offline capture example. The `live-smoke` run consumes that saved LLM output, and the guided curriculum experiment uses it directly. Repair/error handling is covered separately with mocked responses. The [official model documentation](https://developers.openai.com/api/docs/models/gpt-4o-mini) lists Chat Completions support for this model.

Review the generated formula against the English intent: syntax and satisfiability do not establish correct natural-language interpretation. Training reloads and revalidates the formula even if its saved report says it was validated.

## Formal contract and its limits

The original reward-machine compiler recognizes standard syntax but deliberately accepts a smaller fragment:

```
contract := G(boolean_predicate) & ... & F(p & F(q & F(r ...)))
```

Either part is optional; at most one ordered eventuality chain is allowed. For broader specifications, `logic_backend=dfa` uses `temporal.py`, supporting arbitrary nesting of `!`, `&`, `|`, `->`, strong `X`, strong `U`, `F`, and `G`. It constructs a reachable finite-trace DFA and rejects unsatisfiable formulas or resource-limit cases explicitly. Both paths use **LTLf over nonempty finite episodes**; eventualities must finish before the deadline.

Validation checks grammar, known atomic propositions, supported compilation, and exact satisfiability **over the independent Boolean proposition alphabet**. It enumerates 32 valuations, filters them by all safety predicates, and finds a compatible valuation for every ordered goal. This is sufficient and necessary for this fragment and produces a witness trace. It does not prove the witness physically realizable in PyQuaticus; for example, physically conflicting flags may be logically independent in the alphabet. It also does not prove infinite-horizon LTL validity (truth on every possible trace).

The compiled monitor is a deterministic machine with state `(mission_stage, safety_ever_violated)`. It evaluates the initial state and every one-second simulator transition. Safety failure is absorbing; a later capture cannot erase it. Mission stages can still advance after a safety failure for diagnostic reporting, while `ltlf_satisfied` remains false. A missed eventuality counts as mission failure at the deadline, separately from safety-violation steps.

| Proposition | Grounding in PyQuaticus |
|---|---|
| `in_bounds` | Simulator's agent OOB flag is false (boat center strictly inside arena). |
| `tagged` | Agent currently in tagged state, including OOB auto-tagging. |
| `has_flag` | Agent currently carries opposing flag. |
| `captured` | Team capture counter increased on this transition; an event, not a permanent label. |
| `home` | Agent lies within `catch_radius` of its own flag home. |

This pinned PyQuaticus version scores a flag when the carrier returns to its **own territory**. Reaching the flag stand is not required. The return waypoint is the own flag home, so its route passes through the scoring boundary. These definitions are kept in `guidance.py`.

## RL integration

PPO controls seven discrete navigation primitives: enemy flag, own flag, two midfield lanes, two enemy-side lanes, and stop. A deterministic low-level controller translates the selected waypoint into speed and relative heading. It does not decide when to attack, return, or evade; PPO chooses and can change the waypoint every five seconds. This shared action abstraction makes a short vertical slice learnable but reduces the task compared with learning raw helm control. Both agents use it unchanged, along with the real Heron vessel dynamics, 160×80 arena, 10-meter tagging radius, and the simulator's automatic return-home behavior after tagging.

The observation contains both agents' normalized positions, heading sine/cosine, speeds, flag/tag status, tagging cooldowns, and remaining task-time information. Guided PPO additionally receives mission stage and the absorbing safety-failure bit; the baseline receives zeros in those slots. This is a full-state simulation experiment, not a partial-observation or deployment-ready sensor policy. It compares the complete guidance package, not an isolated reward-only ablation.

Common task reward for both agents, per one-second simulator transition:

```
20 × captures + 2 × flag pickups − 2 × entries into tagged state − 0.01
```

The adapter defines this common reward explicitly; it does not use PyQuaticus's default sparse reward. Guided PPO adds, per five-second decision:

```
5 × (gamma × Phi(next) − Phi(current))
  + 2 × new mission milestones
  − 1 × safety-violating simulator steps
```

`gamma=0.99` matches PPO. `Phi` uses the mission stage and normalized distance to the active goal. Capturing while not carrying a flag targets the enemy flag; carrying targets home. Potential is zero at both score endings and the finite deadline, preventing terminal reward leakage. The discounted potential terms telescope; milestone bonuses and safety penalties intentionally change the objective, so **the whole reward is not claimed policy invariant**. The mission memory and time are observed by the guided policy.

The 240-second horizon is a task deadline, not an artificial rollout cutoff. Although PyQuaticus reports a time-limit truncation, the facade reports a terminal finite-horizon task to SB3, avoiding value bootstrapping beyond the deadline, and preserves `environment_time_limit` in `info`.

The extended `TeamArena` also offers a model-based predictive shield. It deep-copies PyQuaticus, tests the requested joint action and bounded alternatives, and commits only a successor whose DFA state remains in the live region. It records interventions, predictions, and deadlocks. This is a hard gate with respect to the copied one-step simulator model; it is not a robust game-theoretic winning-region proof, does not cover continuous disturbances or substep events, and can deadlock when no candidate is safe. The original `MaritimeEnv` remains reward-only for backward compatibility.

## Opponents and curriculum

`static` remains stationary away from the direct flag route, though simulator tagging still applies. `patrol` follows two defensive waypoints at 65% maximum speed. `chase` uses that patrol until the learner enters enemy territory, then pursues at maximum speed. These are deterministic defensive opponents, not attackers or full-game balanced policies.

Train separately with `--opponent patrol` or `--opponent chase`. A practical curriculum uses the previous policy as initialization:

```powershell
mctf train --spec runs/capture-spec.json --output runs/patrol --guided --opponent patrol --resume runs/guided/policy.zip --steps 16384
mctf train --spec runs/capture-spec.json --output runs/chase --guided --opponent chase --resume runs/patrol/policy.zip --steps 16384
```

Resume rejects incompatible formula, guidance, discount, decision duration, dynamics, or horizon. It retains policy and optimizer weights but starts a new environment and logging run; it is curriculum fine-tuning, not a bit-for-bit interrupted-run restore. Run the same curriculum and budgets for the baseline when comparing it. Evaluation always covers all three opponent types with paired held-out episode seeds.

An independent **initial-state curriculum** helps learn the return leg: `--return-start-probability 0.5` starts half of training episodes in enemy territory already carrying the flag. Both methods get the same distribution; the simulator initializes the flag state, the formal monitor observes the initial pickup condition, and no artificial pickup reward is issued. Training rows label these starts. Evaluation always uses the ordinary start with no flag, so curriculum training successes must not be treated as full-game evaluation captures.

```powershell
mctf experiment --output runs/curriculum --steps 8192 --seeds 0 --episodes 20 --return-start-probability 0.5
```

## Metrics and experiment files

Each policy directory records:

- `specification.json`: instruction, backend/model provenance, formula, validation witness.
- `config.json`: seed, environment/training configuration, simulator commit, package versions.
- `policy.zip`: saved PPO network and optimizer.
- `episodes.jsonl`: training reward, common reward, captures, pickups, time, tags, violations, task satisfaction.
- `progress.csv`: PPO losses, entropy, explained variance, throughput, and episodic progress.
- `evaluation.csv`: one row per opponent/held-out seed, including common and guided rewards separately.
- `evaluation.summary.json`: capture rate, successful-capture time, violation/tag means, and episode-level Wilson intervals.
- `evaluation.traces.json` and `evaluation.trajectories.png`: first evaluation episode per opponent, including selected actions, positions, flags, and monitor stage.

Compare **common task reward**, capture rate, and safety metrics across methods; shaped returns are different objectives. Capture time is null for failure and averaged only over successes. `tags_received` counts opposing tags from simulator counters; `tagged_events` also includes OOB-induced tagging. `rule_violation_steps` counts sampled states violating any safety predicate, not distinct violation episodes or the number of clauses violated. `ltlf_satisfied` requires all goals and no safety failure.

The training plot shows a rolling common-reward curve. It provides progress evidence, not a statistical convergence claim. Evaluation episodes vary the learner's initial vertical position; they do not cover different maps or broad initial-state distributions. Episode-level confidence intervals are conditional on a fixed trained policy. Multiple training seeds are needed for robust algorithm comparisons.

## Project map and extension points

```
src/maritime_nsrl/
  translation.py   English backends, endpoint adapter, saved-spec loading
  logic.py         LTL grammar, fragment compiler, satisfiability, monitor
  guidance.py      Simulator labels, monitor-driven potential and rewards
  opponents.py     Defensive policies and low-level waypoint controller
  environment.py   Gymnasium facade over the native PettingZoo game
  experiments.py   PPO, checkpoints, metrics, paired evaluation, plots
  cli.py           Runnable pipeline commands
strategies/        Controlled-English examples
scripts/           Pinned environment bootstrap
tests/             Logic semantics, translation repair, real game integration
results/           Recorded experiments
```

`TeamArena` and `SharedTeamVecEnv` provide a working shared-policy 2v2/3v3 multi-agent PPO path. All blue agents receive independent action lanes through one shared PPO policy, take one joint PyQuaticus step, and receive the common team reward. The wrapper preserves terminal observations and reports one team metrics record. It currently uses scripted red opponents and full-state observations; centralized critics, independent policies, communication, and self-play remain future MARL work. New proposition names require grounding, compiler tests, and translation vocabulary updates together.

The adapter also resets pristine Heron PID controllers on each episode: the pinned upstream `BaseUSV.reset()` resets thrust/rudder but retains PID history. A regression test verifies identical trajectories for repeated reset seeds. The simulator may emit benign parallel-line divide-by-zero warnings from its geometry routine; observations and rewards are checked for finiteness in tests. No upstream physics files were edited.

## Resource investigation

- [MCTF installation](https://www.mctf26.com/installation) and [training tutorial](https://www.mctf26.com/training): the competition specifies Python 3.10 and `mctf2026`. This project pins simulator commit `475d670650ec5d49691ba707cd1054c4e45d8053` instead of tracking branch changes.
- [PyQuaticus](https://github.com/mit-ll-trusted-autonomy/pyquaticus/tree/475d670650ec5d49691ba707cd1054c4e45d8053): inspected reset/state/action conversion, capture/tag events, Heron dynamics, and configuration. SB3's Gymnasium wrapper avoids adding the heavier RLlib stack for a single learner.
- [LTLCodeGen project](https://existentialrobotics.org/LTLCodeGen/) and [source](https://github.com/ExistentialRobotics/LTLCodeGen/tree/b5eb6a19769c3b06c5757136bef3bd444533e328): inspected `code_ltl_translator.py`, `ltl_operators.py`, and model setup. Its complete system uses ROS/Gazebo and semantic maps. The translator generates Python operator calls, writes/imports that code, and retries failures. A direct full-stack integration would add unrelated robotics dependencies. This implementation preserves constrained translation and validation/repair but uses declarative JSON and a finite-trace reward machine instead of executable generated Python and a motion planner. It is an independent adapter/substitute, not a reproduction of their experiments.
- [Stable-Baselines3 PPO 2.7](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/ppo.html): supports the discrete action/Box observation interface used here and provides CPU PPO plus serializable checkpoints.

Upstream projects retain their respective licenses. Their source is cloned under `vendor/` by setup, rather than copied into this project's source tree.

## Tactical attacker follow-up

See [player tactics and reproduction commands](TACTICS_AND_ATTACKER.md) and the [new exploratory report](experiments/tactical-attacker-20261012-v1/REPORT.md). Two imitation-initialized PPO attackers enable a same-checkpoint runtime-shield test; scripted 3v3 role strategies and native replay are available. Earlier experiments remain unchanged.

## Offensive and tactical follow-up screens

See [the new studies and replay commands](FOLLOWUP_STUDIES.md), the [exploratory screen](experiments/followup-20261012-v1/REPORT.md), and the [five-seed selected-condition replication](experiments/offense5-20261012-v1/REPORT.md). The original bounds gate was replicated during offensive approach; braking/reserve failures, role conflicts, and stopped chase variants remain recorded.
