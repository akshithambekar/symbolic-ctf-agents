# Generative neuro-symbolic AI for multi-agent capture the flag

This repository explores how natural-language strategies can guide reinforcement
learning agents in PyQuaticus Maritime Capture the Flag. It connects English
instructions to validated linear temporal logic (LTL) specifications, then uses
those specifications during PPO training and evaluation.

## Research question and goal

**Research question:** Can validated LTL specifications derived from natural-language
strategies improve rule compliance and capture performance for reinforcement
learning agents, compared with PPO that does not use symbolic guidance?

The proof-of-concept goal is to run the full pipeline from an English instruction
through LTL validation to agent behavior in PyQuaticus. The initial evaluation
focuses on 1v1 capture-the-flag against static, patrolling, and reactive opponents.
The repository also supports shared-policy team experiments with up to three
agents per side, where we can study role constraints and coordination.

The code supports guided and unguided training comparisons, plus team experiments
with optional reward shaping, temporal-state observations, and a predictive safety
shield. These are experimental controls, not evidence that symbolic guidance
improves performance. In particular, reliable captures against a full-speed
reactive opponent remain an open problem. See [research progress](docs/PROGRESS.md)
and the [RL pipeline guide](docs/RL_PIPELINE.md) for scope and known limits.

## Setup

Install Git and [uv](https://docs.astral.sh/uv/), then run from this repository:

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked mctf --help
```

Python 3.10.20 and the simulator revision are pinned. `uv.lock` records resolved
dependencies. The simulation install excludes the unused native MOOS bridge and
retains the Apple Silicon SciPy constraint from our existing demo setup.

## Run a 1v1 experiment

Translate a supported English strategy to LTL and validate the saved
specification:

```bash
uv run --locked mctf translate strategies/capture.txt --output runs/capture.json
uv run --locked mctf validate runs/capture.json
```

The default offline translator accepts only the controlled-English clauses in
`src/maritime_nsrl/translation.py`. For other wording, the CLI can call an
OpenAI-compatible chat-completions endpoint and validate the returned formula.
Set `LTL_API_KEY` and pass a model name. Use `--base-url` for a compatible service
that does not use the default OpenAI endpoint:

```bash
LTL_API_KEY=your-key uv run --locked mctf translate strategies/capture.txt \
  --backend llm --model your-model --output runs/capture-llm.json
```

Validation checks formula syntax, known propositions, and satisfiability over
possible labels. It does not establish that the formula preserves the instruction's
meaning or that the simulator can satisfy it.

Train a baseline and a guided policy from the same specification, then evaluate
both against the standard opponents:

```bash
uv run --locked mctf train --spec runs/capture.json --output runs/baseline --steps 16384
uv run --locked mctf train --spec runs/capture.json --output runs/guided --steps 16384 --guided
uv run --locked mctf evaluate runs/baseline/policy.zip --output runs/baseline/evaluation.csv --episodes 20
uv run --locked mctf evaluate runs/guided/policy.zip --output runs/guided/evaluation.csv --episodes 20
uv run --locked mctf plot runs
```

Evaluation covers static, patrol, and chase opponents with held-out seeds. Compare
captures, common task reward, and violations. Guided returns include extra rewards.
Run these commands with fresh output names. Training and evaluation refuse to
overwrite existing runs. `runs/` is gitignored.

For a patrol curriculum, initialize from a compatible guided checkpoint:

```bash
uv run --locked mctf train --spec runs/capture.json --output runs/patrol --opponent patrol --guided --resume runs/guided/policy.zip
```

Resume retains policy/optimizer weights and starts fresh environments and logs.
It does not restore an interrupted run exactly.

## Run a team experiment

The team wrapper supports 1v1, 2v2, and 3v3 runs with a shared PPO policy. Turn
off reward shaping and temporal-state observations for an unguided comparison.
Enable the optional predictive shield with `--shield`:

```bash
uv run --locked mctf train-team --spec runs/capture.json --output runs/team-baseline --team-size 3 --no-reward-shaping --no-temporal-state
uv run --locked mctf train-team --spec runs/capture.json --output runs/team-guided --team-size 3
uv run --locked mctf evaluate-team runs/team-guided/policy.zip --output runs/team-guided/evaluation.csv
```

Team evaluation covers three arena geometries and all three defensive opponents.
`--shield` enables the predictive bounds gate; `--role-constraint` explicitly adds
no-tag and designated-defender constraints to the monitored mission. Those role
constraints are not enforced by the bounds gate.

Team continuation retains the saved arena settings and PPO state, and starts new
environments and logs. Resume a team checkpoint with `train-team`.

## Inspect behavior

```bash
uv run --locked mctf watch oracle --opponent static --team-size 1
uv run --locked mctf watch runs/guided/policy.zip --opponent patrol
uv run --locked mctf watch oracle --headless --record runs/oracle-replay --max-frames 30
```

The viewer uses PyQuaticus rendering. Desktop viewing needs a graphical session;
headless recording writes frames and replay data into a fresh directory.

## Project map

| Path | Purpose |
| --- | --- |
| `src/maritime_nsrl/` | RL environments, logic, translation, training, evaluation, viewer |
| `tests/` | Logic semantics, provenance, simulator behavior, PPO integration |
| `strategies/` | Supported controlled-English examples |
| `translators/` | English-to-LTL implementations, separate from PPO |
| `benchmarks/` | Translation smoke test and Spot semantic checker |
| [RL contract](docs/RL_PIPELINE.md) | Observations, actions, rewards, semantics, rule differences |
| [Progress](docs/PROGRESS.md) | Imported scope, recorded outcomes, remaining work |
| [Game rules](GAME_RULES.md) | Official competition gameplay summary |
| [References](DOCUMENTS.md) | External resources |

The English-to-LTL translator benchmark is a separate workflow with its own
dependencies and evaluation setup. Follow the [benchmark README](benchmarks/README.md)
to run its smoke test or compare translator implementations.

The research environments use full simulator state, waypoint actions, and short
first-capture episodes. Review the RL contract before treating them as competition
agents or changing their observation/action interfaces.
