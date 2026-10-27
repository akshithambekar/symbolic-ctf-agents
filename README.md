# Symbolic CTF agents

Train PyQuaticus agents with PPO and optional finite-trace temporal logic guidance:

```text
English strategy → validated specification → temporal monitor/rewards → PPO → PyQuaticus
```

The core was imported from `yahya-akshith_rl` at `663e9b0`. It supports 1v1 and
shared-policy 2v2/3v3 experiments. Historical runs, checkpoints, manuscripts,
translator benchmarks, and tactical follow-up studies remain on that branch.

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

```bash
uv run --locked mctf translate strategies/capture.txt --output runs/capture.json
uv run --locked mctf validate runs/capture.json
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

```bash
uv run --locked mctf train-team --spec runs/capture.json --output runs/team-baseline --team-size 3 --no-reward-shaping --no-temporal-state
uv run --locked mctf train-team --spec runs/capture.json --output runs/team-guided --team-size 3
uv run --locked mctf evaluate-team runs/team-guided/policy.zip --output runs/team-guided/evaluation.csv
```

Team evaluation covers three arena geometries and all three defensive opponents.
`--shield` enables the predictive bounds gate; `--role-constraint` explicitly adds
no-tag and designated-defender constraints to the monitored mission. Those role
constraints are not enforced by the bounds gate.

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
| `demos/` | Existing upstream-policy demo; separate uv environment |
| `ltl/` | Earlier LTLCodeGen-based work; independent of the new pipeline |
| [RL contract](docs/RL_PIPELINE.md) | Observations, actions, rewards, semantics, rule differences |
| [Progress](docs/PROGRESS.md) | Imported scope, recorded outcomes, remaining work |
| [Game rules](GAME_RULES.md) | Official competition gameplay summary |
| [References](DOCUMENTS.md) | External resources |

The research environments use full simulator state, waypoint actions, and short
first-capture episodes. Review the RL contract before treating them as competition
agents or changing their observation/action interfaces.
