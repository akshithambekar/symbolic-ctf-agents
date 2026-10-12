# Player tactics and task-capable shield evaluation

See [the completed exploratory report](experiments/tactical-attacker-20261012-v1/REPORT.md), its raw comparisons and plots. Existing studies are unchanged.

Player tactics are scripted attacker/flank, defender/intercept, carrier/return and support/escort primitives. Fixed, conditional and dynamic defender modes are available. They are not a trained coordination policy, and their monitored contracts are not formal guarantees. Dynamic screening also changes attack style; avoid attributing its results solely to role assignment.

The new neural attacker uses oracle imitation initialization followed by PPO, with no formal training guidance. It remains an eight-action policy, and is distinct from the original scratch-trained baseline. Both seeds passed the held-out static capability screen; chasing opponents remain unsolved.

## Watch and replay

Run from the repository with the configured simulation environment:

```text
mctf watch tactics --config strategies/tactics-3v3-fixed.json --opponent chase
mctf watch tactics --config strategies/tactics-3v3-raiding.json
mctf watch experiments/tactical-attacker-20261012-v1/attackers/seed0/policy.zip --opponent static
mctf replay experiments/tactical-attacker-20261012-v1/tactics/conditional-raid/episode-0.jsonl
```

The native PyQuaticus viewer preserves simulator physics, shows assigned roles, and exports role-labelled traces. Use `--headless --record FRESH_DIRECTORY --max-frames 4` for a viewer check without opening a window. Use `python -m maritime_nsrl.cli` in place of `mctf` when the project is supplied through PYTHONPATH rather than installed.

## Reproduce in fresh directories

```text
python scripts/train_capable_attacker.py --output experiments/NEW/attackers/seed0 --seed 0
python scripts/train_capable_attacker.py --output experiments/NEW/attackers/seed1 --seed 1
python scripts/evaluate_tactics.py --output experiments/NEW/tactics --episodes 3
python scripts/run_runtime_shield_study.py --output experiments/NEW/ordinary --policy-root experiments/NEW/attackers --episodes 5 --evaluation-seed 840000
python scripts/run_runtime_shield_study.py --output experiments/NEW/moving --policy-root experiments/NEW/attackers --scenarios standard --start-modes boundary_lower boundary_upper --boundary-margin 2.01 --boundary-speed 1 --episodes 3 --evaluation-seed 862000
python scripts/analyze_runtime_shield_study.py experiments/NEW/moving
```

The gate checks bounds only. At moderate moving starts it prevented observed violations without losing captures; at higher outward velocities it deadlocked and terminated, which is not useful safe control. Boundary exposure probes are separate from ordinary task evaluations. Neither the controller library nor this two-seed study establishes general superiority or a solution to chase.
