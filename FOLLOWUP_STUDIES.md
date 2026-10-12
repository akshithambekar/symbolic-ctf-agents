# Offensive shielding and tactical follow-up

New outputs live under `experiments/followup-20261012-v1` and `experiments/offense5-20261012-v1`. Previous studies remain unchanged. The first directory contains exploratory screens; the second replicates the selected offensive condition across five training seeds. Two of those checkpoints were used in selection, and three are new replicas; all evaluation seeds are fresh.

## Comparisons

- Same saved attacker, gate off versus original gate versus extra brake/turn candidates versus a proactive velocity-reserve gate. Enemy-territory approach and initialized-carrier return states are separate from ordinary full missions. Reserve and emergency controls are opt-in experimental classes; old arena behavior is unchanged.
- Scripted 3v3 fixed, conditional, dynamic and no-defender strategies, with each attack style held constant across role comparisons. These are not trained team coordination.
- Explicit Agent-0-carrier/opponent-intrusion conflicts, with carrier-aware reassignment off/on. Carrier priority is preserved; instantaneous role satisfaction is still not guaranteed.
- Short chase continuation with ordinary task reward, flag/home progress reward, or progress-gated local evasion. Each speed stage advances only after held-out pickups and static capture retention. Stopped runs are transfer-tested at full speed and must not be described as having completed the curriculum.

Read each experiment's `REPORT.md`, raw tables, training curves/logs and traces before drawing conclusions. Deadlock termination is not successful safe control. An intervention while carrying is not inferred merely from a return-start label: the raw event records possession at the intervention.

## Reproduction

Use the existing simulation environment and fresh output names:

```text
python scripts/run_followup.py --output experiments/NEW_SCREEN
python scripts/report_followup.py experiments/NEW_SCREEN
python scripts/export_followup_replays.py experiments/NEW_SCREEN
python scripts/confirm_offensive.py --output experiments/NEW_REPLICATION --screen experiments/NEW_SCREEN
python scripts/report_offense_confirmation.py experiments/NEW_REPLICATION
```

The five-seed runner refuses promotion unless the original gate's offensive screen has nonzero captures and interventions, fewer bounds episodes, no capture loss, and no deadlocks. Its narrow selected condition is frozen before new training/evaluation. Failed recovery variants stay in the screen; they are not discarded from the report.

## Inspect the actual shield event

```text
mctf replay experiments/followup-20261012-v1/replays/approach_tight-off.jsonl
mctf replay experiments/followup-20261012-v1/replays/approach_tight-original.jsonl
mctf replay experiments/followup-20261012-v1/replays/approach_tight-proactive.jsonl
mctf replay experiments/followup-20261012-v1/replays/conflict-fixed-False.jsonl
mctf replay experiments/followup-20261012-v1/replays/conflict-fixed-True.jsonl
```

The native viewer shows recorded physical states rather than resimulating hypothetical physics. Policy/configuration hashes, source snapshots, simulator hashes and raw substituted controls preserve the context of each result. The bounds contract does not protect against tagging or guarantee captures.

The [post-hoc lower-speed reference](experiments/chase-ref-20261012-v1/summary.csv) separates existing easy-chase ability from learning. The chase reward audit reports positive training returns without pickups and does not equate shaped return with task success. For role comparisons, use actual contract satisfaction; the legacy fixed-safety column counts missing defenders even in no-defender mode.

Completed outcomes: the selected original shield prevented bounds violations in 75/75 held-out shielded episodes, retained/increased the observed captures (50 versus 46), and had no deadlocks. Brake candidates were redundant; the proactive reserve aborted useful states. All six chase variants stopped at speed 0.85 and had zero full-speed pickups. New viewer-compatible continuation bundles live in experiments/followup-20261012-v1/viewer_policies/.
