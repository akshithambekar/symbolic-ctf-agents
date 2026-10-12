# Runtime-only bounds shielding: exploratory paired study

The same saved unshielded neural checkpoints were evaluated with and without the existing runtime bounds gate. No policy retraining, temporal observation changes, or reward shaping was introduced. Each arm used the same held-out seeds, maps, opponents, simulator, and horizon.

| Arm | Episodes | Captures | Pickups (%) | Tags (%) | OOB (%) | Common reward | Duration | Deadlocks (%) | Interventions/checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unshielded | 90 | 42 | 66.67 | 53.33 | 0.00 | 6.681 | 187.49 | 0.00 | 0/0 |
| runtime_bounds_shield | 90 | 42 | 66.67 | 53.33 | 0.00 | 6.681 | 187.49 | 0.00 | 0/16874 |

## Interpretation limits

This is a two-training-seed exploratory study, not a superiority test. Evaluation episodes are paired within checkpoints; they are not independent training replicates. The gate enforces predicted bounds safety only. Tags and mission satisfaction remain measured outcomes, not guaranteed properties. A deadlock ends an episode early, reducing exposure to both tags and time penalties; common reward must be interpreted alongside duration and deadlock rate. Rejecting an action near the flag is evidence of intervention context, not proof that a feasible capture was prevented.

See per_seed_summary.csv, paired_deltas.csv, episodes.jsonl, traces.jsonl, inputs/, configs/, and source hashes for the full record.
