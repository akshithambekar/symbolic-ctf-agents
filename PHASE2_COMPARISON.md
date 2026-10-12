# Phase 2: 3v3 comparison

The Phase 2 architecture is now implemented end to end. `TeamArena` runs a real 3v3 PyQuaticus game, `SharedTeamVecEnv` exposes three correlated action lanes to one shared-policy multi-agent PPO policy, and `train-team` / `evaluate-team` produce saved policies and scenario summaries. This is shared-policy multi-agent PPO, not a claim of centralized critics, independent policies, communication, or self-play.

The corrected role contract for NEW runs is:

```text
G(in_bounds) & G(!tagged) &
G(defender_behind_center) & F(has_flag & F(captured))
```

Agent 0 is the designated defender. Agents 1 and 2 are attackers. The grounded propositions are computed from the actual PyQuaticus state: the defender's x-coordinate relative to the centerline, whether any attacker is forward, whether any teammate is tagged, and team capture events. The LTL DFA and predictive shield are used during both training and evaluation.

The NEW neural baseline commands below use the same 3v3 action interface, observations, opponent, arena geometries, and PPO budget, but receives no temporal-state channel, no temporal reward shaping, and no shield. This isolates the neuro-symbolic package at the architecture level, although the short smoke budget is not enough for a quality claim.

## Historical smoke runs: not a valid neural-versus-symbolic comparison

Audit correction: `phase2-3v3-neural` predates the ablation flags and used default shaping and temporal-state inputs. It was not a purely neural baseline. The historical shielded run's config records `G(in_bounds) & G(!tagged) & G(defender_behind_center -> teammate_safe) & F(captured)`, not the corrected contract. Its saved translation metadata is inconsistent. Historical checkpoints are preserved; the new provenance checks reject inconsistent specifications. No corrected-contract retraining is claimed.

| Policy | Training decisions | Team | Opponent | Shield | Evaluation | Capture rate |
|---|---:|---:|---|---|---|---:|
| Legacy unshielded guided policy | 2,048 | 3v3 | chase | no | 1 episode × 9 scenario/opponent cells | 0/9 |
| Legacy shielded guided policy | 2,048 | 3v3 | chase | yes | 1 episode × 9 scenario/opponent cells | 0/9 |

Artifacts:

- [Neural policy](results/phase2-3v3-neural/policy.zip)
- [Neural evaluation](results/phase2-3v3-neural/evaluation.summary.json)
- [Neuro-symbolic policy](results/phase2-3v3-neurosymbolic/policy.zip)
- [Neuro-symbolic evaluation](results/phase2-3v3-neurosymbolic/evaluation.summary.json)
- [Role specification](results/phase2-role-spec.json)

The equal result is expected from a 2,048-decision smoke budget: neither team has learned reliable flag capture. It verifies that both teams train, save, reload, step jointly, evaluate on all three map geometries, and report captures, rewards, tagging, and shield interventions. It is not evidence of equivalence or superiority.

The corrected role contract now means the defender remains behind the centerline at every monitored state. The historical runs still retain the prior conditional formula. Strict no-tag behavior is also formula-controlled: `G(!tagged)` is a hard absorbing LTLf failure, while omitting it gives a recoverable tagging metric without formal mission failure.

The CLI shield enforces `G(in_bounds)` only. `--role-constraint` adds explicit human constraints to the monitored mission; it does not change the shield predicate. Thus shield-only measures the effect of the arena-bounds gate. The no-tag and defender clauses are monitored constraints, not hard shield guarantees.

All five examples use the same specification and observation dimensions (disabled temporal slots are zeros). The ablation controls are explicit:

```powershell
# neural baseline: no DFA state, no shaping, no shield
mctf train-team --spec results/phase2-role-spec.json --output runs/abl-neural --team-size 3 --no-reward-shaping --no-temporal-state
# reward shaping only
mctf train-team --spec results/phase2-role-spec.json --output runs/abl-reward --team-size 3 --no-temporal-state
# temporal state only
mctf train-team --spec results/phase2-role-spec.json --output runs/abl-dfa --team-size 3 --no-reward-shaping
# shield only: arena-bounds gate, no shaping or temporal state
mctf train-team --spec results/phase2-role-spec.json --output runs/abl-shield --team-size 3 --shield --role-constraint --no-reward-shaping --no-temporal-state
# full system
mctf train-team --spec results/phase2-role-spec.json --output runs/abl-full --team-size 3 --shield --role-constraint
```

For a meaningful final comparison, run at least five seeds, 100k–500k decisions per seed, 30 held-out episodes per scenario/opponent, and report confidence intervals over training seeds. The command is:

```powershell
mctf train-team --spec results/phase2-role-spec.json --output runs/3v3-symbolic --team-size 3 --opponent chase --shield --role-constraint --steps 200000 --seed 0
mctf evaluate-team runs/3v3-symbolic/policy.zip --output runs/3v3-symbolic/evaluation.csv --episodes 30
mctf train-team --spec results/phase2-role-spec.json --output runs/3v3-neural --team-size 3 --opponent chase --steps 200000 --seed 0 --no-reward-shaping --no-temporal-state
mctf evaluate-team runs/3v3-neural/policy.zip --output runs/3v3-neural/evaluation.csv --episodes 30
```

The current implementation is therefore complete as an executable Phase 2 research system and reproducible comparison harness. The smoke artifacts deliberately do not claim that a short run has solved emergent 3v3 coordination.
