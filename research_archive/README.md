# Historical 1v1 experiments

These files were copied unchanged from `origin/yahya-akshith_rl` at commit
`663e9b0922e450bec0c9233017919877112072bf`. The source branch contains the full
studies and additional artifacts. Some paths referenced by the preserved reports
are not included in this selection.

## Patrol policies

`phase3-confirmatory-20260927-v1/stages/02-1v1-patrol/` contains all five
completed `softened_guidance_no_tag_penalty` runs. Each run includes its policy,
configuration, training logs, evaluation rows, and trajectories. The study used
30 deterministic held-out episodes per seed. Captures were 24, 29, 30, 26, and
29 for seeds 0 through 4, respectively. Seed 2 uses the completed
`seed2_interrupted_rerun1` directory; `run_registry.json` records the mapping.
The opponent patrolled fixed waypoints.

The study protocol, stage plan, result summaries, interruption records, and the
study-level source snapshot are included. The stage plan records source hashes.
The four vendored translator files in the snapshot differ from those hashes, and
the recorded `maritime_nsrl.egg-info` files are absent. The remaining recorded
snapshot files match. Treat the snapshot as historical evidence with these
provenance limits, and verify compatibility before loading a policy in current code.

## Chase results

`phase3-confirmatory-20260927-v1/stages/03-1v1-chase/` contains the stage plan
and summaries. The full-speed chase study recorded zero pickups and captures in
150 evaluations across five seeds. `chase-recovery-20261002-v1/` contains the
protocol and summaries for a later two-seed diagnostic, which also recorded zero
pickups. The follow-up report and reward audit are under
`../experiments/followup-20261012-v1/`; those experiments found zero full-speed
pickups and positive shaped returns in some episodes without pickups.

On the current checkout, all five policies loaded and completed one evaluation
against each of patrol and chase. Seed 2 also reproduced 30/30 patrol captures
on its original evaluation seeds, with matching capture times and tag counts.
These checks do not reproduce the full five-seed study. The legacy
`enemy_territory_time` metric in the chase records is 10 times too low, as
explained in the reward audit.
