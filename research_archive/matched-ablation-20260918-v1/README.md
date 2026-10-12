# Frozen matched ablation suite

25 fresh training runs: five modes × seeds 0–4. Each requests 100,000 individual
agent transitions (100,608 after PPO rounding; 33,536 joint 3v3 decisions).
Training uses the standard arena and chasing scripted red opponents.
Each policy evaluates 30 held-out episodes per cell across three maps and three
opponents, with the same evaluation seeds 20000–20029: 6,750 episodes total.

`protocol.json` hashes every frozen source/spec/dependency file. Each run saves
its specification, arena/PPO configs, enabled components, actual budget, policy,
training progress and raw evaluation rows. `status.json` records runner status;
`runner.log` and `runner-errors.log` contain process output. `comparison.json`
is updated after each completed evaluation and is partial until all 25 finish.

Common reward = 20 × captures + 2 × pickups − 2 × entries into tagged state
− 0.01 per executed simulator step. Shaping and the shield abort penalty are
excluded. Aborted episodes still end early and are reported via deadlock rate.
Rule violations count simulator steps where any blue boat is out of bounds,
any blue boat is tagged, or designated defender 0 is across the centerline.
This is the fixed role safety contract, not accumulated absorbing DFA failure.
Tag rate is the fraction of episodes with tags; tags per episode is also given.
LTLf satisfaction includes the full role and ordered capture mission.
Capture time excludes unsuccessful episodes; missing successes are null, not zero.
Report seed means with descriptive bootstrap intervals over training seeds,
not pseudoreplicated episodes. Capture-time intervals use only seeds with
successes, and the number contributing is recorded. Five seeds is a starting
point, not a guarantee of statistical power. No superiority claim is assumed.

Resume the frozen runner using the project virtual environment and
`snapshot/scripts/matched_suite.py <this-directory>`. Completed policies and
evaluations are reused. Incomplete training directories cause a visible failure
and require investigation; the runner never silently overwrites them.
