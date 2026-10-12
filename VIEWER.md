# Watch and replay PyQuaticus

The viewer runs the saved policy in the existing simulator and mirrors states into
PyQuaticus's pygame renderer. It does not replace simulator physics. The separate
rendering instance prevents pygame surfaces from entering shield simulator copies.
Training and evaluation behavior are unchanged. Windows with a graphical desktop
can open the window; `--headless --record NEW_DIRECTORY` renders PNGs without one.

Run from this project with its virtual environment activated (`mctf`), or substitute
`.venv/Scripts/python.exe -m maritime_nsrl.cli` for `mctf`.

```powershell
# Saved 1v1 neural baseline
mctf watch research_archive/offensive-phase-20260924-v1/runs/stage1-1v1-static/neural/seed0/policy.zip --scenario standard --opponent static --seed 40000

# Saved softened guidance; try patrol or chase as diagnostic overrides
mctf watch research_archive/offensive-phase-20260924-v1/runs/stage1-1v1-static/soft_guidance/seed0/policy.zip --opponent patrol --episodes 2 --speed 10

# Scripted/oracle agent, 1v1 or 3v3
mctf watch oracle --team-size 1 --scenario wide --opponent chase --seed 30000
mctf watch oracle --team-size 3 --scenario standard --opponent static --seed 30000

# Completed 3v3 full-guidance policy
mctf watch research_archive/matched-ablation-20260918-v1/full-seed0/policy.zip --opponent chase --speed 20

# Historical Phase 3 episode (zero-based episode index in JSONL)
mctf replay research_archive/offensive-phase-20260924-v1/runs/stage1-1v1-static/soft_guidance/seed0/trajectories.jsonl --episode 0 --speed 10

# Record fresh, complete snapshots and PNG frames, then replay without a policy
mctf watch oracle --team-size 3 --record ../watch-capture-01 --speed 20
mctf replay ../watch-capture-01/replay.jsonl --episode 0 --speed 10

# Headless smoke check or presentation frames
mctf watch oracle --team-size 3 --headless --max-frames 3 --record ../watch-smoke-01
```

`--scenario` is standard, wide or narrow; `--opponent` is static, patrol or chase.
`--team-size` is inferred for saved models and cannot change their observation shape.
`--config` supplies an explicit ArenaConfig JSON for diagnostic use; saved policy
spaces and formal specification must still match. Such overrides are new diagnostic
rollouts, not historical evaluations. Legacy seven-action 1v1 checkpoints are also
supported in their original standard arena. Stale/inconsistent saved specifications
are rejected instead of silently reinterpreted. `--stochastic` samples actions;
deterministic inference is the default. Space pauses and Esc/window-close exits.

`--speed 10` displays ten seconds of game time per wall-clock second, excluding
simulation/rendering overhead. Frames sample decisions (typically five simulated
seconds apart), so movement is not smooth physics-frame animation. This is suitable
for diagnosing paths and events; it is not a high-frame-rate simulator capture.

The native renderer shows boundaries, centerline, flags, boat IDs, flag possession,
and tagging/cooldown indicators. Added labels distinguish designated defender 0,
attackers 1–2, and red opponents. Role labels describe the convention and do not
force a learned policy to defend. The overlay displays time, team scores/tags,
action names, final executed discrete actions when available, shield counters,
DFA state, propositions and formal failure, plus approach/return/capture stage.
Executed actions describe the final simulator substep of the decision; the shield
can substitute more than once within a decision.

Fresh recordings contain all blue/red positions, headings, speed, possession,
tag/cooldown state, sides, flag homes, score, actions, and formal diagnostics. They
also save viewer/policy hashes and arguments. PNG frames are the portable recording
format; video encoding is not a required dependency. Existing directories are never
overwritten. Unbounded headless recording can use substantial disk space; use
`--max-frames` for a short sample.

Historical Phase 3 JSONL records contain blue positions, possession, tags and some
actions/DFA states only. Replay explicitly labels these as PARTIAL, hides opponents,
uses default headings when absent, and does not invent their motion or cooldowns.
Flag homes come from arena geometry; historical flag status is incomplete. Old
Phase 2 aggregate CSVs contain no trajectories and cannot be faithfully replayed.
Use `watch` with the saved Phase 2 policy and seed for a new diagnostic rollout,
clearly distinct from replay of an original episode. No retraining is required.
