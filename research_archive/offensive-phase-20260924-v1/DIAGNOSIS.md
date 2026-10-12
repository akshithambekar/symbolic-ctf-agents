# Phase 3 diagnosis: offensive recovery

## Evidence from the immutable matched study

The main failure occurs before flag acquisition. Pickup rates were 0.5%
(neural), 1.3% (temporal state), 2.7%
(reward shaping), 8.7% (shield), and 0.0%
(full guidance). Full guidance never picked up a flag. Reward shaping contributed
-14.79 mean return in the shaping-only condition and
-5.11 in the full condition. Against a +2 pickup reward, these
large negative shaping totals can dominate the sparse offensive signal. The full
policy's favorable common reward is partly associated with conservative behavior
and early shield termination; it did not complete the mission.

## Scripted feasibility and failure stages

The discrete action abstraction is sufficient for reliable 1v1 static capture in
all three arenas. A coordinated two-attacker 3v3 policy also captures reliably in
standard static play by using the defender's tag cooldown. Thus zero learned 3v3
captures are not caused by an impossible simulator or missing capture action.

Patrol failures generally happen after pickup on the return leg. In standard 1v1,
the simple oracle reached/picked up the flag in 5/5 episodes but captured in 1/5.
Chase failures happen earlier: it crossed midfield in every episode but never
reached the flag in any tested map or team size. This separates the research
problem into return-path robustness against patrol and approach/evasion against chase.

The oracle is deliberately simple and failure does not prove a configuration is
physically impossible. It only identifies what this action abstraction and tested
heuristic could demonstrate. Full raw trajectories are retained.

## Current exploratory screen

| Stage | Variant | Captures | Rate | Promoted |
|---|---|---:|---:|---|
Not complete yet.

The screen starts with 1v1 static using two seeds and 30,720 transitions. Variants
must reach 10% capture over 40 held-out episodes to advance. The initial candidates
are neural, temporal-state-only, and soft guidance without `G(!tagged)` or a fixed
defender. Soft guidance increases pickup/capture reward, reduces tag/time penalties,
and removes absorbing-failure penalties. No confirmatory claim will be made from
this exploratory screen.

## Recommendation gate

Do not schedule a five-seed confirmatory matrix until a candidate captures reliably
in 1v1 static and patrol, shows a nonzero approach/pickup rate against chase, and
does not materially worsen tags or fixed safety violations. Shield variants should
be reconsidered only after action-substitution traces show where interventions occur;
the current arena-bound shield is not an offensive constraint and its deadlocks can
artificially shorten episodes.
