## Observations

Both wrappers read `game.state`. Policies receive full information about every
boat, including opponents. This assumes perfect simulation-state access and does
not establish a sensor-limited deployment interface.

| Wrapper | Policy input |
| --- | --- |
| `MaritimeEnv` (1v1) | 19 floats: both positions (4), heading sine/cosine (4), speeds (2), possession (2), tagged status (2), cooldowns (2), elapsed fraction (1), mission stage (1), absorbing safety-failure bit (1) |
| `TeamArena` | Eight values per boat: x/y, heading sine/cosine, speed, possession, tagged status, cooldown; then elapsed fraction, controlled-agent identity, ever-carried flag, and one-hot DFA state |

The wrappers scale positions by arena dimensions, speeds by each boat's maximum,
cooldowns by their configured duration, and elapsed time by the episode horizon.
They clip `float32` values to `[-2, 2]`. Team observations put the acting boat first,
followed by its teammates and opponents. For team size N and S DFA states, each
input has `16N + 3 + S` values. An optional defender-distance diagnostic adds one.

Disabled temporal channels remain zero to preserve dimensions within a comparison.
The two wrappers have different contracts; their checkpoints are not interchangeable.

## Actions and timing

PPO selects a discrete waypoint primitive. A controller converts that target into
native continuous `[speed, relative_heading]` commands; Heron dynamics execute them.
Tagged boats follow the simulator's recovery controller.

| ID | 1v1 target | Team target |
| --- | --- | --- |
| 0 | Enemy flag home `(140, 40)` | Enemy flag home |
| 1 | Own flag home `(20, 40)` | Own flag home |
| 2–3 | Midfield flank `(85, 18/62)` | Low/high offensive flank |
| 4–5 | Forward flank `(120, 18/62)` | Low/high return lane |
| 6 | Hold | Evasion target |
| 7 | Unavailable | Hold |

Team targets adapt to arena dimensions and current positions; inspect `target()`
in `research_env.py` for exact coordinates. Each native environment step advances
one simulated second (`tau=0.1`, speedup=10). One PPO decision executes up to five
such steps, stopping early on an episode ending.

Defaults: 240-second horizon, first capture ends the episode, tagging cooldown
30 seconds. Time expiration is a terminal finite-horizon task, preventing PPO from
bootstrapping beyond the deadline. PPO rounds requested budgets to complete batches:
512 decisions for 1v1; 256 decisions per controlled boat for team training.

## Rewards

The default common reward per native step is:

```text
20 × own captures + 2 × own pickups − 2 × newly tagged controlled boats − 0.01
```

Counts are transition deltas. Initialized possession does not earn a pickup bonus.
The team wrapper permits reward coefficients to vary and separately logs the fixed
common reward for comparison. Neither default includes opponent-capture or collision
penalties; decide those explicitly before balanced offensive/defensive training.

1v1 guidance adds `5 × (0.99 × next_potential − current_potential)`, milestone
bonuses, and safety penalties. Terminal potential is zero. Team guidance uses its
DFA state and acceptance/failure terms. The full shaping package changes the
objective; policy invariance is not established. Compare common rewards and actual
task outcomes across arms.

## Logic and shielding

- `fragment`: conjunctions of `G(boolean)` constraints and one eventual goal
  sequence, such as `F(has_flag & F(captured))`. Same-step goal matches are allowed.
  This backend feeds the original 1v1 environment.
- `dfa`: Boolean operators, `X`, `U`, `F`, `G`, and nested finite-trace formulas;
  used by the team environment. Strong `X` requires another state. Compilation
  limits produce an inconclusive error.
- Supported propositions: bounds, tagged, possession, capture, home, designated
  defender position, attacker position, teammate safety, and opponent intrusion.
  The 1v1 grounder supplies only the first five and rejects team-only constraints.
  Team grounding supplies all nine.
- Validation checks syntax, vocabulary, satisfiability, and saved provenance. It
  does not prove English equivalence or physical realizability.
- The predictive shield checks one-step simulator copies and commits a safe
  successor. Its default contract is `G(in_bounds)`. Failure to find a safe candidate
  ends the episode as a deadlock. It gives no robust real-world safety guarantee.

Offline translation accepts listed sentences exactly. The optional LLM backend
uses `LTL_API_KEY`, `--model`, and `--base-url`, with at most two validation repairs.
Gameplay makes no model calls. `compose` records human constraints separately from
the original translation and revalidates their combined formula.

## Competition rule alignment

Reference: [official rules](https://www.mctf26.com/rules), reviewed 2026-10-13.

| Topic | Current research behavior | Required follow-up |
| --- | --- | --- |
| Capture/recovery | Capture on return to own territory; untag near own flag home | Implement/confirm corner bases and clarify their geometry |
| Boundary exit | Auto-tag and controller-driven recovery | Resolve the rules' immediate nearest-base return |
| Powerplay | No implementation in imported wrappers | Obtain scheduling/agent-selection details, then implement |
| Duration/scoring | 240 seconds; first capture; offensive reward | Add a ten-minute match configuration and score-difference evaluation |
| Tagging | 10 m threshold plus 30-second cooldown | Confirm cooldown and exact threshold semantics |
| Collisions | Upstream detection uses summed boat radii | Verify event counting against the competition's 4 m criterion |

Rule alignment remains open. Preserving this setup lets us reproduce its experiments
while resolving the competition contract separately.
