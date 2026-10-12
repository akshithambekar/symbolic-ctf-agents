# Scripted per-player tactics: exploratory validation

This is an explicit heuristic library, not trained MARL coordination. Role-specific waypoint semantics are separate from the unchanged eight-action 1v1 PPO attacker. Conditional contracts react to grounded opponent intrusion; dynamic assignments exclude flag carriers. Roles and constraints are recorded, not hard-shielded.

| Roles | Opponent | Captures | Pickups | Red captures | Role violations (decisions) | Intrusion coverage |
|---|---|---:|---:|---:|---:|---:|
| none | static | 0/3 | 3/3 | 0 | 0 | not exposed |
| none | patrol | 1/3 | 3/3 | 0 | 0 | not exposed |
| none | chase | 0/3 | 1/3 | 0 | 0 | not exposed |
| none | raid | 0/3 | 3/3 | 3 | 0 | 0.0% |
| fixed | static | 0/3 | 2/3 | 0 | 0 | not exposed |
| fixed | patrol | 0/3 | 3/3 | 0 | 0 | not exposed |
| fixed | chase | 0/3 | 0/3 | 0 | 0 | not exposed |
| fixed | raid | 3/3 | 3/3 | 0 | 0 | 100.0% |
| conditional | static | 0/3 | 2/3 | 0 | 0 | not exposed |
| conditional | patrol | 2/3 | 3/3 | 0 | 0 | not exposed |
| conditional | chase | 0/3 | 0/3 | 0 | 0 | not exposed |
| conditional | raid | 3/3 | 3/3 | 0 | 0 | 100.0% |
| dynamic | static | 0/3 | 3/3 | 0 | 0 | not exposed |
| dynamic | patrol | 0/3 | 3/3 | 0 | 0 | not exposed |
| dynamic | chase | 0/3 | 0/3 | 0 | 0 | not exposed |
| dynamic | raid | 3/3 | 3/3 | 0 | 0 | 100.0% |

Three episodes per cell are feasibility/debugging probes, not a ranking or superiority claim. Static/patrol/chase defenders do not ordinarily invade blue territory; the separately labeled raiding policy tests the defensive trigger. The current conditional contract is instantaneous: a defender outside the home half when intrusion begins violates it, even if the controller later returns. Dynamic assignment is a heuristic and does not guarantee that any agent can return quickly enough.