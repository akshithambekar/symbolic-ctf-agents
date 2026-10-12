"""Author-defined references, fixed before model calls; require human review."""
VERSION = "ctf-translators-1.0"
GLOSSARY = """CTF benchmark domain: 3v3 team, nonempty finite episodes, strong X, inclusive F.
in_bounds = all controlled boats inside arena; tagged = any controlled boat tagged;
has_flag = any controlled boat carrying enemy flag; captured = team score event;
home = any controlled boat within catch radius of own flag home (NOT own half);
defender_behind_center = designated agent 0 behind center (team size >=2);
attacker_forward = any agent 1..n-1 beyond center; teammate_safe = no teammate tagged.
Only these eight predicates are available. An unspecified defender is NOT necessarily
agent 0. No predicate represents opponent approach, support, communication, distance,
or an arbitrary boat's own-half occupancy. Ask for clarification / return an error
for unsupported or ambiguous intent. For ordinary 'then', same-state witnesses are
allowed; 'strictly later' requires X. No implicit safety clauses may be added.
"""

# (category, paraphrase group, gold or None, first phrasing, paraphrase, rationale)
CASES = [
 ("safety", "bounds", "G(in_bounds)", "Always stay inside the arena.", "Remain within arena bounds throughout the episode.", "All steps in bounds."),
 ("safety", "tag", "G(!tagged)", "Never get tagged.", "At no point may a controlled boat be tagged.", "Absorbing safety requirement."),
 ("eventual", "capture", "F(captured)", "Eventually capture the enemy flag.", "Score at least once before the episode ends.", "No additional no-tag clause."),
 ("eventual", "home", "F(home)", "Eventually reach the home flag stand.", "At some time reach within catch radius of our flag home.", "Home means flag vicinity, not own half."),
 ("ordered", "pickup_capture", "F(has_flag & F(captured))", "Pick up the enemy flag and then capture it.", "Obtain the opposing flag, then eventually score.", "Inclusive sequencing as specified by benchmark context."),
 ("ordered", "strict", "F(has_flag & X(F(captured)))", "Carry the flag and capture it strictly later.", "At some point possess the flag; score at a later step, not that same step.", "Strong next distinguishes strict from inclusive order."),
 ("ordered", "home_capture", "F(home & F(captured))", "Reach home and then eventually capture.", "Visit our flag home, followed by a score.", "Reverse witness must not suffice."),
 ("conditional", "response", "G(has_flag -> F(home))", "Whenever carrying the flag, eventually reach our flag stand.", "Every possession of the flag must be followed by reaching home.", "Response includes same step and applies at every time."),
 ("conditional", "response_capture", "G(has_flag -> F(captured))", "Whenever we have the flag, eventually score.", "Each time a boat carries the flag, the team must eventually capture.", "Vacuously true if no possession; no invented eventual possession."),
 ("conditional", "safe_attacker", "G(attacker_forward -> teammate_safe)", "Whenever an attacker is forward, no teammate may be tagged.", "If an attacker is beyond midfield, the team must be untagged at that step.", "Direction of implication matters."),
 ("role", "defender", "G(defender_behind_center)", "Keep designated defender agent 0 behind the centerline at all times.", "Agent zero must always remain in our half.", "Designated role only, not existential quantification."),
 ("role", "conditional_defender", "G(attacker_forward -> defender_behind_center)", "While an attacker is forward, agent 0 must remain behind midfield.", "If any attacker crosses into enemy territory, keep the designated defender in our half.", "Do not replace this with a global defender obligation."),
 ("combined", "safe_capture", "G(in_bounds) & G(!tagged) & F(captured)", "Stay in bounds and eventually capture without getting tagged.", "Always remain inside, never be tagged, and score before the deadline.", "Three independent clauses."),
 ("combined", "bounds_order", "G(in_bounds) & F(has_flag & F(captured))", "Always stay in bounds. Pick up the enemy flag and then capture it.", "Remain inside the arena while completing possession followed by capture.", "Safety plus ordered task."),
 ("difficult", "until", "(!tagged U captured)", "Stay untagged until capturing, and do capture.", "Avoid tags until the first score; scoring must eventually happen.", "Strong until, not weak until and not global no-tag."),
 ("difficult", "next", "X(in_bounds)", "On the next step, be inside the arena.", "The immediately following state must be in bounds.", "Strong next rejects a one-state trace."),
 ("ambiguous", "own_side", None, "If you have the flag, return to your side.", "When carrying it, go back to our half.", "Own-half predicate is absent; home would silently strengthen the instruction."),
 ("ambiguous", "one_defender", None, "Keep one defender behind the centerline.", "At least one team member should stay back to defend.", "Existential role cannot be replaced with designated agent 0."),
 ("unsupported", "approach", None, "If an opponent approaches, retreat toward home.", "Fall back when the enemy gets close.", "Approach/close and retreat direction have no grounded AP."),
 ("unsupported", "support", None, "One attacker should capture while another supports.", "Have one teammate score as a second teammate provides support.", "Support and per-agent task quantification are not grounded."),
]


def dataset():
    rows = []
    for category, group, gold, first, second, rationale in CASES:
        for variant, instruction in enumerate((first, second)):
            rows.append(dict(id=f"{group}-{variant}", category=category,
                paraphrase_group=group, instruction=instruction, reference_ltl=gold,
                expected="formula" if gold else "clarify_or_reject",
                rationale=rationale, ordering_sensitive=category=="ordered",
                reference_author="project-authored; independent human review pending"))
    return {"version":VERSION, "semantics":"LTLf over nonempty finite traces",
            "glossary":GLOSSARY, "items":rows}
