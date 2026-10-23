"""Validate dataset.jsonl against props.yaml (v0.2).
Usage: python validate.py [dataset.jsonl] [props.yaml]
"""
import json, re, sys
from collections import defaultdict
import yaml

ds_path = sys.argv[1] if len(sys.argv) > 1 else "dataset.jsonl"
props_path = sys.argv[2] if len(sys.argv) > 2 else "props.yaml"

cfg = yaml.safe_load(open(props_path))
agents = cfg["agents"]["own"] + cfg["agents"]["opponent"]
VALID = set(cfg["global"].keys())
for a in agents:
    for name in cfg["per_agent"]:
        VALID.add(f"{a}_{name}")

ATOM = re.compile(r"\b[a-z][a-z0-9_]*\b")
def atoms(s):
    return set(ATOM.findall(s)) - {"true", "false"}

# Expand domain constraints over every agent.
DOMAIN = []
for tmpl in cfg.get("domain_constraints", []):
    if "{agent}" in tmpl:
        DOMAIN += [tmpl.replace("{agent}", a) for a in agents]
    else:
        DOMAIN.append(tmpl)

def relevant_domain(formula_str):
    """Constraints reachable from the formula's atoms (keeps Spot fast and is sound for these local constraints)."""
    seen, chosen, changed = atoms(formula_str), set(), True
    while changed:
        changed = False
        for i, d in enumerate(DOMAIN):
            if i not in chosen and atoms(d) & seen:
                chosen.add(i); seen |= atoms(d); changed = True
    return [DOMAIN[i] for i in sorted(chosen)]

rows = [json.loads(l) for l in open(ds_path) if l.strip()]
errors, warnings = [], []

try:
    import spot
    HAVE_SPOT = True
except ImportError:
    HAVE_SPOT = False
    warnings.append("Spot not installed: skipping parse/satisfiability/domain/equivalence checks.")

def sat(formula_str, with_domain):
    parts = [formula_str] + (relevant_domain(formula_str) if with_domain else [])
    f = spot.formula(" & ".join(f"({p})" for p in parts))
    return not spot.translate(f).is_empty()

ids, parsed = set(), []
for r in rows:
    if r["id"] in ids:
        errors.append(f"{r['id']}: duplicate id")
    ids.add(r["id"])
    ltl, err = r["ltl"], r["expected_error"]

    if ltl is None:
        if not err:
            errors.append(f"{r['id']}: ltl is null but expected_error is empty")
        continue

    unknown = atoms(ltl) - VALID
    if unknown:
        errors.append(f"{r['id']}: unknown propositions {sorted(unknown)}")
    if not HAVE_SPOT:
        continue
    try:
        f = spot.formula(ltl)
    except Exception as e:
        errors.append(f"{r['id']}: parse error: {e}")
        continue

    plain, dom = sat(ltl, False), sat(ltl, True)
    if err == "UNSAT" and plain:
        errors.append(f"{r['id']}: expected UNSAT but formula is satisfiable")
    elif err == "UNSAT_DOMAIN":
        if not plain:
            warnings.append(f"{r['id']}: marked UNSAT_DOMAIN but already UNSAT without domain constraints")
        if dom:
            errors.append(f"{r['id']}: marked UNSAT_DOMAIN but satisfiable even with domain constraints")
    elif err is None:
        if not plain:
            errors.append(f"{r['id']}: gold formula is unsatisfiable")
        elif not dom:
            errors.append(f"{r['id']}: gold formula violates the game rules (domain constraints)")
        else:
            parsed.append((r, f))

# Equivalent formulas are only allowed within the same paraphrase group.
if HAVE_SPOT:
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            (ri, fi), (rj, fj) = parsed[i], parsed[j]
            if spot.are_equivalent(fi, fj) and (ri["group"] is None or ri["group"] != rj["group"]):
                errors.append(f"{ri['id']} and {rj['id']} have equivalent formulas but are not in the same group")

# Leakage check
fewshot_f = {r["ltl"] for r in rows if r["split"] == "fewshot" and r["ltl"]}
for r in rows:
    if r["split"] == "test" and r["ltl"] in fewshot_f and r["group"] is None:
        warnings.append(f"{r['id']}: test formula identical to a few-shot formula (leakage)")

print(f"{len(rows)} rows checked; {len(errors)} errors, {len(warnings)} warnings")
for w in warnings: print("WARN ", w)
for e in errors: print("ERROR", e)
sys.exit(1 if errors else 0)
