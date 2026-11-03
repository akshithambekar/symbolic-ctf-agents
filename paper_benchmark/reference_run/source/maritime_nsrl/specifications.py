"""Auditable composition of translated intent and explicit human constraints."""
from copy import deepcopy
import json
from pathlib import Path
from .logic import validate
from .temporal import compile_ltlf

ROLE_CONSTRAINTS = ["G(!tagged)", "G(defender_behind_center)"]


def combined_formula(mission, constraints):
    return " & ".join(f"({clause})" for clause in [mission, *constraints])


def check_artifact(data):
    """Reject contradictory provenance, then independently validate active logic.

    This verifies artifact consistency, not English semantic equivalence or
    physical realizability. Original model responses are never rewritten.
    """
    formula = data["ltl"]
    report = data.get("validation", {})
    if report.get("formula", formula) != formula:
        raise ValueError("Specification provenance mismatch: validation formula differs from ltl")
    if data.get("kind") == "composed":
        source = data["mission_translation"]
        check_artifact(source)
        if (data["mission_instruction"] != source["instruction"] or
                data["translated_mission_ltl"] != source["ltl"]):
            raise ValueError("Specification provenance mismatch: translated mission differs from source")
        expected = combined_formula(source["ltl"], data["human_constraints"])
        if formula != expected or data["combined_ltl"] != expected:
            raise ValueError("Specification provenance mismatch: combined formula differs from components")
    elif data.get("backend") == "llm":
        successes = [a for a in data.get("attempts", []) if a.get("valid")]
        if not successes or json.loads(successes[-1]["response"]).get("ltl") != formula:
            raise ValueError("Specification provenance mismatch: LLM response differs from ltl")
    validator = compile_ltlf if data.get("logic_backend") == "dfa" else validate
    return validator(formula)


def read_artifact(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    check_artifact(data)
    return data


def compose(source, constraints):
    check_artifact(source)
    if source.get("kind") == "composed":
        constraints = list(dict.fromkeys([*source["human_constraints"], *constraints]))
        source = source["mission_translation"]
    else:
        constraints = list(dict.fromkeys(constraints))
    formula = combined_formula(source["ltl"], constraints)
    report = compile_ltlf(formula).report()
    return {"schema_version": 1, "kind": "composed", "logic_backend": "dfa",
            "mission_instruction": source["instruction"],
            "translated_mission_ltl": source["ltl"],
            "mission_translation": deepcopy(source), "human_constraints": constraints,
            "combined_ltl": formula, "ltl": formula, "validation": report}
