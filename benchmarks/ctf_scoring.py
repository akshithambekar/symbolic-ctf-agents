"""Finite-trace semantic comparison for the CTF translator benchmark."""

import ast
from collections import deque
import re

from maritime_nsrl.logic import PARSER, atoms
from maritime_nsrl.temporal import compile_ltlf


OPS = {
    "AND": ("&", 2), "OR": ("|", 2), "IMPLY": ("->", 2),
    "NEGATION": ("!", 1), "NEXT": ("X", 1), "UNTIL": ("U", 2),
    "EVENTUALLY": ("F", 1), "ALWAYS": ("G", 1),
}
CTF_AP = frozenset({
    "in_bounds", "tagged", "has_flag", "captured", "home",
    "defender_behind_center", "attacker_forward", "teammate_safe",
})


def normalize_output(value):
    if not isinstance(value, str) or len(value) > 16_000:
        raise ValueError("Invalid formula size or type")
    value = value.strip()
    if value.startswith("["):
        tokens = ast.literal_eval(value)
        if not isinstance(tokens, list) or len(tokens) > 1_000:
            raise ValueError("Invalid prefix output")
        position = 0

        def parse():
            nonlocal position
            if position >= len(tokens):
                raise ValueError("Truncated prefix formula")
            token = tokens[position]
            position += 1
            if token in OPS:
                operator, arity = OPS[token]
                left = parse()
                if arity == 1:
                    return f"{operator}({left})"
                return f"({left}) {operator} ({parse()})"
            match = re.fullmatch(r"reach\((object_\d+)\)", token)
            if match:
                raise ValueError(f"Unmapped object proposition: {token}")
            return token

        value = parse()
        if position != len(tokens):
            raise ValueError("Unused prefix tokens")
    value = (value.replace("&&", "&").replace("||", "|").replace("¬", "!")
             .replace("∧", "&").replace("∨", "|").replace("→", "->"))
    PARSER.parse(value)
    return value


def equivalence(left, right, max_pairs=65_536):
    a, b = compile_ltlf(left), compile_ltlf(right)
    names = sorted(set(a.propositions) | set(b.propositions))
    alphabet = [frozenset(p for i, p in enumerate(names) if mask & (1 << i))
                for mask in range(1 << len(names))]
    queue = deque([(0, 0, [])])
    seen = {(0, 0)}
    left_only = right_only = None
    while queue:
        x, y, path = queue.popleft()
        for labels in alphabet:
            nx, ny = a.advance(x, labels), b.advance(y, labels)
            witness = path + [sorted(labels)]
            accepted_left = nx in a.accepting
            accepted_right = ny in b.accepting
            if accepted_left and not accepted_right and left_only is None:
                left_only = witness
            if accepted_right and not accepted_left and right_only is None:
                right_only = witness
            if left_only is not None and right_only is not None:
                return {"equivalent": False, "left_only": left_only,
                        "right_only": right_only, "exhaustive": False}
            if (nx, ny) not in seen:
                if len(seen) >= max_pairs:
                    raise ValueError("Equivalence resource limit; inconclusive")
                seen.add((nx, ny))
                queue.append((nx, ny, witness))
    return {"equivalent": left_only is None and right_only is None,
            "left_only": left_only, "right_only": right_only, "exhaustive": True}


def score(item, formula=None, explicit_rejection=False, infrastructure_error=False):
    result = {"syntax_valid": False, "supported_predicates": None,
              "unsupported_predicates": [], "compilable": False,
              "semantic_correctness": None, "ordering_preserved": None,
              "score": "Invalid/unusable", "human_review_required": True}
    if infrastructure_error:
        return {**result, "score": "Not measured",
                "reason": "Infrastructure failure; excluded from translator quality metrics"}
    if formula is None:
        if explicit_rejection and item["reference_ltl"] is None:
            return {**result, "score": "Correct", "semantic_correctness": True,
                    "reason": "Appropriate explicit rejection/clarification; review wording"}
        return {**result, "reason": "No usable formula"}
    try:
        normalized = normalize_output(formula)
        result.update(normalized_ltl=normalized, syntax_valid=True)
        unsupported = sorted(atoms(PARSER.parse(normalized)) - CTF_AP)
        result.update(supported_predicates=not unsupported,
                      unsupported_predicates=unsupported)
        if unsupported:
            return {**result, "reason": "Unsupported predicates"}
        result["compilation"] = compile_ltlf(normalized).report()
        result["compilable"] = True
        if item["reference_ltl"] is None:
            return {**result, "score": "Incorrect", "semantic_correctness": False,
                    "reason": "Invented a precise formula for unsupported or ambiguous intent"}
        comparison = equivalence(normalized, item["reference_ltl"])
        result["comparison"] = comparison
        result["semantic_correctness"] = comparison["equivalent"]
        if item.get("ordering_sensitive"):
            result["ordering_preserved"] = comparison["equivalent"]
        if comparison["equivalent"]:
            result.update(score="Correct", human_review_required=False)
        elif comparison["left_only"] is None or comparison["right_only"] is None:
            tautology = equivalence(normalized, "captured | !captured")["equivalent"]
            result.update(score="Incorrect" if tautology else "Partially correct",
                          reason="Tautology adds no obligation" if tautology else
                          "Strict strengthening or weakening; provisional human review")
        else:
            result.update(score="Incorrect", reason="Incomparable trace languages")
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result
