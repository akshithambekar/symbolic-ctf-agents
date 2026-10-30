"""A deliberately restricted, exactly checkable finite-trace LTL compiler.

Supported contracts: conjunctions of G(propositional predicate) and at most one
F(p & F(q & ...)) mission. Eventuality is non-strict (same-time matches allowed).
Unsupported temporal formulas fail closed instead of being approximated.
"""

from dataclasses import dataclass
from itertools import product
from lark import Lark, Transformer, UnexpectedInput

AP = frozenset(
    {
        "in_bounds",
        "tagged",
        "has_flag",
        "captured",
        "home",
        "defender_behind_center",
        "attacker_forward",
        "teammate_safe",
        "opponent_push",
        "probe_upper",
        "pursuit_committed",
        "withdrawn",
        "attack_lower",
    }
)
GRAMMAR = r"""
?start: implication
?implication: disjunction | disjunction "->" implication -> implies
?disjunction: conjunction | disjunction "|" conjunction -> either
?conjunction: temporal | conjunction "&" temporal -> both
?temporal: unary | unary "U" temporal -> until
?unary: NAME -> atom
      | "!" unary -> neg
      | "G" unary -> always
      | "F" unary -> eventually
      | "X" unary -> next_
      | "(" implication ")"
NAME: /[a-z][a-z0-9_]*/
%import common.WS
%ignore WS
"""


class AST(Transformer):
    def atom(self, args):
        return ("ap", str(args[0]))

    def neg(self, args):
        return ("!", *args)

    def always(self, args):
        return ("G", *args)

    def eventually(self, args):
        return ("F", *args)

    def next_(self, args):
        return ("X", *args)

    def both(self, args):
        return ("&", *args)

    def either(self, args):
        return ("|", *args)

    def implies(self, args):
        return ("->", *args)

    def until(self, args):
        return ("U", *args)


PARSER = Lark(GRAMMAR, parser="lalr", transformer=AST())


def atoms(node):
    if node[0] == "ap":
        return {node[1]}
    return set().union(*(atoms(child) for child in node[1:]))


def propositional(node):
    return node[0] == "ap" or (
        node[0] in {"!", "&", "|", "->"} and all(propositional(x) for x in node[1:])
    )


def holds(node, labels):
    op = node[0]
    if op == "ap":
        return node[1] in labels
    if op == "!":
        return not holds(node[1], labels)
    a, b = holds(node[1], labels), holds(node[2], labels)
    return {"&": a and b, "|": a or b, "->": not a or b}[op]


def conjuncts(node):
    return conjuncts(node[1]) + conjuncts(node[2]) if node[0] == "&" else [node]


def sequence(node):
    if node[0] != "F":
        raise ValueError("Mission must begin with F")
    body = node[1]
    if body[0] == "ap":
        return [body[1]]
    if body[0] == "&" and body[1][0] == "ap" and body[2][0] == "F":
        return [body[1][1]] + sequence(body[2])
    raise ValueError("Supported mission: F(p) or F(p & F(q ...))")


@dataclass(frozen=True)
class Specification:
    formula: str
    safety: tuple
    goals: tuple[str, ...]
    witness: tuple[frozenset, ...]

    def safe(self, labels):
        return all(holds(rule, labels) for rule in self.safety)

    def report(self):
        return {
            "formula": self.formula,
            "semantics": "LTLf, nonempty finite episodes",
            "syntax_valid": True,
            "supported_fragment": True,
            "satisfiable_over_boolean_alphabet": True,
            "goals": list(self.goals),
            "safety_clauses": len(self.safety),
            "witness": [sorted(x) for x in self.witness],
            "physical_realizability_proven": False,
        }


def validate(formula: str) -> Specification:
    if not isinstance(formula, str) or not formula.strip() or len(formula) > 2000:
        raise ValueError("Formula must contain 1..2000 characters")
    try:
        tree = PARSER.parse(formula)
    except (UnexpectedInput, RecursionError) as exc:
        raise ValueError("Invalid LTL syntax") from exc
    unknown = atoms(tree) - AP
    if unknown:
        raise ValueError(f"Unknown propositions: {sorted(unknown)}")
    safety, goals = [], None
    for clause in conjuncts(tree):
        if clause[0] == "G" and propositional(clause[1]):
            safety.append(clause[1])
        elif clause[0] == "F" and goals is None:
            goals = sequence(clause)
        else:
            raise ValueError("Outside supported fragment: use G(boolean) and one F sequence")
    names = sorted(AP)
    alphabet = [
        frozenset(n for n, v in zip(names, bits) if v)
        for bits in product([False, True], repeat=len(names))
    ]
    allowed = [v for v in alphabet if all(holds(c, v) for c in safety)]
    if not allowed:
        raise ValueError("Unsatisfiable safety constraints")
    witness = []
    for goal in goals or []:
        candidates = [v for v in allowed if goal in v]
        if not candidates:
            raise ValueError(f"Unsatisfiable goal under safety constraints: {goal}")
        witness.append(candidates[0])
    return Specification(formula, tuple(safety), tuple(goals or []), tuple(witness or [allowed[0]]))


class Monitor:
    """Deterministic reward-machine state: mission index plus absorbing failure bit."""

    def __init__(self, spec):
        self.spec, self.stage, self.violated, self.violation_steps = spec, 0, False, 0

    def update(self, labels):
        before = self.stage
        violation = not self.spec.safe(labels)
        self.violated |= violation
        self.violation_steps += int(violation)
        while self.stage < len(self.spec.goals) and self.spec.goals[self.stage] in labels:
            self.stage += 1
        return self.stage - before, violation

    @property
    def satisfied(self):
        return self.stage == len(self.spec.goals) and not self.violated

    @property
    def active_goal(self):
        return self.spec.goals[self.stage] if self.stage < len(self.spec.goals) else None
