"""General LTLf progression to a reachable deterministic finite automaton.

Boolean connectives, strong next X, until U, eventually F, always G, and arbitrary
nesting are supported. A compilation resource limit is an error, never 'unsat'.
"""

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from .logic import AP, PARSER, atoms

TRUE, FALSE, NONEMPTY = ("true",), ("false",), ("nonempty",)


def negate(node):
    if node == TRUE:
        return FALSE
    if node == FALSE:
        return TRUE
    if node[0] == "!":
        return node[1]
    return ("!", node)


def combine(op, *nodes):
    identity, absorbing = (TRUE, FALSE) if op == "&" else (FALSE, TRUE)
    items = set()
    for node in nodes:
        if node == absorbing:
            return absorbing
        if node == identity:
            continue
        items.update(node[1:] if node[0] == op else [node])
    if any(negate(x) in items for x in items):
        return absorbing
    # Absorption: a | (a & b) == a, and its dual.
    opposite = "|" if op == "&" else "&"
    items = {x for x in items if not (x[0] == opposite and any(y in items for y in x[1:]))}
    if not items:
        return identity
    if len(items) == 1:
        return next(iter(items))
    return (op, *sorted(items, key=repr))


def normalize(node):
    op = node[0]
    if op == "ap":
        return node
    children = [normalize(x) for x in node[1:]]
    if op in {"&", "|"}:
        return combine(op, *children)
    if op == "!":
        return negate(children[0])
    if op == "->":
        return combine("|", negate(children[0]), children[1])
    return (op, *children)


@lru_cache(maxsize=100000)
def progress(node, labels):
    op = node[0]
    if node in (TRUE, FALSE):
        return node
    if node == NONEMPTY:
        return TRUE
    if op == "ap":
        return TRUE if node[1] in labels else FALSE
    if op == "!":
        return negate(progress(node[1], labels))
    if op in {"&", "|"}:
        return combine(op, *(progress(x, labels) for x in node[1:]))
    if op == "X":
        return combine("&", NONEMPTY, node[1])
    if op == "F":
        return combine("|", progress(node[1], labels), node)
    if op == "G":
        return combine("&", progress(node[1], labels), node)
    if op == "U":
        return combine(
            "|", progress(node[2], labels), combine("&", progress(node[1], labels), node)
        )
    raise ValueError(op)


def accepts_empty(node):
    op = node[0]
    if op in {"true", "G"}:
        return True
    if op in {"false", "nonempty", "ap", "X", "F", "U"}:
        return False
    if op == "!":
        return not accepts_empty(node[1])
    if op == "&":
        return all(accepts_empty(x) for x in node[1:])
    if op == "|":
        return any(accepts_empty(x) for x in node[1:])
    raise ValueError(op)


@dataclass
class Automaton:
    formula: str
    propositions: tuple
    states: tuple
    transitions: tuple
    accepting: frozenset
    live: frozenset
    witness: tuple

    def label_index(self, labels):
        return sum((1 << i) for i, p in enumerate(self.propositions) if p in labels)

    def advance(self, state, labels):
        return self.transitions[state][self.label_index(labels)]

    def report(self):
        return {
            "formula": self.formula,
            "semantics": "LTLf; strong X; nonempty finite traces",
            "operators": ["!", "&", "|", "->", "X", "U", "F", "G"],
            "states": len(self.states),
            "accepting": sorted(self.accepting),
            "satisfiable": True,
            "witness": [sorted(x) for x in self.witness],
            "physical_realizability_proven": False,
        }


@lru_cache(maxsize=64)
def compile_ltlf(formula, max_states=256):
    if not isinstance(formula, str) or not 0 < len(formula) <= 2000:
        raise ValueError("Formula length must be 1..2000")
    try:
        tree = PARSER.parse(formula)
        names = tuple(sorted(atoms(tree)))
        if set(names) - AP:
            raise ValueError(f"Unknown propositions: {set(names) - AP}")
        first = normalize(tree)
    except Exception as exc:
        raise ValueError(f"Cannot parse formula: {exc}") from exc
    alphabet = tuple(
        frozenset(p for i, p in enumerate(names) if mask & (1 << i))
        for mask in range(2 ** len(names))
    )
    states, indexes, transitions, traces = [first], {first: 0}, [], [()]
    witness = None
    for node in states:
        row = []
        for labels in alphabet:
            nxt = progress(node, labels)
            path = traces[indexes[node]] + (labels,)
            if witness is None and accepts_empty(nxt):
                witness = path
            if nxt not in indexes:
                if len(states) >= max_states:
                    raise ValueError(
                        f"DFA compilation exceeded {max_states} states; validation inconclusive"
                    )
                indexes[nxt] = len(states)
                states.append(nxt)
                traces.append(path)
            row.append(indexes[nxt])
        transitions.append(tuple(row))
    if witness is None:
        raise ValueError("Unsatisfiable on all nonempty finite traces")
    accepting = frozenset(i for i, x in enumerate(states) if accepts_empty(x))
    reverse = [set() for _ in states]
    for source, row in enumerate(transitions):
        for target in row:
            reverse[target].add(source)
    live, queue = set(accepting), deque(accepting)
    while queue:
        for predecessor in reverse[queue.popleft()]:
            if predecessor not in live:
                live.add(predecessor)
                queue.append(predecessor)
    return Automaton(
        formula, names, tuple(states), tuple(transitions), accepting, frozenset(live), witness
    )


class TemporalMonitor:
    def __init__(self, automaton):
        self.automaton, self.state, self.length = automaton, 0, 0

    def update(self, labels):
        self.state = self.automaton.advance(self.state, labels)
        self.length += 1
        return self.state

    @property
    def satisfied(self):
        return self.length > 0 and self.state in self.automaton.accepting

    @property
    def impossible(self):
        return self.state not in self.automaton.live
