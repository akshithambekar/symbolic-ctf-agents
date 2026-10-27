from itertools import product
import pytest
from maritime_nsrl.logic import validate, Monitor, PARSER, holds


def evaluate_trace(node, trace, i=0):
    op = node[0]
    if op == "G":
        return all(evaluate_trace(node[1], trace, j) for j in range(i, len(trace)))
    if op == "F":
        return any(evaluate_trace(node[1], trace, j) for j in range(i, len(trace)))
    if op == "&":
        return evaluate_trace(node[1], trace, i) and evaluate_trace(node[2], trace, i)
    return holds(node, trace[i])


@pytest.mark.parametrize(
    "formula",
    [
        "G(in_bounds) & F(has_flag & F(captured))",
        "G(!tagged) & F(home)",
        "G(in_bounds -> !tagged)",
        "F(has_flag & F(has_flag))",
    ],
)
def test_monitor_matches_independent_trace_semantics(formula):
    alphabet = [
        set(),
        {"in_bounds"},
        {"in_bounds", "has_flag"},
        {"in_bounds", "captured"},
        {"in_bounds", "tagged", "home"},
        {"has_flag", "captured", "home"},
    ]
    spec = validate(formula)
    for n in range(1, 4):
        for trace in product(alphabet, repeat=n):
            monitor = Monitor(spec)
            for labels in trace:
                monitor.update(labels)
            assert monitor.satisfied == evaluate_trace(PARSER.parse(formula), trace)


@pytest.mark.parametrize(
    "formula",
    [
        "G(",
        "F(unknown)",
        "X(home)",
        "G(has_flag -> F(captured))",
        "F(home) | F(captured)",
        "G(home) & G(!home)",
        "G(!captured) & F(captured)",
        "",
    ],
)
def test_reject_invalid_unsupported_or_unsatisfiable(formula):
    with pytest.raises(ValueError):
        validate(formula)


def test_safety_violation_is_permanent():
    m = Monitor(validate("G(!tagged) & F(captured)"))
    m.update({"tagged"})
    m.update({"captured"})
    assert m.stage == 1 and not m.satisfied and m.violation_steps == 1


def test_capture_before_pickup_does_not_complete_sequence():
    m = Monitor(validate("F(has_flag & F(captured))"))
    m.update({"captured"})
    m.update({"has_flag"})
    assert not m.satisfied
    m.update({"captured"})
    assert m.satisfied
