from itertools import product
import pytest
from maritime_nsrl.logic import PARSER
from maritime_nsrl.temporal import compile_ltlf, TemporalMonitor


def semantics(f, trace, i=0):
    op = f[0]
    if op == "ap": return f[1] in trace[i]
    if op == "!": return not semantics(f[1],trace,i)
    if op == "&": return semantics(f[1],trace,i) and semantics(f[2],trace,i)
    if op == "|": return semantics(f[1],trace,i) or semantics(f[2],trace,i)
    if op == "->": return not semantics(f[1],trace,i) or semantics(f[2],trace,i)
    if op == "X": return i+1<len(trace) and semantics(f[1],trace,i+1)
    if op == "F": return any(semantics(f[1],trace,j) for j in range(i,len(trace)))
    if op == "G": return all(semantics(f[1],trace,j) for j in range(i,len(trace)))
    if op == "U": return any(semantics(f[2],trace,j) and all(semantics(f[1],trace,k)
                              for k in range(i,j)) for j in range(i,len(trace)))
    raise AssertionError(op)


@pytest.mark.parametrize("formula", ["X(!tagged)", "!(X tagged)", "G(F captured)",
    "F(G !tagged)", "(!tagged U captured)", "G(tagged -> F(captured))",
    "G(tagged -> X(captured))", "F(tagged) | G(!captured)",
    "!(tagged U captured)", "F(tagged & X(captured))"])
def test_dfa_matches_independent_finite_trace_semantics(formula):
    dfa = compile_ltlf(formula)
    for n in (1,2,3,4):
        for trace in product([set(), {"tagged"}, {"captured"}, {"tagged","captured"}],repeat=n):
            monitor = TemporalMonitor(dfa)
            for labels in trace: monitor.update(labels)
            assert monitor.satisfied == semantics(PARSER.parse(formula),trace)


@pytest.mark.parametrize("formula", ["G(X in_bounds)", "G(!captured) & F(captured)",
                                    "F(captured & !captured)"])
def test_unsatisfiable_finite_formulas(formula):
    with pytest.raises(ValueError,match="Unsatisfiable"): compile_ltlf(formula)


def test_resource_limit_is_not_misreported_as_unsatisfiable():
    with pytest.raises(ValueError,match="inconclusive"): compile_ltlf("F(captured)",max_states=1)


def test_end_of_trace_pending_next_is_not_satisfied():
    m = TemporalMonitor(compile_ltlf("X(!tagged)"))
    m.update(set())
    assert not m.satisfied and not m.impossible
    m.update(set())
    assert m.satisfied
