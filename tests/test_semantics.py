"""Independent checks against real Spot, plus bounded failure-path tests."""

import json
import os
import shutil
import sys
from time import perf_counter

import pytest

from benchmarks.formula_notation import MAX_CHARACTERS, MAX_DEPTH, MAX_TOKENS, to_spot
from benchmarks import semantics


EQUIVALENCES = [
    ("F p", "true U p"),  # Supplied Caltech slide 5 definition.
    ("G p", "!F !p"),  # Supplied slide 5 definition.
    ("F p", "!G !p"),  # Duality, independently justified.
    ("G F p", "G (true U p)"),
    ("F G p", "true U (!F !p)"),
    ("p -> q", "!p | q"),
    ("X (p & q)", "X p & X q"),
    ("p U q", "q | (p & X (p U q))"),
    ("F (p | q)", "F p | F q"),
    ("F reach(object_7)", "!G !reach(object_7)"),
    ("G (F visit(other_99) -> X !alert)", "G (!F visit(other_99) | X !alert)"),
]
NON_EQUIVALENCES = [
    ("F (p & q)", "F p & F q"),  # Supplied Caltech slide 12.
    ("G (p | q)", "G p | G q"),  # Supplied slide 12.
    ("G F p", "F G p"),  # Supplied slide 5: progress versus stability.
    ("F p", "X F p"),  # p can hold only at the current step.
    ("p", "q"),
    ("F reach(object_7)", "F reach(object_8)"),
    ("F p", "G p"),
    ("p U q", "q U p"),
    ("true", "false"),
]


@pytest.fixture
def spot():
    executable = shutil.which(os.environ.get("LTLFILT", "ltlfilt"))
    if executable is None:
        pytest.fail("Install Spot's ltlfilt to run semantic integration tests (see benchmarks/SEMANTICS.md).")
    return executable


@pytest.mark.parametrize("generated,reference", EQUIVALENCES)
def test_semantic_equivalence(spot, generated, reference):
    assert generated != reference  # Every pair would fail a string-equality oracle.
    result = semantics.evaluate(generated, reference, generated_notation="infix", checker_path=spot)
    assert result["outcome"] == "equivalent", result
    assert result["checker_version"]


@pytest.mark.parametrize("generated,reference", NON_EQUIVALENCES)
def test_semantic_non_equivalence(spot, generated, reference):
    result = semantics.evaluate(generated, reference, generated_notation="infix", checker_path=spot)
    assert result["outcome"] == "semantic_mismatch", result


@pytest.mark.parametrize("prefix,infix", [
    ("['AND', 'p', 'q']", "p & q"),
    ("['OR', 'p', 'q']", "p | q"),
    ("['NEGATION', 'p']", "!p"),
    ("['NEXT', 'p']", "X p"),
    ("['UNTIL', 'p', 'q']", "p U q"),
    ("['EVENTUALLY', 'p']", "F p"),
    ("['ALWAYS', 'p']", "G p"),
    ("['IMPLY', 'p', 'q']", "p -> q"),
    ("['ALWAYS', 'EVENTUALLY', 'reach(object_7)']", "G F reach(object_7)"),
    ('["EVENTUALLY", "reach(object_7)",]', "F reach(object_7)"),
])
def test_native_operator_conversion(spot, prefix, infix):
    result = semantics.evaluate(prefix, infix, checker_path=spot)
    assert result["outcome"] == "equivalent", result


def test_atomic_proposition_mapping_is_shared_and_injective(spot):
    result = semantics.evaluate("['AND', 'reach(object_7)', 'reach_object_7']", "reach_object_7 & reach(object_7)", checker_path=spot)
    assert result["outcome"] == "equivalent"
    assert result["atomic_propositions"] == ["reach(object_7)", "reach_object_7"]
    assert '"reach(object_7)"' in result["normalized_generated"]
    assert '"reach(object_7)"' in result["normalized_reference"]
    mismatch = semantics.evaluate("['reach(object_7)']", "reach_object_7", checker_path=spot)
    assert mismatch["outcome"] == "semantic_mismatch"


@pytest.mark.parametrize("bad", [
    "", "[]", "p", "['AND', 'p']", "['p', 'q']", "[['p']]", "[1]", "[True]",
    "['UNKNOWN', 'p']", "['p'] + ['q']", "[__import__('os')]", "['p' 'q']",
    "['reach(object_7,object_8)']", "['reach(object-7)']", "['reach(object_7) | q']",
    "['p\\n']", "['EVENTUALLY', '']", "['EVENTUALLY', 'F']", "['p',,]",
    "['p']\n['q']", "['p'] garbage", None, ["EVENTUALLY", "p"],
    "x" * (MAX_CHARACTERS + 1),
    "[" + ",".join(["'NEXT'"] * MAX_DEPTH + ["'p'"]) + "]",
    "[" + ",".join(["'p'"] * (MAX_TOKENS + 1)) + "]",
])
def test_invalid_generated_formula(bad):
    result = semantics.evaluate(bad, "F p")
    assert result["outcome"] == "invalid_generated_formula", result


@pytest.mark.parametrize("bad", [
    "", "F", "F (p", "p &", "(p U q U)", "p q", "p && q", "p; q", "p R q",
    "reach(object_7,object_8)", "reach()", "p.__class__", "F __import__('os')",
    "p\nq", None, "(" * (MAX_DEPTH + 1) + "p" + ")" * (MAX_DEPTH + 1),
    "X " * (MAX_DEPTH + 1) + "p", "p & " * MAX_DEPTH + "p",
    "x" * (MAX_CHARACTERS + 1),
])
def test_invalid_reference_formula(bad):
    result = semantics.evaluate("['p']", bad)
    assert result["outcome"] == "invalid_reference_formula", result


def test_generated_invalid_takes_precedence_even_when_reference_oversized():
    result = semantics.evaluate("['AND']", "x" * (MAX_CHARACTERS + 1), checker_path="/no/checker")
    assert result["outcome"] == "invalid_generated_formula"


def test_input_is_never_executed(tmp_path):
    sentinel = tmp_path / "should-not-exist"
    malicious = f"[__import__('pathlib').Path({str(sentinel)!r}).touch()]"
    assert semantics.evaluate(malicious, "p")["outcome"] == "invalid_generated_formula"
    assert not sentinel.exists()


def test_infix_precedence_and_parentheses():
    assert to_spot("p | q & r", "infix")[0] == '("p" | ("q" & "r"))'
    assert to_spot("p -> q -> r", "infix")[0] == '("p" -> ("q" -> "r"))'
    assert to_spot("p U q U r", "infix")[0] == '("p" U ("q" U "r"))'
    assert to_spot("!F(p) & X q", "infix")[0] == '(!(F("p")) & X("q"))'


def test_missing_checker_is_error():
    result = semantics.evaluate("['p']", "p", checker_path="/no/such/ltlfilt")
    assert result["outcome"] == "checker_error"
    assert result["checker_version"] is None


def fake_checker(tmp_path, body):
    executable = tmp_path / "ltlfilt-test"
    executable.write_text(f"#!{sys.executable}\nimport sys, time\n" + body)
    executable.chmod(0o700)
    return str(executable)


@pytest.mark.parametrize("body", [
    "raise SystemExit(2)",
    "print('not Spot')",
    "print('ltlfilt (spot) 0.test') if '--version' in sys.argv else sys.exit(2)",
    "print('ltlfilt (spot) 0.test' if '--version' in sys.argv else '0')",
    "print('ltlfilt (spot) 0.test' if '--version' in sys.argv else '1'); "
    "sys.exit(2 if any(a.startswith('--equivalent-to=') for a in sys.argv) else 0)",
])
def test_checker_initialization_and_runtime_failures(tmp_path, body):
    result = semantics.evaluate("['p']", "p", checker_path=fake_checker(tmp_path, body))
    assert result["outcome"] == "checker_error", result


def test_timeout_covers_checker_initialization(tmp_path):
    path = fake_checker(tmp_path, "time.sleep(30)")
    start = perf_counter()
    result = semantics.evaluate("['p']", "p", checker_path=path, timeout_seconds=0.4)
    assert result["outcome"] == "checker_timeout"
    assert perf_counter() - start < 3


def test_timeout_covers_equivalence(tmp_path):
    path = fake_checker(tmp_path,
        "if '--version' in sys.argv:\n    print('ltlfilt (spot) 0.test')\n"
        "elif any(a.startswith('--equivalent-to=') for a in sys.argv):\n    time.sleep(30)\n"
        "else:\n    print('1')\n")
    result = semantics.evaluate("['p']", "p", checker_path=path, timeout_seconds=0.6)
    assert result["outcome"] == "checker_timeout"


def test_deadline_also_covers_notation_conversion(monkeypatch):
    original = semantics.subprocess.Popen
    script = (
        "import time; import benchmarks.semantics as s; "
        "s.to_spot = lambda *args: time.sleep(30); s._main()"
    )
    def launch(args, **kwargs):
        return original([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(semantics.subprocess, "Popen", launch)
    result = semantics.evaluate("['p']", "p", timeout_seconds=0.4)
    assert result["outcome"] == "checker_timeout"


def test_worker_failure_is_not_a_semantic_mismatch(monkeypatch):
    def cannot_launch(*args, **kwargs):
        raise OSError("do not persist raw diagnostics")

    monkeypatch.setattr(semantics.subprocess, "Popen", cannot_launch)
    result = semantics.evaluate("['p']", "p")
    assert result["outcome"] == "checker_error"
    assert "do not persist" not in json.dumps(result)
