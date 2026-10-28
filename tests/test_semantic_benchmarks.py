"""Shared semantic checks for both implementations, using fixed model responses."""

import json
import os
from pathlib import Path

from langchain_core.language_models import FakeListLLM
import pytest

from benchmarks import run
from benchmarks.semantics import DEFAULT_TIMEOUT_SECONDS


COMPARISON_CASES = run.load_cases(Path(__file__).resolve().parents[1] / "benchmarks/comparison_cases.json")
BODIES = [
    "return ltl_eventually(ap(reach, 'object_7'))",
    "return ltl_always(ltl_not(ap(reach, 'object_3')))",
    "return ltl_and(ltl_eventually(ap(reach, 'object_1')), ltl_eventually(ap(reach, 'object_2')))",
    "return ltl_eventually(ltl_or(ap(reach, 'object_1'), ap(reach, 'object_2')))",
]
IMPLEMENTATIONS = [
    "restricted",
    pytest.param("ltlcodegen-container", marks=pytest.mark.skipif(
        os.environ.get("LTL_DOCKER_TESTS") != "1", reason="Docker integration tests are opt-in",
    )),
]


def fixed_model(bodies, implementation):
    responses = ["def question():\n    " + body for body in bodies]
    if implementation == "ltlcodegen-container":
        responses = ["```python\n" + code + "\n```" for code in responses]
    return FakeListLLM(responses=responses)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_fixed_comparison_cases(implementation, tmp_path):
    records = run.run_cases(COMPARISON_CASES, fixed_model(BODIES, implementation), "offline/fixed", "fake", implementation=implementation)
    assert len(records) == 4
    for record in records:
        assert record["outcome"] == "translated", record
        assert record["semantic_evaluation"]["outcome"] == "equivalent", record
        assert record["semantic_evaluation"]["timeout_seconds"] == DEFAULT_TIMEOUT_SECONDS
        assert record["attempts"] == record["model_invocations"] == 1
        assert record["provider_requests"] == 0
        assert record["failure"] is None
    output = tmp_path / "fixed-comparison.json"
    run.write_results(output, records)
    assert json.loads(output.read_text())["schema_version"] == 2


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_semantic_mismatch_preserves_translation(implementation):
    model = fixed_model(["return ltl_always(ap(reach, 'object_7'))"], implementation)
    record = run.run_cases(COMPARISON_CASES[:1], model, "offline/fixed", "fake", implementation=implementation)[0]
    assert record["outcome"] == "translated"
    assert record["failure"] is None
    assert record["semantic_evaluation"]["outcome"] == "semantic_mismatch"
    assert record["attempts"] == record["model_invocations"] == 1


def test_undeclared_proposition_is_valid_formula_data():
    model = fixed_model(["return ltl_eventually(ap(reach, 'undeclared_object'))"], "restricted")
    case = {**COMPARISON_CASES[0], "propositions": {}}
    record = run.run_cases([case], model, "offline/fixed", "fake")[0]
    assert record["outcome"] == "translated"
    assert record["semantic_evaluation"]["outcome"] == "semantic_mismatch"
    assert record["semantic_evaluation"]["atomic_propositions"] == ["reach(object_7)", "reach(undeclared_object)"]


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_invalid_reference_preserves_translation(implementation):
    case = {**COMPARISON_CASES[0], "reference_formula": "F ("}
    record = run.run_cases([case], fixed_model(BODIES[:1], implementation), "offline/fixed", "fake", implementation=implementation)[0]
    assert record["outcome"] == "translated"
    assert record["semantic_evaluation"]["outcome"] == "invalid_reference_formula"


@pytest.mark.parametrize("formula", ["[not a list]", "['EVENTUALLY']", "['p', 'q']"])
def test_untrusted_baseline_output_is_not_a_translation_or_checker_failure(monkeypatch, formula):
    from benchmarks.ltlcodegen_runner import runner

    monkeypatch.setattr(runner, "execute", lambda code: (formula, True))
    record = run.run_cases(COMPARISON_CASES[:1], fixed_model(BODIES[:1], "ltlcodegen-container"), "offline/fixed", "fake", implementation="ltlcodegen-container")[0]
    assert record["outcome"] == "translated"
    assert record["formula"] == formula
    assert record["semantic_evaluation"]["outcome"] == "invalid_generated_formula"


def test_actual_baseline_can_return_invalid_formula():
    if os.environ.get("LTL_DOCKER_TESTS") != "1":
        pytest.skip("Docker integration tests are opt-in")
    model = fixed_model(["return 'not an LTL formula'"], "ltlcodegen-container")
    record = run.run_cases(COMPARISON_CASES[:1], model, "offline/fixed", "fake", implementation="ltlcodegen-container")[0]
    assert record["outcome"] == "translated"
    assert record["semantic_evaluation"]["outcome"] == "invalid_generated_formula"


def test_failed_translation_does_not_initialize_checker(monkeypatch):
    monkeypatch.setattr(run, "evaluate_semantics", lambda *a, **kw: pytest.fail("Must not check a failed translation"))
    model = fixed_model(["return not_a_function()"], "restricted")
    record = run.run_cases(COMPARISON_CASES[:1], model, "offline/fixed", "fake")[0]
    assert record["outcome"] == "translation_error"
    assert record["semantic_evaluation"]["outcome"] == "not_run"


@pytest.mark.parametrize("outcome", ["checker_error", "checker_timeout", "semantic_mismatch"])
def test_cli_exit_status_remains_translation_based(monkeypatch, tmp_path, outcome):
    monkeypatch.setattr(run, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(run, "evaluate_semantics", lambda *a, **kw: {"outcome": outcome})
    output = tmp_path / "result.json"
    assert run.main(["--offline", "--output", str(output)]) == 0
    result = json.loads(output.read_text())["results"][0]
    assert result["outcome"] == "translated"
    assert result["semantic_evaluation"]["outcome"] == outcome


def test_missing_checker_keeps_translation_success(monkeypatch):
    monkeypatch.setenv("LTLFILT", "/no/such/checker")
    record = run.run_cases(COMPARISON_CASES[:1], fixed_model(BODIES[:1], "restricted"), "offline/fixed", "fake")[0]
    assert record["outcome"] == "translated"
    assert record["semantic_evaluation"]["outcome"] == "checker_error"


def test_semantic_details_are_redacted_in_saved_results(tmp_path):
    secret = "private_atom"
    case = {**COMPARISON_CASES[0], "reference_formula": "F reach(private_atom)"}
    model = fixed_model(["return ltl_eventually(ap(reach, 'private_atom'))"], "restricted")
    records = run.run_cases([case], model, "offline/fixed", "fake")
    assert records[0]["semantic_evaluation"]["outcome"] == "equivalent"
    assert secret in records[0]["semantic_evaluation"]["normalized_generated"]
    output = tmp_path / "redacted.json"
    run.write_results(output, records, secret)
    assert secret not in output.read_text()


def test_unexpected_evaluator_failure_does_not_replace_translation(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("must not persist this raw diagnostic")

    monkeypatch.setattr(run, "evaluate_semantics", broken)
    record = run.run_cases(COMPARISON_CASES[:1], fixed_model(BODIES[:1], "restricted"), "offline/fixed", "fake")[0]
    assert record["outcome"] == "translated"
    assert record["failure"] is None
    assert record["semantic_evaluation"]["outcome"] == "checker_error"
    assert "must not persist" not in json.dumps(record)
