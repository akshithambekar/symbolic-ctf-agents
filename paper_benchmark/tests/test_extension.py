import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extension.audit import usage_of, finish_of, failure_bucket


def test_usage_and_finish_extract_both_saved_formats():
    usage = dict(prompt_tokens=10, completion_tokens=20, cost=0)
    assert usage_of({'usage': usage}) == usage
    assert usage_of({'provider_response': {'usage': usage}}) == usage
    assert usage_of({}) is None
    assert finish_of({'response': json.dumps({'choices': [{'finish_reason': 'length'}]})}) == 'length'
    assert finish_of({'provider_response': {'choices': [{'finish_reason': 'stop'}]}}) == 'stop'


def test_failures_distinguish_refusal_truncation_and_semantics():
    row = dict(infrastructure_error=False, assessment={'score': 'Invalid/unusable'},
               formula=None, reference_ltl='F(captured)', explicit_rejection=True, attempts=[])
    assert failure_bucket(row) == 'refused_formula_expected'
    row.update(explicit_rejection=False, attempts=[{'finish_reason': 'length'}])
    assert failure_bucket(row) == 'no_formula_output_limit'
    row.update(attempts=[{'code_error': 'Unsupported generated expression'}])
    assert failure_bucket(row) == 'no_formula_code_interpretation_failed'
    row.update(formula='G(in_bounds)', assessment=dict(score='Partially correct',syntax_valid=True,
        compilable=True,comparison={'left_only': None, 'right_only': [['captured']]}))
    assert failure_bucket(row) == 'strictly_stronger_than_reference'


def test_recovered_upstream_has_45_instructions_without_invented_gold():
    # Run separately because the upstream evaluator owns a non-CTF proposition profile.
    import subprocess
    code = ('from extension.upstream import parse_instructions; from pathlib import Path; '
            'items=parse_instructions(Path("paper_benchmark/extension/upstream_data/real_world_evaluation_prompts.yaml").read_text(encoding="utf-8")); '
            'assert len(items)==45; assert all(i["reference_ltl"] is None for i in items)')
    result = subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(0,"paper_benchmark"); '+code],
                            cwd=ROOT.parent, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
