import json
import pytest
from maritime_nsrl.translator_benchmark.dataset import dataset
from maritime_nsrl.translator_benchmark.adapters import interpret, MAPPING
from maritime_nsrl.translator_benchmark.scoring import normalize_output, equivalence, score
from maritime_nsrl.temporal import compile_ltlf


def test_all_gold_references_compile_and_dataset_is_balanced():
    data=dataset()
    assert len(data['items'])==40
    assert len({r['id'] for r in data['items']})==40
    assert len({r['category'] for r in data['items']})==9
    for item in data['items']:
        if item['reference_ltl']:
            compile_ltlf(item['reference_ltl'])
            assert score(item,item['reference_ltl'])['score']=='Correct'


def test_nonidentical_equivalent_formulas():
    assert equivalence('G(!tagged)','!F(tagged)')['equivalent']
    assert equivalence('G(has_flag -> F(captured))','G(!has_flag | F(captured))')['equivalent']


def test_ordering_counterexample_and_strong_next():
    result=equivalence('F(has_flag & F(captured))','F(captured & F(has_flag))')
    assert not result['equivalent'] and result['left_only'] and result['right_only']
    assert not equivalence('F(has_flag & F(captured))','F(has_flag & X(F(captured)))')['equivalent']
    assert equivalence('X(in_bounds)','X(in_bounds)')['equivalent']


def test_prefix_normalization_and_unknown_object():
    assert normalize_output("['ALWAYS', 'NEGATION', 'reach(object_1)']",MAPPING)=='G(!(tagged))'
    with pytest.raises(ValueError): normalize_output("['EVENTUALLY','reach(object_99)']",MAPPING)
    with pytest.raises(ValueError): normalize_output("['EVENTUALLY']",MAPPING)


def test_restricted_codegen_uses_upstream_operators():
    raw='''```python
def question():
    a=ap(reach, "object_2")
    b=ap(reach, "object_3")
    return ltl_eventually(ltl_and(a, ltl_eventually(b)))
```'''
    formula=normalize_output(interpret(raw),MAPPING)
    assert equivalence(formula,'F(has_flag & F(captured))')['equivalent']


@pytest.mark.parametrize('code',[
    'import os\ndef question():\n return "captured"',
    'def question():\n return __import__("os").system("echo unsafe")',
    'def question():\n while True: pass',
    '@print("unsafe")\ndef question():\n return "captured"',
])
def test_generated_code_cannot_execute_arbitrary_python(code):
    with pytest.raises(ValueError): interpret(code)


def test_unsupported_ambiguous_and_infrastructure_are_separate():
    item=dataset()['items'][0]
    assert score(item,'G(invented)')['score']=='Invalid/unusable'
    assert score(item,infrastructure_error=True)['score']=='Not measured'
    ambiguous=next(r for r in dataset()['items'] if r['reference_ltl'] is None)
    assert score(ambiguous,explicit_rejection=True)['score']=='Correct'
    assert score(ambiguous,'F(home)')['score']=='Incorrect'
    assert score(item,'G(in_bounds) & F(captured)')['score']=='Partially correct'


def test_current_adapter_records_repairs_and_provider_model(monkeypatch):
    import requests
    from maritime_nsrl.translator_benchmark.adapters import call_backend
    raw=iter(['{"ltl":"F(invented)"}','{"ltl":"G(in_bounds)"}'])
    def post(*args,**kwargs):
        r=requests.Response(); r.status_code=200
        r._content=json.dumps(dict(model='resolved-test',choices=[dict(message=dict(content=next(raw)))])).encode()
        return r
    monkeypatch.setattr(requests,'post',post)
    result=call_backend('current',dataset()['items'][0],'mock','http://localhost')
    assert result['formula']=='G(in_bounds)' and result['repair_attempts']==1
    assert result['attempts'][0]['resolved_model']=='resolved-test'
    assert len(result['translation_artifact']['attempts'])==2
