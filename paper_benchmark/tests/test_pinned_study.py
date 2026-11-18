import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pinned_study.run import Budget, normalize, reserve_cost, validate_predicates
from pinned_study.freeze import normalized_instruction, extract_template


def test_prefix_and_equality():
    mapping={'object_0':'p0','object_1':'p1'}
    assert normalize("['EVENTUALLY','reach(object_0)']",'paper_direct',mapping)=='F(p0)'
    assert normalize("['EQUAL','reach(object_0)','reach(object_1)']",'paper_direct',mapping)=='((p0) -> (p1)) & ((p1) -> (p0))'
    for raw in ["__import__('os').system('bad')", "['AND','reach(object_0)']", "['reach(object_99)']"]:
        with pytest.raises((ValueError,SyntaxError)):
            normalize(raw,'paper_direct',mapping)


def test_budget_reserves_before_call_and_accounts_missing_usage(tmp_path):
    ledger=Budget(tmp_path/'budget.json',.1)
    messages=[dict(role='user',content='test')]
    amount=ledger.reserve(messages)
    assert json.loads(ledger.path.read_text())['pending_reservation_usd']==amount
    ledger.settle(None)
    assert ledger.accounted==amount
    ledger.reserve(messages)
    ledger.settle({'cost':.001})
    assert ledger.accounted==pytest.approx(amount+.001)
    with pytest.raises(RuntimeError,match='budget_exhausted'):
        ledger.reserve([dict(role='user',content='x'*100000)])


def test_rejects_unbounded_spend(tmp_path):
    for limit in [0,6,float('nan'),float('inf')]:
        with pytest.raises(ValueError):Budget(tmp_path/'budget.json',limit)

def test_validates_proposition_list():
    validate_predicates('F(p0)', ['p0'])
    with pytest.raises(ValueError,match='Unsupported predicates'):
        validate_predicates('G(p1)', ['p0'])


def test_freeze_instruction_identity_and_static_source():
    assert normalized_instruction('Go to BLUE room.')==normalized_instruction('go to blue room!')
    source='def get_prompt():\n    template="safe"\n    return template\n'
    assert extract_template(source,'get_prompt')=='safe'


def test_frozen_dataset_if_available():
    path=Path(__file__).resolve().parents[1]/'runs/gpt4o-frozen-1118-v1/frozen.json'
    if not path.exists():pytest.skip('Private downloaded dataset is not bundled')
    data=json.loads(path.read_text(encoding='utf-8'))
    assert len(data['items'])==90
    for domain,count in data['example_counts'].items():
        assert len(data['examples'][domain])==count
        banned={normalized_instruction(e['instruction']) for e in data['examples'][domain]}
        assert not any(normalized_instruction(e['instruction']) in banned for e in data['items'] if e['domain']==domain)
    assert data['exact_authors_sample'] is False
