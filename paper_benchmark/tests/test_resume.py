import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resume_matrix import completed, daily_quota_reset


def test_daily_quota_reset_requires_explicit_daily_limit(tmp_path):
    path = tmp_path / 'raw.jsonl'
    response = {'error': {'metadata': {'limit_source': 'openrouter_free_tier_daily',
                                     'headers': {'X-RateLimit-Reset': '1793836800000'}}}}
    path.write_text(json.dumps({'attempts': [{'response': json.dumps(response)}]}))
    assert daily_quota_reset(path) == 1793836800
    path.write_text(json.dumps({'attempts': [{'response': '{"error":{"code":429}}'}]}))
    assert daily_quota_reset(path) is None


def test_resume_preserves_unusable_but_retries_infrastructure(tmp_path):
    old = tmp_path / 'old/trial-0'
    old.mkdir(parents=True)
    base = dict(requested_model='model', method='ibm', item_id='a', infrastructure_error=False,
                assessment={'score': 'Invalid/unusable'})
    (old / 'raw.jsonl').write_text(json.dumps(base) + '\n' +
        json.dumps(dict(base, item_id='b', infrastructure_error=True)) + '\n', encoding='utf-8')
    smoke = tmp_path / 'old/smoke'
    smoke.mkdir()
    (smoke / 'raw.jsonl').write_text(json.dumps(dict(base, item_id='smoke')), encoding='utf-8')
    new = tmp_path / 'new/batch'
    new.mkdir(parents=True)
    (new / 'condition.json').write_text('{"matrix_trial": 0}')
    (new / 'raw.jsonl').write_text(json.dumps(dict(base, item_id='b')) + '\n{"partial":', encoding='utf-8')
    rows = completed([tmp_path / 'old', tmp_path / 'new'])
    assert set(rows) == {('model', 0, 'ibm', 'a'), ('model', 0, 'ibm', 'b')}
    assert rows[('model', 0, 'ibm', 'a')]['source'] == str(old / 'raw.jsonl')
