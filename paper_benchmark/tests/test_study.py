import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'study/manage.py'


def run(command, directory):
    return subprocess.run([sys.executable, str(SCRIPT), command, '--directory', str(directory)],
                          capture_output=True, text=True)


def test_candidate_integrity_and_pending_review():
    directory = ROOT / 'study/candidate_v1'
    result = run('check', directory)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['total_items'] == 120
    assert report['semantic_families'] == 60
    assert report['splits'] == {'development': 40, 'test_candidate': 80}
    assert report['historical_equivalence_overlap'] == 0
    assert report['review_complete'] is False
    assert report['new_model_requests'] == 0
    status = json.loads(run('review-status', directory).stdout)
    assert status['counts'] == {'pending': 120}
    assert not status['context_review_ready']
    assert not status['ready_for_adjudication_or_release_review']
    sources = json.loads((ROOT / 'study/sources.json').read_text(encoding='utf-8'))
    assert sources


def test_fresh_build_and_tamper_detection(tmp_path):
    directory = tmp_path / 'candidate'
    result = run('build', directory)
    assert result.returncode == 0, result.stderr
    assert run('build', directory).returncode != 0
    split = directory / 'splits.json'
    split.write_text('{}', encoding='utf-8')
    result = run('check', directory)
    assert result.returncode != 0
    assert 'changed without versioning' in result.stderr


def test_equivalent_reviews_do_not_imply_release(tmp_path):
    directory = tmp_path / 'candidate'
    assert run('build', directory).returncode == 0
    for label, formula in [('A', 'G(in_bounds)'), ('B', '!F(!in_bounds)')]:
        path = directory / f'reviewer_{label}.csv'
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        rows[0].update(reviewer_id=label, decision='formula', proposed_ltl=formula,
                       comments='Synthetic test fixture only.')
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    status = json.loads(run('review-status', directory).stdout)
    assert status['counts'] == {'agreement': 1, 'pending': 119}
    assert not status['ready_for_adjudication_or_release_review']
