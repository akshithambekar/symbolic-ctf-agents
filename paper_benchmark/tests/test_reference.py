import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_reference_hashes_and_raw_result_denominators():
    result = subprocess.run([sys.executable, str(ROOT / 'run.py'), 'verify'],
                            capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert data['verified']
    assert data['dataset_items'] == 40
    assert data['configuration']['model'] == 'nvidia/nemotron-3-ultra-550b-a55b:free'
    assert data['configuration']['trials'] == 3
    assert data['hash_counts']['original_manifest'] == 25
    for method in ('current', 'ltlcodegen'):
        assert data['methods'][method]['measured'] == 120
        assert data['methods'][method]['formula_reference_outputs'] == 96
    assert data['methods']['current']['semantic_correct_rate'] == 90 / 96
    assert data['methods']['ltlcodegen']['semantic_correct_rate'] == 69 / 96


def test_original_normalization_scoring_and_mocked_translation_in_isolation():
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(ROOT / 'reference_run/source')
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run([sys.executable, '-m', 'pytest', '-q',
                    str(ROOT / 'verification/legacy_cases.py')],
                   env=environment, check=True, capture_output=True, text=True)
