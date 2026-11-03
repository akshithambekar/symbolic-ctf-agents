import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ibm.adapter import call_backend, worker, WorkerUnavailable

MODEL = 'nvidia/nemotron-3-ultra-550b-a55b:free'
GLOSSARY = 'captured: team score; has_flag: flag possession; tagged: tagged boat; in_bounds: inside arena.'


def parse(text):
    return worker({'action': 'parse', 'instruction': 'Test instruction.',
                   'model': MODEL, 'glossary': GLOSSARY, 'response': text})


@pytest.mark.parametrize('text, expected', [
    ('PATTERN: existence\nSYMBOLS: captured', 'F(captured)'),
    ('PATTERN: absence\nSYMBOLS: tagged', '!(F(tagged))'),
    ('PATTERN: response\nSYMBOLS: has_flag, captured', 'G((has_flag) -> (F(captured)))'),
    ('PATTERN: chainResponse\nSYMBOLS: has_flag, captured', 'G((has_flag) -> (X(captured)))'),
    ('PATTERN: existenceTwo\nSYMBOLS: captured', 'F((captured) & (X(F(captured))))'),
])
def test_official_template_construction_and_strong_next(text, expected):
    assert parse(text)['formula'] == expected


def test_unknown_predicate_retained_for_common_validator():
    assert parse('PATTERN: existence\nSYMBOLS: invented')['formula'] == 'F(invented)'


def test_explicit_rejection_malformed_and_native_extra_symbol_behavior():
    assert parse('UNSUPPORTED: This intent needs clarification.')['explicit_rejection']
    assert parse('not a pattern')['formula'] is None
    result = parse('PATTERN: existence\nSYMBOLS: captured, has_flag')
    # IBM's unary grounder consumes the first symbol; preserve and expose this
    # behavior rather than silently inventing a conjunction of both goals.
    assert result['formula'] == 'F(captured)'
    assert result['symbols'] == ['captured', 'has_flag']


@pytest.mark.parametrize('budget', [200, 4096])
def test_native_request_with_same_model_and_no_gold(monkeypatch, budget):
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({'model': 'resolved-mock', 'choices': [
            {'message': {'content': 'PATTERN: existence\nSYMBOLS: captured'}}]}).encode()
        return response
    monkeypatch.setattr(requests, 'post', post)
    item = {'instruction': 'Eventually capture the enemy flag.', 'reference_ltl': 'F(captured)'}
    result = call_backend(item, MODEL, 'https://openrouter.ai/api/v1', 12, GLOSSARY, 'test-key', max_tokens=budget)
    assert result['formula'] == 'F(captured)'
    assert len(calls) == 1 and result['repair_attempts'] == 0
    payload = calls[0][1]['json']
    assert payload['model'] == MODEL and payload['max_tokens'] == budget
    assert result['native_generation_settings']['max_tokens'] == 200
    if budget != 200:
        assert result['implementation_id'].endswith('tokens-4096')
    assert payload['temperature'] == 0.5 and payload['stop'] == ['\n\n']
    assert item['reference_ltl'] not in payload['messages'][0]['content']
    assert result['attempts'][0]['resolved_model'] == 'resolved-mock'


def test_provider_failure_and_missing_worker_are_not_measured(monkeypatch, tmp_path):
    def post(*args, **kwargs):
        response = requests.Response()
        response.status_code = 429
        raise requests.HTTPError(response=response)
    monkeypatch.setattr(requests, 'post', post)
    result = call_backend({'instruction': 'Capture.'}, MODEL, 'https://openrouter.ai/api/v1',
                          10, GLOSSARY, 'test-key')
    assert result['infrastructure_error'] and result['attempts'][0]['http_status'] == 429
    assert len(result['attempts']) == 1
    with pytest.raises(WorkerUnavailable):
        worker({}, tmp_path / 'missing-python')


@pytest.mark.parametrize('candidate_dataset', [False, True])
def test_matched_three_way_driver_with_fake_http_and_frozen_scorer(tmp_path, candidate_dataset):
    requests_seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests_seen.append(data)
            text = json.dumps(data)
            if 'CTF DOMAIN ADAPTATION' in text:
                answer = 'PATTERN: existence\nSYMBOLS: captured'
            elif 'question' in text:
                answer = 'def question():\n return ltl_eventually(ap(reach, "object_3"))'
            else:
                answer = '{"ltl":"F(captured)"}'
            body = json.dumps({'model': 'resolved-mock', 'diagnostic': 'test-key',
                               'choices': [{'message': {'content': answer}}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = {**os.environ, 'LTL_API_KEY': 'test-key'}
    output = tmp_path / 'comparison'
    try:
        extra = ['--dataset', str(ROOT / 'study/candidate_v1/dataset.json')] if candidate_dataset else []
        subprocess.run([sys.executable, str(ROOT / 'run.py'), 'compare', '--trials', '1',
                        '--model', MODEL, '--endpoint', f'http://127.0.0.1:{server.server_port}/v1',
                        '--case-id', 'dev-capture-0' if candidate_dataset else 'capture-0', '--env-file', str(tmp_path / 'missing.env'),
                        '--output', str(output)] + extra, check=True, capture_output=True, text=True, env=env)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    rows = [json.loads(line) for line in (output / 'raw.jsonl').read_text().splitlines()]
    assert {r['method'] for r in rows} == {'current', 'ltlcodegen', 'ibm'}
    assert all(r['assessment']['score'] == 'Correct' for r in rows)
    assert len(requests_seen) == 3
    assert 'test-key' not in (output / 'raw.jsonl').read_text()
    assert (output / 'human_review.csv').exists()
    assert (output / 'source/comparison_adapter/ibm/upstream/prompt.json').exists()
    hashes = json.loads((output / 'source_snapshot_sha256.json').read_text())
    assert 'source/maritime_nsrl/translation.py' in hashes
    assert 'source/comparison_adapter/ibm/worker.py' in hashes
    assert 'dataset.json' in hashes
    summary = json.loads((output / 'summary.json').read_text())
    assert len(summary) == 3 and all(s['measured'] == 1 for s in summary)
