"""CTF/OpenRouter adaptation of IBM NL2LTL 0.0.6, using a separate interpreter."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_ID = 'ibm-nl2ltl-0.0.6-ctf-openrouter-v1'


def implementation_id(max_tokens=200):
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 16384:
        raise ValueError('IBM max tokens must be an integer in 1..16384')
    return IMPLEMENTATION_ID if max_tokens == 200 else f'{IMPLEMENTATION_ID}-tokens-{max_tokens}'


class WorkerUnavailable(RuntimeError):
    pass


def default_python():
    return ROOT.parent / '.venv-ibm' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def worker(data, python=None):
    executable = Path(python or default_python())
    if not executable.is_file():
        raise WorkerUnavailable('IBM Python environment is missing; see paper_benchmark/ibm/README.md')
    environment = {key: value for key, value in os.environ.items()
                   if key in {'PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'LD_LIBRARY_PATH'}}
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    try:
        result = subprocess.run([str(executable.resolve()), str(ROOT / 'ibm/worker.py')],
                                input=json.dumps(data), capture_output=True, text=True,
                                encoding='utf-8', timeout=20, env=environment)
        if result.returncode or len(result.stdout) > 131072:
            raise WorkerUnavailable('IBM worker failed; check its pinned environment')
        parsed = json.loads(result.stdout)
        if not isinstance(parsed, dict):
            raise WorkerUnavailable('Invalid IBM worker result')
        return parsed
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise WorkerUnavailable(f'IBM worker could not run: {type(exc).__name__}') from exc


def call_backend(item, model, endpoint, timeout, glossary, api_key, python=None, max_tokens=200):
    start = time.perf_counter()
    result = dict(method='ibm', implementation_id=implementation_id(max_tokens),
                  formula=None, attempts=[], repair_attempts=0, latency_seconds=None,
                  infrastructure_error=False, error=None, explicit_rejection=False,
                  requested_model=model, endpoint=endpoint)
    try:
        data = {'action': 'request', 'instruction': item['instruction'],
                'model': model, 'glossary': glossary}
        built = worker(data, python)
        if 'request' not in built:
            raise WorkerUnavailable('IBM engine could not construct a request: ' + built.get('error', 'unknown error'))
        result['native_generation_settings'] = {k: built['request'][k] for k in ('temperature', 'max_tokens', 'stop')}
        payload = {**built['request'], 'max_tokens': max_tokens}
        attempt = {'request': payload, 'response': None, 'resolved_model': None,
                   'error': None, 'latency_seconds': None}
        result['attempts'].append(attempt)
        tick = time.perf_counter()
        try:
            response = requests.post(endpoint.rstrip('/') + '/chat/completions',
                                     headers={'Authorization': 'Bearer ' + api_key},
                                     json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            attempt['provider_response'] = body
            attempt['resolved_model'] = body.get('model')
            attempt['finish_reason'] = body['choices'][0].get('finish_reason')
            raw = body['choices'][0]['message']['content']
            if not isinstance(raw, str):
                raise ValueError('Provider returned no final text; finish_reason=' + str(attempt['finish_reason']))
            attempt['response'] = raw
        except requests.RequestException as exc:
            attempt['error'] = type(exc).__name__
            attempt['http_status'] = exc.response.status_code if exc.response is not None else None
            if exc.response is not None:
                attempt['provider_error_response'] = exc.response.text
            result.update(infrastructure_error=True, error=type(exc).__name__)
            return result
        finally:
            attempt['latency_seconds'] = time.perf_counter() - tick
        interpreted = worker({**data, 'action': 'parse', 'response': raw}, python)
        result.update(ibm_output=interpreted, formula=interpreted.get('formula'),
                      explicit_rejection=interpreted.get('explicit_rejection', False),
                      error=interpreted.get('error'))
    except WorkerUnavailable as exc:
        result.update(infrastructure_error=True, error=str(exc))
    except (ValueError, KeyError, TypeError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        result['latency_seconds'] = time.perf_counter() - start
    return result


def source_hashes():
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / 'ibm').rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts}
