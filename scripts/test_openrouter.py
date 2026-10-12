"""Run from normal PowerShell to test both translator paths, not the full suite."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid
import sys

# Translator-only smoke tests can run in an isolated interpreter without the
# simulator/PPO dependencies or an editable project installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from maritime_nsrl.provider_config import load_env, api_key
from maritime_nsrl.translator_benchmark.adapters import call_backend
from maritime_nsrl.translator_benchmark.dataset import dataset
from maritime_nsrl.translator_benchmark.scoring import score


def main():
    root = Path(__file__).resolve().parents[1]
    load_env(root / '.env')
    endpoint = os.environ.get('LTL_BASE_URL', 'https://openrouter.ai/api/v1')
    model = os.environ.get('OPENROUTER_MODEL')
    key = api_key(endpoint)
    if not key or not model:
        raise SystemExit('Save OPENROUTER_API_KEY and OPENROUTER_MODEL in .env first.')
    output = root.parent / ('openrouter-smoke-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    output.mkdir(exist_ok=False)
    item = dataset()['items'][0]
    old_key = os.environ.get('LTL_API_KEY')
    os.environ['LTL_API_KEY'] = key
    try:
        for method in ('current', 'ltlcodegen'):
            print(f'Testing {method} with {model}; each request may take up to 120 seconds...', flush=True)
            result = call_backend(method, item, model, endpoint, timeout=120)
            result['assessment'] = score(item, result['formula'], result['explicit_rejection'], result['infrastructure_error'])
            result['input'] = item
            # No credentials are intentionally logged; redact defensively in errors.
            serialized = json.dumps(result, indent=2).replace(key, '[REDACTED]')
            (output / f'{method}.json').write_text(serialized, encoding='utf-8')
            print('Result:', result['assessment']['score'], flush=True)
            print('Formula:', result['formula'], flush=True)
            if result['error']:
                print('Error:', result['error'].replace(key, '[REDACTED]'), flush=True)
            if result['infrastructure_error']:
                break
    finally:
        if old_key is None:
            os.environ.pop('LTL_API_KEY', None)
        else:
            os.environ['LTL_API_KEY'] = old_key
    print('Results saved to:', output, flush=True)


if __name__ == '__main__':
    main()
