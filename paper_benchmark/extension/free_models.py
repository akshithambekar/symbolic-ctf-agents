"""Archive a free-model inventory and verify advertised endpoint availability."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
GEMMA = ['google/gemma-4-26b-a4b-it:free', 'google/gemma-4-31b-it:free']
CANDIDATES = ['openai/gpt-oss-20b:free', 'openai/gpt-oss-120b:free',
              'qwen/qwen3-30b-a3b:free', 'qwen/qwen3-235b-a22b:free',
              'qwen/qwen3.6-plus-preview:free']


def discover(output):
    output.mkdir(parents=True, exist_ok=False)
    catalog = requests.get('https://openrouter.ai/api/v1/models', timeout=30).json()
    (output / 'catalog.json').write_text(json.dumps(catalog, indent=2), encoding='utf-8')
    free = [m for m in catalog['data'] if m['id'].endswith(':free') and
            'text' in m.get('architecture', {}).get('output_modalities', []) and
            all(float(m['pricing'][k]) == 0 for k in ('prompt', 'completion')) and
            m['id'] != 'openrouter/free']
    groups = defaultdict(list)
    for model in free:
        groups[model['id'].split('/')[0]].append(model['id'])
    endpoints = {}
    for model in GEMMA + CANDIDATES:
        response = requests.get('https://openrouter.ai/api/v1/models/' + model + '/endpoints', timeout=30)
        endpoints[model] = dict(http_status=response.status_code, body=response.json())
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), free_models=free,
                  provider_groups=dict(groups), endpoints=endpoints,
                  selected_pair=GEMMA,
                  matching_rule='Report total and active parameters separately; no exact parameter match claimed',
                  sizes={'nvidia_lightning': {'total_b': 30, 'active_b': 3},
                         'nvidia_super_vl': {'total_b': 120, 'active_b': 12},
                         'nvidia_ultra': {'total_b': 550, 'active_b': 55},
                         GEMMA[0]: {'total_b': 25.2, 'active_b': 3.8},
                         GEMMA[1]: {'total_b': 30.7, 'active_b': 30.7, 'architecture': 'dense'}},
                  second_family_status='Pending: require two confirmed free serving endpoints, not historical webpages')
    (output / 'selection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = discover(args.output)
    print(json.dumps(result['provider_groups'], indent=2))
