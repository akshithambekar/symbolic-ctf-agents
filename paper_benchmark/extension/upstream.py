"""Translation-only validity evaluation on the recovered 45 upstream instructions.

No reference formulas are provided by this source. Semantic accuracy is not scored.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import use_frozen_source
use_frozen_source()
from maritime_nsrl.provider_config import load_env, api_key
from maritime_nsrl import logic, temporal
from maritime_nsrl.translator_benchmark.adapters import interpret
from maritime_nsrl.translator_benchmark.scoring import normalize_output
from comparison import redact

REVISION = 'e152ebd0d0b328f945e07b6c050106ce89bdb794'
URL = f'https://raw.githubusercontent.com/ExistentialRobotics/LTLCodeGen/{REVISION}/real_world_evaluation_prompts.yaml'
OBJECTS = ['chair', 'umbrella', 'backpack', 'person', 'potted plant', 'microwave', 'TV', 'laptop']


def parse_instructions(text):
    """Parse the source's restricted YAML shape without executing or importing it."""
    tier, items = None, []
    for number, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith('Tier ') and line.strip().endswith(':'):
            tier = line.strip()[:-1]
        elif line.strip().startswith('- '):
            instruction = json.loads(line.strip()[2:])
            if tier is None or not isinstance(instruction, str):
                raise ValueError('Malformed upstream tier/instruction')
            items.append(dict(id=f'upstream-{len(items):02}', tier=tier, instruction=instruction,
                              source_line=number, reference_ltl=None, semantic_score='Not available'))
    if Counter(i['tier'] for i in items) != {'Tier 1': 10, 'Tier 2': 10, 'Tier 3': 15, 'Tier 4': 10}:
        raise ValueError('Unexpected upstream dataset shape')
    return items


def run(output, source, env_file, model, limit=None):
    load_env(env_file)
    key = api_key('https://openrouter.ai/api/v1')
    if not key:
        raise ValueError('Missing API key')
    # Require an advertised zero-price text-generation route, not a stale page.
    catalog = requests.get('https://openrouter.ai/api/v1/models', timeout=30).json()
    match = next((m for m in catalog['data'] if m['id'] == model), None)
    if not match or any(float(match['pricing'][k]) != 0 for k in ('prompt', 'completion')):
        raise ValueError('Requested model is not a currently listed zero-price route')
    output.mkdir(parents=True, exist_ok=False)
    items = parse_instructions(source.read_text(encoding='utf-8'))
    if limit:
        items = items[:limit]
    dataset = dict(source_url=URL, source_revision=REVISION,
                   sha256=hashlib.sha256(source.read_bytes()).hexdigest(), items=items,
                   reference_formulas_available=False, review_complete=False)
    (output / 'dataset.json').write_text(json.dumps(dataset, indent=2), encoding='utf-8')
    template_path = ROOT / 'reference_run/source/maritime_nsrl/translator_benchmark/upstream/code_ltl_prompt_template.py'
    template = template_path.read_text()
    mapping = {f'object_{i}': f'object_{i}' for i in range(8)}
    # Separate process, separate proposition profile. Archived files stay unchanged.
    logic.AP = temporal.AP = frozenset(mapping)
    snapshot = output / 'source'
    snapshot.mkdir()
    for file in [Path(__file__), template_path,
                 ROOT / 'reference_run/source/maritime_nsrl/translator_benchmark/adapters.py',
                 ROOT / 'reference_run/source/maritime_nsrl/translator_benchmark/scoring.py',
                 ROOT / 'reference_run/source/maritime_nsrl/temporal.py',
                 ROOT / 'reference_run/source/maritime_nsrl/logic.py',
                 template_path.with_name('ltl_operators.py')]:
        shutil.copyfile(file, snapshot / file.name)
    config = dict(kind='exploratory-upstream-translation-only', model=model, trials=1,
                  implementation='upstream-prompt-restricted-ast-object-profile-v1',
                  request_cap=4, reference_formulas_available=False,
                  semantic_accuracy=None, objects=OBJECTS, provider_allow_fallbacks=False,
                  original_upstream_execution=False,
                  limitation='Validity diagnostic under LTLf; no reproduction of infinite-trace planner success or paper accuracy',
                  source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()},
                  started=datetime.now(timezone.utc).isoformat())
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    rows = []
    with (output / 'raw.jsonl').open('x', encoding='utf-8') as stream:
        for item in items:
            started = time.perf_counter()
            row = dict(item, method='ltlcodegen', requested_model=model, attempts=[], formula=None,
                       syntax_valid=False, compilable=False, infrastructure_error=False,
                       semantic_accuracy=None)
            instruction = ('Object vocabulary: ' + ', '.join(f'{name}=object_{i}' for i, name in enumerate(OBJECTS)) +
                           '. Use reach(object_i) for proximity to that mapped object.\nInstruction: ' + item['instruction'])
            previous = reason = ''
            for attempt_index in range(4):
                payload = dict(model=model, messages=[{'role': 'user', 'content': template.format(
                    instruction=instruction, previous_answer=previous, failure_reason=reason)}],
                    provider={'allow_fallbacks': False})
                attempt = {'request': payload}
                row['attempts'].append(attempt)
                try:
                    tick = time.perf_counter()
                    response = requests.post('https://openrouter.ai/api/v1/chat/completions',
                        headers={'Authorization': 'Bearer ' + key}, json=payload, timeout=120)
                    attempt.update(http_status=response.status_code, response=response.text,
                                   latency_seconds=time.perf_counter()-tick)
                    response.raise_for_status()
                    body = response.json()
                    raw = body['choices'][0]['message']['content']
                    attempt.update(usage=body.get('usage'), resolved_model=body.get('model'),
                                   raw_output=raw, finish_reason=body['choices'][0].get('finish_reason'))
                except requests.RequestException as exc:
                    row.update(infrastructure_error=True, error=type(exc).__name__)
                    break
                try:
                    formula = normalize_output(interpret(raw), mapping)
                    row.update(formula=formula, syntax_valid=True)
                    row['compilation'] = temporal.compile_ltlf(formula).report()
                    row['compilable'] = True
                    break
                except Exception as exc:
                    attempt['code_error'] = str(exc)
                    previous = '\nHere is your previous answer:\n' + str(raw)
                    reason = '\nBut it failed with the following error:\n' + str(exc)
            row['latency_seconds'] = time.perf_counter()-started
            row = redact(row, key)
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            summary = dict(scheduled=len(items), recorded=len(rows),
                measured=sum(not r['infrastructure_error'] for r in rows),
                syntax_valid=sum(r['syntax_valid'] for r in rows),
                compilable=sum(r['compilable'] for r in rows), semantic_accuracy=None,
                note='Instruction-only upstream set: human semantic review required')
            (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            print(item['id'], row['syntax_valid'], row['compilable'], flush=True)
            if row['infrastructure_error']:
                break
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='nvidia/nemotron-3-ultra-550b-a55b:free')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(args.output, args.source, args.env_file, args.model, args.limit)
