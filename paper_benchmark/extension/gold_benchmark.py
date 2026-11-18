"""Reference-scored Drone/Cleanup/Pick pilot with Spot infinite-trace equivalence."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))
from run import use_frozen_source
use_frozen_source()
from benchmarks.formula_notation import infix_tree, to_spot
from maritime_nsrl.provider_config import load_env, api_key
from maritime_nsrl.translator_benchmark.adapters import interpret
from maritime_nsrl.translator_benchmark.scoring import normalize_output
from ibm.adapter import call_backend as ibm_call
from comparison import redact

REVISION = '7c778144c6bfcfecd14b3df5fbb6a40d2e15fd17'
MODELS = ['nvidia/nemotron-3-ultra-550b-a55b:free', 'nvidia/nemotron-3.5-super-vl:free']
DOMAIN_SIZES = {'drone-planning': 20, 'clean-up': 20, 'pick-and-place': 5}


def native_tree(raw, domain):
    if domain == 'drone-planning':
        def convert(node):
            return ('ap', node.atom) if node.operator == 'atom' else (node.operator, *(convert(n) for n in node.operands))
        return convert(infix_tree(raw.replace('~', '!')))
    tokens = iter(raw.split())
    def parse():
        token = next(tokens)
        # Cleanup uses X as a native proposition (blue room with chair),
        # not the temporal next operator. Preserve that domain distinction.
        arity = {'F': 1, 'G': 1, '!': 1, '&': 2, '|': 2, 'U': 2, '->': 2}.get(token)
        return (token, *(parse() for _ in range(arity))) if arity else ('ap', token)
    result = parse()
    if next(tokens, None) is not None:
        raise ValueError('Unconsumed reference prefix tokens')
    return result


def atoms(tree):
    return {tree[1]} if tree[0] == 'ap' else set().union(*(atoms(c) for c in tree[1:]))


def render(tree, mapping):
    if tree[0] == 'ap':
        return mapping[tree[1]]
    if len(tree) == 2:
        return f'{tree[0]}({render(tree[1], mapping)})'
    return f'({render(tree[1], mapping)}) {tree[0]} ({render(tree[2], mapping)})'


def prepare(source, output):
    rng = random.Random(20261117)
    datasets, all_items = {}, []
    for domain, count in DOMAIN_SIZES.items():
        path = source / (domain + '.jsonl')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            tree = native_tree(row['raw_ltl'], domain)
            groups[row['raw_ltl']].append((index, row, tree))
        names = sorted(set().union(*(atoms(g[0][2]) for g in groups.values())))
        mapping = {name: f'p{i}' for i, name in enumerate(names)}
        if domain == 'drone-planning':
            meanings = {n: n.replace('_', ' ') for n in names}
        elif domain == 'clean-up':
            meanings = {'B': 'go to the blue room', 'C': 'go to the green room',
                        'R': 'go to the red room', 'Y': 'go to the yellow room',
                        'X': 'go to the blue room with chair', 'Z': 'go to the green room with chair'}
        else:
            meanings = {'A': 'any cubes', 'B': 'any non blue cubes', 'C': 'any non green cubes',
                        'R': 'any non red cubes', 'S': 'scan', 'Y': 'any non yellow cubes'}
        glossary = '\n'.join(f'{mapping[n]}: {meanings[n]} (source symbol {n})' for n in names)
        datasets[domain] = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            original_items=len(rows), reference_string_groups=len(groups), sample_count=count,
            mapping=mapping, glossary=glossary)
        selected = rng.sample(sorted(groups), count)
        for group in selected:
            index, row, tree = rng.choice(groups[group])
            all_items.append(dict(id=f'{domain}-{index}', domain=domain, source_row=index,
                instruction=row['natural'], original_reference=row['raw_ltl'],
                reference=render(tree, mapping), glossary=glossary,
                allowed=list(mapping.values())))
    value = dict(version='gold-domain-pilot-v1', source_revision=REVISION,
                 seed=20261117, semantics='standard LTL over infinite traces',
                 sampling='one random instruction per sampled raw reference string group',
                 domains=datasets, items=all_items)
    (output / 'dataset.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    return value


def linux_path(path):
    path = path.resolve()
    return '/mnt/' + path.drive[0].lower() + path.as_posix()[2:]


def check(generated, reference):
    result = subprocess.run(['wsl', '-d', 'Ubuntu', '--', 'python3',
        linux_path(ROOT / 'extension/spot_bridge.py')],
        input=json.dumps(dict(generated=generated, reference=reference)),
        capture_output=True, text=True, encoding='utf-8', timeout=30)
    if result.returncode:
        raise RuntimeError('Spot bridge failed: ' + result.stderr[-600:])
    checked = json.loads(result.stdout)
    if checked['outcome'] in ('checker_error', 'checker_timeout', 'invalid_reference_formula'):
        raise RuntimeError('Semantic checker unavailable or invalid reference: ' + str(checked))
    return checked


def translate(item, method, model, key):
    if method == 'ibm':
        result = ibm_call(dict(instruction=item['instruction']), model, 'https://openrouter.ai/api/v1',
            120, item['glossary'], key, max_tokens=4096,
            domain_profile=dict(domain_name=item['domain'], allowed_symbols=item['allowed']))
        return result
    started = time.perf_counter()
    result = dict(formula=None, attempts=[], infrastructure_error=False, explicit_rejection=False)
    glossary = (item['glossary'] + '\nUse standard LTL over INFINITE traces, not finite episodes. '
                'Return an error for unsupported meaning. Do not add unstated obligations.')
    system = ('Translate the instruction into JSON {"ltl":"formula"}. No Markdown. '
              'Use ! & | -> X U F G and only the supplied proposition identifiers.\n' + glossary)
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': item['instruction']}]
    template = (ROOT / 'reference_run/source/maritime_nsrl/translator_benchmark/upstream/code_ltl_prompt_template.py').read_text()
    mapping = {f'object_{i}': p for i, p in enumerate(item['allowed'])}
    previous = reason = ''
    for index in range(3 if method == 'current' else 4):
        if method == 'ltlcodegen':
            context = glossary + '\nObject mapping (reach means truth of the mapped predicate): ' + json.dumps(mapping)
            prompt = template.format(instruction=context + '\nInstruction: ' + item['instruction'],
                                     previous_answer=previous, failure_reason=reason)
            payload = dict(model=model, messages=[{'role': 'user', 'content': prompt}])
        else:
            payload = dict(model=model, messages=messages.copy())
        attempt = {'request': payload}
        result['attempts'].append(attempt)
        try:
            tick = time.perf_counter()
            response = requests.post('https://openrouter.ai/api/v1/chat/completions',
                headers={'Authorization': 'Bearer ' + key}, json=payload, timeout=120)
            attempt.update(http_status=response.status_code,response=response.text,
                           latency_seconds=time.perf_counter()-tick)
            response.raise_for_status()
            body = response.json()
            raw = body['choices'][0]['message']['content']
            attempt.update(usage=body.get('usage'), resolved_model=body.get('model'),
                raw_output=raw, finish_reason=body['choices'][0].get('finish_reason'))
        except requests.RequestException as exc:
            result.update(infrastructure_error=True,error=type(exc).__name__)
            break
        try:
            if method == 'current':
                data = json.loads(raw)
                if 'error' in data:
                    result.update(explicit_rejection=True, error=data['error'])
                    break
                formula = data['ltl']
            else:
                formula = normalize_output(interpret(raw), mapping)
            _, names = to_spot(formula, 'infix')
            if set(names) - set(item['allowed']):
                raise ValueError('Unsupported predicates: ' + str(set(names)-set(item['allowed'])))
            validation = check(formula, formula)
            if validation['outcome'] != 'equivalent':
                raise ValueError('Invalid formula')
            result.update(formula=formula, validation=validation)
            break
        except (ValueError, KeyError, TypeError) as exc:
            attempt['validation_error'] = str(exc)
            if method == 'ltlcodegen':
                previous = '\nPrevious answer:\n' + str(raw)
                reason = '\nFailure:\n' + str(exc)
            else:
                messages += [{'role': 'assistant','content': str(raw)},
                             {'role': 'user','content': 'Correct the JSON/formula: '+str(exc)}]
    result.update(latency_seconds=time.perf_counter()-started,
                  repair_attempts=max(0,len(result['attempts'])-1))
    return result


def run(args):
    load_env(args.env_file)
    key = api_key('https://openrouter.ai/api/v1')
    if not key:
        raise ValueError('Missing API key')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog = requests.get('https://openrouter.ai/api/v1/models',timeout=30).json()
    for model in MODELS:
        record = next(m for m in catalog['data'] if m['id'] == model)
        if any(float(record['pricing'][k]) for k in ('prompt','completion')):
            raise ValueError('Selected model is not free')
    dataset = prepare(args.source, output)
    (output/'status.json').write_text(json.dumps(dict(status='validating_references',scheduled=270)))
    # No live translation starts until every gold reference parses and passes Spot.
    checks = [check(i['reference'],i['reference']) for i in dataset['items']]
    (output / 'reference_validation.json').write_text(json.dumps(checks,indent=2))
    config = dict(kind='exploratory-reference-scored-pilot',models=MODELS,
        methods=['current','ltlcodegen','ibm'],trials=1,items=45,scheduled=270,
        semantics='infinite-trace LTL',checker_version=checks[0]['checker_version'],
        reference_formulas_sent_to_model=False, ibm_max_tokens=4096,
        adaptations='Domain glossary and bijective proposition renaming; native workflows have different repair caps',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),ROOT/'extension/spot_bridge.py',ROOT/'ibm/worker.py',
                      ROOT/'ibm/adapter.py',REPO/'benchmarks/formula_notation.py',REPO/'benchmarks/semantics.py']})
    (output / 'config.json').write_text(json.dumps(config,indent=2))
    snapshot = output/'source';snapshot.mkdir()
    for file in [Path(__file__),ROOT/'extension/spot_bridge.py',ROOT/'ibm/worker.py',
                 ROOT/'ibm/adapter.py',REPO/'benchmarks/formula_notation.py',REPO/'benchmarks/semantics.py']:
        shutil.copyfile(file,snapshot/file.name)
    rows=[]
    for previous in args.previous:
        prior_data=json.loads((previous/'dataset.json').read_text())
        if prior_data != dataset:
            raise ValueError('Cannot resume a different pilot dataset')
        for line in (previous/'raw.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if not row['infrastructure_error']:
                identity=(row['requested_model'],row['method'],row['item_id'])
                if not any((r['requested_model'],r['method'],r['item_id'])==identity for r in rows):
                    rows.append(dict(row,source_file=str(previous/'raw.jsonl')))
    retained={(r['requested_model'],r['method'],r['item_id']) for r in rows}
    (output/'status.json').write_text(json.dumps(dict(status='running',recorded=len(rows),scheduled=270)))
    with (output/'raw.jsonl').open('x',encoding='utf-8') as stream:
        for index,item in enumerate(dataset['items']):
            for model in MODELS:
                methods=['current','ltlcodegen','ibm'];offset=index%3
                for method in methods[offset:]+methods[:offset]:
                    if (model,method,item['id']) in retained:
                        continue
                    result=translate(item,method,model,key)
                    assessment={'outcome':'not_translated'}
                    if result['infrastructure_error']:
                        assessment={'outcome':'not_measured'}
                    elif result.get('formula'):
                        _, names=to_spot(result['formula'],'infix')
                        assessment=(check(result['formula'],item['reference']) if set(names)<=set(item['allowed'])
                                    else {'outcome':'unsupported_predicates'})
                    row=dict(result)
                    row.update(item_id=item['id'],domain=item['domain'],method=method,
                        requested_model=model,raw_input=item['instruction'],reference=item['reference'],
                        original_reference=item['original_reference'],assessment=assessment)
                    row=redact(row,key)
                    rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                    summary=[]
                    for domain in DOMAIN_SIZES:
                        for mm in MODELS:
                            for mt in methods:
                                group=[r for r in rows if r['domain']==domain and r['requested_model']==mm and r['method']==mt]
                                if not group:continue
                                measured=[r for r in group if not r['infrastructure_error']]
                                summary.append(dict(domain=domain,model=mm,method=mt,recorded=len(group),
                                    measured=len(measured),equivalent=sum(r['assessment']['outcome']=='equivalent' for r in measured)))
                    (output/'summary.json').write_text(json.dumps(summary,indent=2))
                    (output/'status.json').write_text(json.dumps(dict(status='running',recorded=len(rows),scheduled=270)))
                    print(model,method,item['id'],assessment['outcome'],flush=True)
                    if result['infrastructure_error']:
                        # Pause visibly rather than silently converting quota failures into quality zeros.
                        (output/'status.json').write_text(json.dumps(dict(status='incomplete_infrastructure',
                            recorded=len(rows),scheduled=270),indent=2))
                        return
    (output/'status.json').write_text(json.dumps(dict(status='complete',recorded=len(rows),scheduled=270),indent=2))
    with (output/'combined_measured.jsonl').open('x',encoding='utf-8') as stream:
        for row in rows:
            if not row['infrastructure_error']:
                stream.write(json.dumps(row)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--env-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path,action='append',default=[])
    run(parser.parse_args())
