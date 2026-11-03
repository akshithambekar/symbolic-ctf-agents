"""Fresh, explicitly labeled three-translator comparisons using the paper scorer."""
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import itertools
import json
import os
from pathlib import Path
import shutil
import statistics

from ibm.adapter import IMPLEMENTATION_ID, call_backend as call_ibm, source_hashes, implementation_id
from maritime_nsrl.provider_config import api_key
from maritime_nsrl.translator_benchmark.adapters import call_backend
from maritime_nsrl.translator_benchmark.dataset import GLOSSARY
from maritime_nsrl.translator_benchmark.scoring import equivalence, score

ROOT = Path(__file__).resolve().parent
IDS = {'current': 'paper-direct-json-frozen-20260926',
       'ltlcodegen': 'paper-ctf-codegen-frozen-20260926', 'ibm': IMPLEMENTATION_ID}


def redact(value, key):
    if isinstance(value, str):
        return value.replace(key, '[REDACTED]') if key else value
    if isinstance(value, list):
        return [redact(item, key) for item in value]
    if isinstance(value, dict):
        return {name: redact(item, key) for name, item in value.items()}
    return value


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def summaries(output, rows, methods):
    results, pairs = [], []
    mean = lambda values: statistics.mean(values) if values else None
    for method in methods:
        chosen = [r for r in rows if r['method'] == method]
        measured = [r for r in chosen if not r['infrastructure_error']]
        formulas = [r for r in measured if r['reference_ltl'] is not None]
        rejection_items = [r for r in measured if r['reference_ltl'] is None]
        for trial in sorted({r['trial'] for r in chosen}):
            for group in sorted({r['paraphrase_group'] for r in chosen}):
                group_rows = [r for r in chosen if r['trial'] == trial and r['paraphrase_group'] == group]
                for a, b in itertools.combinations(group_rows, 2):
                    pair = {'method': method, 'trial': trial, 'group': group, 'equivalent': None}
                    if a['formula'] and b['formula']:
                        try:
                            pair.update(equivalence(a['formula'], b['formula']))
                        except Exception as exc:
                            pair['error'] = str(exc)
                    else:
                        pair['error'] = 'A formula is missing; excluded from pair equivalence denominator'
                    pairs.append(pair)
        valid_pairs = [p for p in pairs if p['method'] == method and p['equivalent'] is not None]
        results.append({
            'method': method, 'implementation_id': chosen[0]['implementation_id'], 'scheduled': len(chosen),
            'measured': len(measured), 'not_measured': len(chosen) - len(measured),
            'formula_reference_outputs': len(formulas), 'rejection_reference_outputs': len(rejection_items),
            'syntax_valid_rate': mean([float(r['assessment']['syntax_valid']) for r in formulas]),
            'compilable_rate': mean([float(r['assessment']['compilable']) for r in formulas]),
            'semantic_correct_rate': mean([float(r['assessment']['score'] == 'Correct') for r in formulas]),
            'unsupported_predicate_rate': mean([float(bool(r['assessment']['unsupported_predicates'])) for r in formulas]),
            'appropriate_rejection_rate': mean([float(r['assessment']['score'] == 'Correct') for r in rejection_items]),
            'repair_rate': mean([float(r['repair_attempts'] > 0) for r in measured]),
            'mean_latency_seconds': mean([r['latency_seconds'] for r in measured]),
            'provider_requests': sum(len(r['attempts']) for r in chosen),
            'paraphrase_equivalence_rate': mean([float(p['equivalent']) for p in valid_pairs]),
            'paraphrase_pairs_compared': len(valid_pairs)})
    write(output / 'summary.json', results)
    write(output / 'paraphrases.json', pairs)
    with (output / 'summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (output / 'human_review.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ['item_id', 'method', 'trial', 'raw_input', 'reference_ltl', 'formula',
                  'automatic_score', 'reviewer_score', 'reviewer_rationale', 'reference_approved']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in fields if k in row},
                             'automatic_score': row['assessment']['score']})
    lines = ['# New exploratory translator comparison', '',
             'These are new calls, separate from the preserved September results.',
             'No historical rows were pooled into this run.', '',
             'All methods use the same inputs, requested model, CTF glossary, and frozen LTLf scorer.',
             'Native workflow call caps differ: direct JSON 3; CTF codegen 4; IBM 1.',
             'IBM therefore has no validation repair loop; zero repairs is not evidence of better quality.',
             'IBM uses temperature 0.5 and native stop settings; its explicit token cap and variant ID are recorded in config.json. Native cap is 200.',
             'This compares adapted workflows, not isolated prompts at equal call/token budgets.',
             'Gold interpretations require independent approval. Syntax and satisfiability do not prove English meaning.', '',
             '| Method | Measured/scheduled | Formula-reference outputs | Correct/reference |',
             '| --- | --- | --- | --- |']
    for row in results:
        correct = sum(r['assessment']['score'] == 'Correct' for r in rows
                      if r['method'] == row['method'] and not r['infrastructure_error']
                      and r['reference_ltl'] is not None)
        lines.append(f"| {row['method']} | {row['measured']}/{row['scheduled']} | "
                     f"{row['formula_reference_outputs']} | {correct}/{row['formula_reference_outputs']} |")
    lines += ['', 'Read raw.jsonl, counterexamples, model metadata, and the human review sheet before interpretation.',
              'Subset/smoke runs cannot establish translator superiority. No automatic quality ranking is made.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, row in enumerate(results):
        value = row['semantic_correct_rate']
        if value is not None:
            ax.bar(i, value)
        else:
            ax.text(i, .05, 'Not measured', ha='center')
    ax.set(xticks=range(len(results)), xticklabels=methods, ylim=(0, 1),
           ylabel='Agreement with project reference (eligible measured outputs)',
           title='Exploratory run — inspect counts and uncertainty before comparison')
    fig.tight_layout()
    fig.savefig(output / 'semantic_agreement.png')
    plt.close(fig)
    return results


def run(output, *, methods, model, endpoint, trials, timeout, case_ids=None, ibm_python=None,
        ibm_max_tokens=200):
    if trials < 1 or timeout <= 0:
        raise ValueError('Trials and timeout must be positive')
    if not methods or len(set(methods)) != len(methods) or set(methods) - IDS.keys():
        raise ValueError('Select distinct known methods')
    actual_ids = {**IDS, 'ibm': implementation_id(ibm_max_tokens)}
    key = api_key(endpoint)
    if not key:
        raise ValueError('Missing API key for selected endpoint')
    data = json.loads((ROOT / 'reference_run/dataset.json').read_text())
    if case_ids:
        unknown = set(case_ids) - {item['id'] for item in data['items']}
        if unknown:
            raise ValueError(f'Unknown case IDs: {sorted(unknown)}')
        data['items'] = [item for item in data['items'] if item['id'] in case_ids]
    output.mkdir(exist_ok=False)
    write(output / 'dataset.json', data)
    write(output / 'config.json', {
        'experiment_kind': 'exploratory', 'methods': methods, 'implementation_ids': actual_ids,
        'model': model, 'endpoint': endpoint, 'trials': trials, 'timeout_seconds': timeout,
        'benchmark_version': data['version'], 'case_ids': [i['id'] for i in data['items']],
        'semantics': 'Nonempty finite traces; strong X; inclusive F',
        'request_caps': {'current': 3, 'ltlcodegen': 4, 'ibm': 1},
        'ibm_native_request': {'temperature': 0.5, 'max_tokens': 200, 'top_p': 1.0, 'stop': ['\n\n']},
        'ibm_actual_max_tokens': ibm_max_tokens,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'ibm_python': str(ibm_python) if ibm_python else 'default isolated .venv-ibm',
        'historical_results_pooled': False, 'gold_review': 'Independent human approval pending'})
    paths = [Path(__file__).resolve(), ROOT / 'run.py']
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    hashes.update(source_hashes())
    write(output / 'adapter_source_hashes.json', hashes)
    write(output / 'dependencies.json', {d.metadata['Name']: d.version for d in distributions()})
    snapshot = output / 'source'
    shutil.copytree(ROOT / 'reference_run/source', snapshot, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT / 'ibm', snapshot / 'comparison_adapter/ibm', ignore=shutil.ignore_patterns('__pycache__'))
    for path in paths:
        shutil.copyfile(path, snapshot / 'comparison_adapter' / path.name)
    unavailable, rows = {}, []
    old_key = os.environ.get('LTL_API_KEY')
    os.environ['LTL_API_KEY'] = key
    try:
        with (output / 'raw.jsonl').open('x', encoding='utf-8') as stream:
            for trial in range(trials):
                for index, item in enumerate(data['items']):
                    offset = (trial + index) % len(methods)
                    for method in methods[offset:] + methods[:offset]:
                        if method in unavailable:
                            result = dict(formula=None, attempts=[], repair_attempts=0,
                                          latency_seconds=None, infrastructure_error=True,
                                          explicit_rejection=False, error='Skipped after infrastructure failure: ' + unavailable[method])
                        elif method == 'ibm':
                            result = call_ibm(item, model, endpoint, timeout, GLOSSARY, key,
                                              ibm_python, ibm_max_tokens)
                        else:
                            result = call_backend(method, item, model, endpoint, timeout, 'dfa')
                        if result['infrastructure_error']:
                            unavailable[method] = result['error']
                        assessment = score(item, result['formula'], result['explicit_rejection'], result['infrastructure_error'])
                        row = dict(**result)
                        row.update(method=method, implementation_id=actual_ids[method], item_id=item['id'],
                                   trial=trial, category=item['category'], raw_input=item['instruction'],
                                   reference_ltl=item['reference_ltl'], paraphrase_group=item['paraphrase_group'],
                                   assessment=assessment, requested_model=model, endpoint=endpoint)
                        row = redact(row, key)
                        rows.append(row)
                        stream.write(json.dumps(row) + '\n')
                        stream.flush()
                        print(f"{method} trial={trial} {item['id']}: {assessment['score']}", flush=True)
    finally:
        if old_key is None:
            os.environ.pop('LTL_API_KEY', None)
        else:
            os.environ['LTL_API_KEY'] = old_key
        if rows:
            summaries(output, rows, methods)
    return output
