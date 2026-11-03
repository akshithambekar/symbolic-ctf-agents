"""Verify, re-score, or repeat the original paper benchmark without replacing it."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'reference_run'
SOURCE = REFERENCE / 'source'
MODEL = 'nvidia/nemotron-3-ultra-550b-a55b:free'
ENDPOINT = 'https://openrouter.ai/api/v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify():
    manifests = [('original_manifest', REFERENCE, read(REFERENCE / 'manifest.json')),
                 ('preserved_bundle', REFERENCE, read(ROOT / 'reference_files.sha256.json'))]
    counts = {}
    for label, base, hashes in manifests:
        for name, expected in hashes.items():
            path = base / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Historical artifact changed: {name}')
        counts[label] = len(hashes)
    rows = [json.loads(line) for line in (REFERENCE / 'raw.jsonl').read_text().splitlines()]
    methods = {}
    for summary in read(REFERENCE / 'summary.json'):
        method = summary['method']
        measured = [r for r in rows if r['method'] == method and not r['infrastructure_error']]
        formula_rows = [r for r in measured if r['reference_ltl'] is not None]
        recomputed = {
            'measured': len(measured),
            'syntax_valid_rate': statistics.mean(float(r['assessment']['syntax_valid']) for r in formula_rows),
            'compilable_rate': statistics.mean(float(r['assessment']['compilable']) for r in formula_rows),
            'semantic_correct_rate': statistics.mean(float(r['assessment']['score'] == 'Correct') for r in formula_rows),
            'repair_rate': statistics.mean(float(r['repair_attempts'] > 0) for r in measured),
            'mean_latency_seconds': statistics.mean(r['latency_seconds'] for r in measured),
        }
        for key, value in recomputed.items():
            if value != summary[key]:
                raise ValueError(f'Raw results disagree with summary: {method}/{key}')
        methods[method] = {'formula_reference_outputs': len(formula_rows), **recomputed}
    return {'verified': True, 'hash_counts': counts,
            'dataset_items': len(read(REFERENCE / 'dataset.json')['items']),
            'configuration': read(REFERENCE / 'config.json'), 'methods': methods}


def use_frozen_source():
    # Avoid modifying the archived source, including by generating bytecode files.
    sys.dont_write_bytecode = True
    existing = sys.modules.get('maritime_nsrl')
    if existing and not Path(existing.__file__).resolve().is_relative_to(SOURCE.resolve()):
        raise RuntimeError('Run in a fresh Python process; another translator package is loaded.')
    sys.path.insert(0, str(SOURCE))


def output_directory(value, kind):
    if value:
        target = Path(value).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        target = ROOT / 'runs' / f'{kind}-{stamp}-{uuid4().hex[:6]}'
    if target == REFERENCE or REFERENCE in target.parents:
        raise ValueError('Cannot write new results inside the historical reference run.')
    if target.exists():
        raise FileExistsError(f'Output already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('verify', help='Check saved hashes and summaries; no API calls')
    rescore = commands.add_parser('rescore', help='Re-score the saved model outputs; no API calls')
    rescore.add_argument('--output', type=Path)
    for name in ('run', 'compare'):
        live = commands.add_parser(name, help='New provider calls: frozen pair or comparison including IBM')
        live.add_argument('--model', default=MODEL)
        live.add_argument('--endpoint', default=ENDPOINT)
        live.add_argument('--trials', type=int, default=3)
        live.add_argument('--timeout', type=float, default=120)
        live.add_argument('--env-file', type=Path, default=ROOT.parent / '.env')
        live.add_argument('--output', type=Path)
        if name == 'compare':
            live.add_argument('--methods', nargs='+', choices=('current', 'ltlcodegen', 'ibm'),
                              default=['current', 'ltlcodegen', 'ibm'])
            live.add_argument('--case-id', action='append', dest='case_ids')
            live.add_argument('--ibm-python', type=Path)
            live.add_argument('--ibm-max-tokens', type=int, default=200,
                              help='IBM native cap is 200; other caps get a separate variant ID')
    args = parser.parse_args(argv)
    evidence = verify()
    if args.command == 'verify':
        print(json.dumps(evidence, indent=2))
        return
    use_frozen_source()
    from maritime_nsrl.translator_benchmark import runner
    output = output_directory(args.output, args.command)
    if args.command == 'rescore':
        from maritime_nsrl.translator_benchmark.scoring import score
        items = {item['id']: item for item in read(REFERENCE / 'dataset.json')['items']}
        rows = [json.loads(line) for line in (REFERENCE / 'raw.jsonl').read_text().splitlines()]
        changed = []
        for row in rows:
            assessment = score(items[row['item_id']], row['formula'],
                               row['explicit_rejection'], row['infrastructure_error'])
            if assessment != row['assessment']:
                changed.append({'method': row['method'], 'item_id': row['item_id'], 'trial': row['trial']})
            row['assessment'] = assessment
        output.mkdir()
        runner.save(output / 'provenance.json', {
            'kind': 'offline_rescore', 'provider_requests': 0,
            'reference_run': REFERENCE.name, 'assessment_differences': changed})
        with (output / 'raw.jsonl').open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
        runner.summarize(output, rows)
        print(json.dumps({'output': str(output), 'provider_requests': 0,
                          'assessment_differences': len(changed)}))
    else:
        from maritime_nsrl.provider_config import load_env
        load_env(args.env_file)
        if args.command == 'compare':
            from comparison import run as compare
            compare(output, methods=args.methods, model=args.model, endpoint=args.endpoint,
                    trials=args.trials, timeout=args.timeout, case_ids=args.case_ids,
                    ibm_python=args.ibm_python, ibm_max_tokens=args.ibm_max_tokens)
        else:
            runner.run(output, model=args.model, endpoint=args.endpoint, trials=args.trials,
                       timeout=args.timeout, logic_backend='dfa')
        runner.save(output / 'launch_provenance.json', {
            'kind': 'new_comparison' if args.command == 'compare' else 'new_live_run', 'reference_model': MODEL,
            'same_model_as_paper': args.model == MODEL,
            'source_mode': 'frozen pair plus separately labeled IBM adapter' if args.command == 'compare'
                           else 'unaltered reference_run/source',
            'original_artifacts_modified': False})
        print(f'New results saved to {output}')


if __name__ == '__main__':
    main()
