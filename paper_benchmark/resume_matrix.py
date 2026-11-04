"""Continue missing matrix outcomes into new files; never replace a measured outcome."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from model_matrix import MODELS, ROOT, save


def daily_quota_reset(path):
    """Return provider reset time only for an explicit daily-quota response."""
    if not path.exists():
        return None
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for attempt in row.get('attempts', []):
            try:
                error = json.loads(attempt.get('response') or '{}').get('error', {})
                metadata = error.get('metadata', {})
                if metadata.get('limit_source') == 'openrouter_free_tier_daily':
                    reset = metadata.get('headers', {}).get('X-RateLimit-Reset')
                    return float(reset) / 1000 if reset else time.time() + 3600
            except (ValueError, TypeError, AttributeError):
                continue
    return None


def completed(roots):
    outcomes = {}
    for root in roots:
        for path in sorted(root.rglob('raw.jsonl')):
            if path.parent.name == 'smoke':
                continue
            metadata = path.parent / 'condition.json'
            if metadata.exists():
                trial = json.loads(metadata.read_text())['matrix_trial']
            elif path.parent.name.startswith('trial-'):
                trial = int(path.parent.name.split('-')[1])
            else:
                continue
            for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A partial final write is not a measured outcome.
                key = (row['requested_model'], trial, row['method'], row['item_id'])
                if not row['infrastructure_error'] and key not in outcomes:
                    outcomes[key] = dict(source=str(path), line=number, row=row)
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, action='append', required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    previous = [p.resolve() for p in args.previous]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = previous[0] / 'dataset.json'
    frozen = output / 'dataset.json'
    frozen.write_bytes(dataset.read_bytes())
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    for root in previous:
        if hashlib.sha256((root / 'dataset.json').read_bytes()).hexdigest() != digest:
            raise ValueError('Cannot combine different dataset snapshots')
    data = json.loads(frozen.read_text(encoding='utf-8'))
    roots = previous + [output]
    state = dict(kind='exploratory-continuation', previous=[str(p) for p in previous],
                 started=datetime.now(timezone.utc).isoformat(), dataset_sha256=digest,
                 source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 scheduled=4320, retained=len(completed(previous)), conditions=[], status='running',
                 policy='Keep first non-infrastructure outcome, including incorrect/unusable outputs; retry only infrastructure/missing outcomes')
    for index, model in enumerate(MODELS):
        for trial in range(3):
            for method in ('current', 'ltlcodegen', 'ibm'):
                for retry in range(3):
                    known = completed(roots)
                    missing = [i['id'] for i in data['items'] if (model, trial, method, i['id']) not in known]
                    if not missing:
                        break
                    # Keep snapshot paths below Windows' legacy path-length limit.
                    target = output / f'm{index}t{trial}{method[:1]}a{retry}'
                    condition = dict(model=model, matrix_trial=trial, method=method,
                                     missing=len(missing), output=str(target), status='running')
                    state['conditions'].append(condition)
                    state['measured_outputs'] = len(known)
                    save(output / 'status.json', state)
                    command = [sys.executable, '-u', str(ROOT / 'run.py'), 'compare',
                               '--dataset', str(frozen), '--model', model, '--trials', '1',
                               '--methods', method, '--timeout', '120', '--ibm-max-tokens', '4096',
                               '--env-file', str(args.env_file.resolve()), '--output', str(target)]
                    for item_id in missing:
                        command += ['--case-id', item_id]
                    with (output / (target.name + '.log')).open('x', encoding='utf-8') as log:
                        process = subprocess.Popen(command, cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
                        condition['pid'] = process.pid
                        save(output / 'status.json', state)
                        # Persist trial identity as soon as the child creates its directory.
                        while process.poll() is None:
                            if target.exists() and not (target / 'condition.json').exists():
                                save(target / 'condition.json', condition)
                            time.sleep(1)
                    if target.exists():
                        save(target / 'condition.json', condition)
                    known = completed(roots)
                    remaining = sum((model, trial, method, i) not in known for i in missing)
                    condition.update(status='complete' if not remaining else 'incomplete',
                                     remaining=remaining, exit_code=process.returncode)
                    state['measured_outputs'] = len(known)
                    save(output / 'status.json', state)
                    reset = daily_quota_reset(target / 'raw.jsonl')
                    if reset is not None:
                        # Keep the background launcher alive without issuing more
                        # API requests until the provider's daily reset.
                        wake = max(reset + 60, time.time() + 60)
                        state.update(status='waiting_for_daily_quota',
                                     resume_at=datetime.fromtimestamp(wake, timezone.utc).isoformat())
                        save(output / 'status.json', state)
                        while time.time() < wake:
                            time.sleep(min(30, max(0.1, wake - time.time())))
                        state.update(status='running')
                        state.pop('resume_at', None)
                        save(output / 'status.json', state)
                    if remaining:
                        time.sleep(60)
    outcomes = completed(roots)
    with (output / 'combined_measured.jsonl').open('x', encoding='utf-8') as stream:
        for (model, trial, method, item_id), entry in sorted(outcomes.items()):
            stream.write(json.dumps(dict(**entry, matrix_trial=trial)) + '\n')
    state.update(status='complete' if len(outcomes) == 4320 else 'incomplete',
                 measured_outputs=len(outcomes), finished=datetime.now(timezone.utc).isoformat())
    save(output / 'status.json', state)


if __name__ == '__main__':
    main()
