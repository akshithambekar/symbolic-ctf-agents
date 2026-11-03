"""Run a fresh, restartable-by-condition exploratory four-model matrix."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
MODELS = [
    'nvidia/nemotron-3-ultra-550b-a55b:free',
    'nvidia/nemotron-3.5-super-vl:free',
    'nvidia/nemotron-3.5-lightning:free',
    'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
]


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = ROOT / 'study/candidate_v1/dataset.json'
    data = json.loads(dataset.read_text(encoding='utf-8'))
    assert len(data['items']) == 120
    frozen = output / 'dataset.json'
    frozen.write_bytes(dataset.read_bytes())
    state = dict(kind='exploratory', models=MODELS, methods=['current', 'ltlcodegen', 'ibm'],
                 items=120, trials=3, scheduled_outputs=4320, historical_rows_pooled=False,
                 reference_review_complete=False, started=datetime.now(timezone.utc).isoformat(),
                 source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 dataset_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                 scoring='Unconstrained Boolean LTLf; proposed grounding invariant not applied',
                 ibm_max_tokens=4096, conditions=[])
    save(output / 'status.json', state)
    for index, model in enumerate(MODELS):
        for trial in range(-1, 3):
            smoke = trial == -1
            target = output / f'model-{index}-' / ('smoke' if smoke else f'trial-{trial}')
            command = [sys.executable, str(ROOT / 'run.py'), 'compare', '--dataset', str(frozen),
                       '--model', model, '--trials', '1', '--timeout', '120',
                       '--ibm-max-tokens', '4096', '--env-file', str(args.env_file.resolve()),
                       '--output', str(target)]
            if smoke:
                command += ['--case-id', data['items'][0]['id']]
            condition = dict(model=model, trial=trial, smoke=smoke, output=str(target), status='running')
            state['conditions'].append(condition)
            save(output / 'status.json', state)
            target.parent.mkdir(exist_ok=True)
            with (target.parent / (target.name + '.log')).open('x', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
            raw = target / 'raw.jsonl'
            rows = [json.loads(line) for line in raw.read_text(encoding='utf-8').splitlines()] if raw.exists() else []
            failures = sum(bool(row['infrastructure_error']) for row in rows)
            condition.update(status='complete' if result.returncode == 0 and not failures else 'blocked',
                             exit_code=result.returncode, outputs=len(rows), infrastructure_failures=failures)
            save(output / 'status.json', state)
            if condition['status'] == 'blocked':
                break  # Preserve evidence; do not spend more calls after an availability/rate-limit failure.
    state['finished'] = datetime.now(timezone.utc).isoformat()
    state['status'] = 'complete' if all(c['status'] == 'complete' for c in state['conditions']) else 'incomplete'
    save(output / 'status.json', state)


if __name__ == '__main__':
    main()
