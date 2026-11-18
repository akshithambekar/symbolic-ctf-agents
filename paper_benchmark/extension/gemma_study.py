"""Fresh two-Gemma exploratory comparison; original 120-item protocol unchanged."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_matrix import save
from extension.free_models import GEMMA


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog = requests.get('https://openrouter.ai/api/v1/models', timeout=30).json()
    for model in GEMMA:
        record = next((m for m in catalog['data'] if m['id'] == model), None)
        if record is None or any(float(record['pricing'][k]) != 0 for k in ('prompt', 'completion')):
            raise ValueError('Selected route is not currently advertised at zero price: ' + model)
    (output / 'catalog.json').write_text(json.dumps(catalog, indent=2), encoding='utf-8')
    dataset = ROOT / 'study/candidate_v1/dataset.json'
    (output / 'dataset.json').write_bytes(dataset.read_bytes())
    state = dict(kind='exploratory-model-extension', models=GEMMA, status='running',
                 items=120, trials=3, translators=3, scheduled=2160,
                 started=datetime.now(timezone.utc).isoformat(), conditions=[])
    for index, model in enumerate(GEMMA):
        # Same workflow settings as NVIDIA, with no new prompt tuning.
        for stage in ('smoke', 'full'):
            target = output / f'g{index}-{stage}'
            command = [sys.executable, '-u', str(ROOT / 'run.py'), 'compare', '--model', model,
                       '--dataset', str(output / 'dataset.json'), '--trials', '1' if stage == 'smoke' else '3',
                       '--ibm-max-tokens', '4096', '--timeout', '120',
                       '--env-file', str(args.env_file.resolve()), '--output', str(target)]
            if stage == 'smoke':
                command += ['--case-id', 'dev-bounds-0', '--case-id', 'dev-capture-0']
            entry = dict(model=model, stage=stage, output=str(target), status='running')
            state['conditions'].append(entry)
            save(output / 'status.json', state)
            with (output / f'g{index}-{stage}.log').open('x', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
            path = target / 'raw.jsonl'
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
            entry.update(exit_code=result.returncode, measured=sum(not r['infrastructure_error'] for r in rows),
                         status='complete' if result.returncode == 0 and rows and
                         not any(r['infrastructure_error'] for r in rows) else 'incomplete')
            save(output / 'status.json', state)
            if entry['status'] != 'complete':
                break
    state.update(status='complete' if len(state['conditions']) == 4 and
                 all(c['status'] == 'complete' for c in state['conditions']) else 'incomplete',
                 finished=datetime.now(timezone.utc).isoformat())
    save(output / 'status.json', state)


if __name__ == '__main__':
    main()
