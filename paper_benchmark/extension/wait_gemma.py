"""Retry zero-measurement Gemma capacity checks without replacing benchmark data."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_matrix import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='waiting_for_free_capacity', rounds=[])
    for index in range(24):
        target = ROOT / 'runs' / f'gm{int(time.time()):x}'
        state.update(status='checking_free_capacity', active_output=str(target))
        save(output / 'status.json', state)
        with (output / f'round-{index}.log').open('x', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-u', str(ROOT / 'extension/gemma_study.py'),
                '--env-file', str(args.env_file.resolve()), '--output', str(target)],
                cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
        status_path = target / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        state['rounds'].append(dict(output=str(target), exit_code=result.returncode,
                                    status=status.get('status', 'local_error')))
        # Once measured rows exist, retain that run for selective recovery rather
        # than repeating the whole experiment and selecting favorable results.
        if any(c.get('measured', 0) for c in status.get('conditions', [])):
            state.update(status=status.get('status', 'incomplete'),
                         measured_output=str(target), note='Preserve this measured run; no full rerun')
            save(output / 'status.json', state)
            return
        state.update(status='waiting_for_free_capacity', retry_delay_seconds=1800,
                     heartbeat=datetime.now(timezone.utc).isoformat())
        save(output / 'status.json', state)
        time.sleep(1800)
    state.update(status='capacity_unavailable', note='24 half-hour capacity checks exhausted')
    save(output / 'status.json', state)


if __name__ == '__main__':
    main()
