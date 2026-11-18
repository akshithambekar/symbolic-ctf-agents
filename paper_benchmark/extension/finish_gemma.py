"""Recover missing Gemma outcomes until both models have 1,080 measurements."""
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
from resume_matrix import completed
from extension.free_models import GEMMA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    roots = [args.previous.resolve()]
    state = dict(models=GEMMA, target=2160, status='running', roots=[str(r) for r in roots],
                 note='Four-model final comparison: Ultra, Super VL, and these two Gemma models')
    while True:
        known = completed(roots)
        count = sum(k[0] in GEMMA for k in known)
        state.update(measured=count, heartbeat=datetime.now(timezone.utc).isoformat())
        if count == 2160:
            state['status'] = 'complete'
            save(output / 'status.json', state)
            return
        target = ROOT / 'runs' / f'gr{int(time.time()):x}'
        command = [sys.executable, '-u', str(ROOT / 'resume_matrix.py'),
                   '--output', str(target), '--env-file', str(args.env_file.resolve()),
                   '--models', *GEMMA]
        for root in roots:
            command += ['--previous', str(root)]
        state.update(status='recovering', active_output=str(target))
        save(output / 'status.json', state)
        with (output / f'{target.name}.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            save(output / 'status.json', state)
            while child.poll() is None:
                if (target / 'dataset.json').exists() and target not in roots:
                    roots.append(target)
                    state['roots'] = [str(r) for r in roots]
                state.update(measured=sum(k[0] in GEMMA for k in completed(roots)),
                             heartbeat=datetime.now(timezone.utc).isoformat())
                save(output / 'status.json', state)
                time.sleep(30)
        state['last_exit_code'] = child.returncode
        if (target / 'dataset.json').exists() and target not in roots:
            roots.append(target)
            state['roots'] = [str(r) for r in roots]
        if sum(k[0] in GEMMA for k in completed(roots)) == 2160:
            continue
        state.update(status='waiting_before_recovery', retry_delay_seconds=1800)
        save(output / 'status.json', state)
        time.sleep(1800)


if __name__ == '__main__':
    main()
