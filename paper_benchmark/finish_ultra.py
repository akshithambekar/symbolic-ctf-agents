"""Supervise Ultra recovery until every scheduled outcome is measured."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from model_matrix import ROOT, MODELS, save
from resume_matrix import completed

MODEL = MODELS[0]


def ultra_count(roots):
    return sum(key[0] == MODEL for key in completed(roots))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, action='append', required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--existing-pid', type=int)
    args = parser.parse_args()
    # User requested continuous completion; prevent idle sleep while supervising.
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    roots = [p.resolve() for p in args.previous]
    state = dict(status='supervising', model=MODEL, target=1080,
                 roots=[str(p) for p in roots], source_commit=subprocess.check_output(
                     ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    # Existing recovery remains responsible for its in-flight request.
    existing = args.existing_pid
    round_number = 0
    child = None
    while True:
        try:
            count = ultra_count(roots)
            state.update(measured=count, heartbeat=datetime.now(timezone.utc).isoformat())
            save(output / 'status.json', state)
            if count == 1080:
                break
            if existing:
                import ctypes
                kernel = ctypes.windll.kernel32
                kernel.OpenProcess.restype = ctypes.c_void_p
                kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
                kernel.CloseHandle.argtypes = [ctypes.c_void_p]
                handle = kernel.OpenProcess(0x1000, False, existing)
                if handle:
                    exit_code = ctypes.c_ulong()
                    kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                    kernel.CloseHandle(handle)
                    if exit_code.value == 259:
                        state.update(status='waiting_for_existing_recovery', child_pid=existing)
                        time.sleep(15)
                        continue
                existing = None
            if child is not None and child.poll() is None:
                state.update(status='recovering', child_pid=child.pid)
                time.sleep(15)
                continue
            if child is not None:
                state['last_exit_code'] = child.returncode
                child = None
                time.sleep(60)
            target = ROOT / 'runs' / f'z{int(time.time()):x}-{round_number}'
            command = [sys.executable, '-u', str(ROOT / 'resume_matrix.py'),
                       '--env-file', str(args.env_file.resolve()), '--output', str(target),
                       '--models', MODEL]
            for root in roots:
                command += ['--previous', str(root)]
            log_path = output / f'round-{round_number}.log'
            with log_path.open('x', encoding='utf-8') as log:
                child = subprocess.Popen(command, cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
            # Include the new root as soon as its immutable dataset is written.
            for _ in range(60):
                if (target / 'dataset.json').exists() or child.poll() is not None:
                    break
                time.sleep(1)
            if (target / 'dataset.json').exists():
                roots.append(target)
            state.update(status='recovering', child_pid=child.pid, roots=[str(p) for p in roots])
            round_number += 1
        except Exception as exc:
            state.update(status='retrying_supervisor_error', error=type(exc).__name__ + ': ' + str(exc))
            save(output / 'status.json', state)
            time.sleep(60)
    # Build a consolidated, provenance-linked report with the original scorer.
    from run import use_frozen_source
    use_frozen_source()
    from comparison import summaries
    entries = {key: value for key, value in completed(roots).items() if key[0] == MODEL}
    rows = []
    for key, entry in sorted(entries.items()):
        row = dict(entry['row'], trial=key[1], source_file=entry['source'], source_line=entry['line'])
        rows.append(row)
    with (output / 'raw.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    summaries(output, rows, ['current', 'ltlcodegen', 'ibm'])
    state.update(status='complete', measured=len(rows), finished=datetime.now(timezone.utc).isoformat())
    save(output / 'status.json', state)


if __name__ == '__main__':
    main()
