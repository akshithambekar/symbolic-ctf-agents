"""Supervise the 270-outcome gold-domain pilot, retaining measured failures."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_matrix import save

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--env-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    roots=[];round_number=0
    state=dict(status='running',target=270,semantics='infinite-trace LTL',roots=[])
    while True:
        target=ROOT/'runs'/f'gd{int(time.time()):x}-{round_number}'
        command=[sys.executable,'-u',str(ROOT/'extension/gold_benchmark.py'),
                 '--source',str(args.source.resolve()),'--env-file',str(args.env_file.resolve()),
                 '--output',str(target)]
        for root in roots:
            command+=['--previous',str(root)]
        state.update(status='running',active_output=str(target),heartbeat=datetime.now(timezone.utc).isoformat())
        with (output/f'round-{round_number}.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save(output/'status.json',state)
            child.wait()
        if (target/'dataset.json').exists() and (target/'raw.jsonl').exists():
            roots.append(target);state['roots']=[str(p) for p in roots]
        current=json.loads((target/'status.json').read_text()) if (target/'status.json').exists() else {}
        state.update(last_exit_code=child.returncode,last_run_status=current.get('status'),
                     recorded=current.get('recorded',0))
        if current.get('status')=='complete':
            state.update(status='complete',finished=datetime.now(timezone.utc).isoformat())
            save(output/'status.json',state);return
        state.update(status='waiting_before_recovery',retry_delay_seconds=1800)
        save(output/'status.json',state)
        time.sleep(1800)
        round_number+=1

if __name__=='__main__':main()
