"""Recover missing Cohere/Dots cells; preserve first measured outcomes and audit calls."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_matrix import save
from resume_matrix import completed
from extension.alternative_study import MODELS
from extension.audit import audit

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--env-file',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    roots=list(dict.fromkeys(path.resolve() for path in args.previous))
    state=dict(models=MODELS,target=2160,status='starting',roots=[str(p) for p in roots],
        policy='Preserve first measured output, including failures; retry missing/infrastructure only.',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip())
    count=lambda:sum(k[0] in MODELS for k in completed(roots))
    no_progress=0
    for round_index in range(12):
        before=count()
        if before==2160:break
        target=ROOT/'runs'/f'cdrec-{int(time.time()):x}'
        command=[sys.executable,'-u',str(ROOT/'resume_matrix.py'),'--output',str(target),
            '--env-file',str(args.env_file.resolve()),'--models',*MODELS]
        for root in roots:command+=['--previous',str(root)]
        state.update(status='running',active_output=str(target),measured=before,round=round_index)
        save(output/'status.json',state)
        with (output/f'round-{round_index}.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid
            while child.poll() is None:
                if (target/'dataset.json').exists() and target not in roots:roots.append(target)
                state.update(roots=[str(p) for p in roots],measured=count(),
                    heartbeat=datetime.now(timezone.utc).isoformat())
                save(output/'status.json',state)
                time.sleep(15)
        if (target/'dataset.json').exists() and target not in roots:roots.append(target)
        after=count()
        state.update(measured=after,roots=[str(p) for p in roots],last_exit_code=child.returncode)
        audit(roots,output/f'audit-{round_index:02d}')
        no_progress=no_progress+1 if after==before else 0
        if after==2160:break
        if no_progress>=3:
            state.update(status='stopped_after_three_no_progress_passes')
            save(output/'status.json',state)
            return
        state.update(status='waiting_before_retry',retry_delay_seconds=1800)
        save(output/'status.json',state)
        for _ in range(120):time.sleep(15)
    state.update(status='complete' if count()==2160 else 'incomplete',measured=count())
    save(output/'status.json',state)

if __name__=='__main__':main()
