"""Fresh Cohere/Dots CTF extension with attempt-level usage and failure audits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_matrix import save
from resume_matrix import completed
from extension.audit import audit

MODELS=['cohere/north-mini-code:free','dots-studio/dots-3-note-preview:free']
METHODS=['current','ltlcodegen','ibm']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--env-file',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    dataset=ROOT/'study/candidate_v1/dataset.json'
    shutil.copyfile(dataset,output/'dataset.json')
    catalog=requests.get('https://openrouter.ai/api/v1/models',timeout=30)
    catalog.raise_for_status();catalog=catalog.json()
    for model in MODELS:
        entry=next(m for m in catalog['data'] if m['id']==model)
        if any(float(entry['pricing'][k])!=0 for k in ['prompt','completion']):
            raise ValueError('Free route unavailable: '+model)
    save(output/'catalog.json',catalog)
    save(output/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),ROOT/'comparison.py',ROOT/'extension/audit.py',ROOT/'run.py']})
    shutil.copyfile(__file__,output/'alternative_study.py')
    state=dict(status='starting',kind='exploratory-model-extension',models=MODELS,methods=METHODS,
        items=120,trials=3,target=2160,measured=0,roots=[str(output)],conditions=[],
        started=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        independent_gold_review_complete=False,
        protocol='Existing frozen CTF prompts, native per-workflow settings, Boolean LTLf scorer. No prompt tuning.',
        accounting='All actual calls including repairs/failures. Missing usage/cost unknown. Reasoning tokens are a subset.',
        study_scope='CTF three-translator extension, separate from the two-arm GPT-4o Pan pilot.')
    save(output/'status.json',state)
    blocked=set()
    def condition(model,index,trial):
        smoke=trial==-1
        target=output/f'm{index}'/('smoke' if smoke else f'trial-{trial}')
        target.parent.mkdir(exist_ok=True)
        cmd=[sys.executable,'-u',str(ROOT/'run.py'),'compare','--dataset',str(output/'dataset.json'),
            '--model',model,'--methods',*METHODS,'--trials','1','--timeout','120',
            '--ibm-max-tokens','4096','--env-file',str(args.env_file.resolve()),'--output',str(target)]
        if smoke:cmd+=['--case-id','dev-bounds-0','--case-id','dev-capture-0']
        entry=dict(model=model,matrix_trial=trial,smoke=smoke,output=str(target),status='running')
        state['conditions'].append(entry)
        with (target.parent/(target.name+'.log')).open('x',encoding='utf-8') as log:
            process=subprocess.Popen(cmd,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
            entry['pid']=process.pid
            while process.poll() is None:
                state.update(status='running',heartbeat=datetime.now(timezone.utc).isoformat(),
                    measured=sum(k[0] in MODELS for k in completed([output])))
                save(output/'status.json',state)
                time.sleep(10)
        raw=target/'raw.jsonl'
        rows=[json.loads(x) for x in raw.read_text(encoding='utf-8').splitlines()] if raw.exists() else []
        failed=process.returncode!=0 or not rows or any(r['infrastructure_error'] for r in rows)
        entry.update(status='infrastructure_blocked' if failed else 'complete',exit_code=process.returncode,
            measured=sum(not r['infrastructure_error'] for r in rows))
        if failed:blocked.add(model)
        # Audit actual paid/free usage, retries, truncation, refusals and semantic
        # mismatches separately. Smoke is excluded from main quality/usage totals.
        audit([output],output/f'audit-{len(state["conditions"]):02d}')
        save(output/'status.json',state)
    for index,model in enumerate(MODELS):condition(model,index,-1)
    for trial in range(3):
        for index,model in enumerate(MODELS):
            if model not in blocked:condition(model,index,trial)
    measured=sum(k[0] in MODELS for k in completed([output]))
    state.update(status='complete' if measured==2160 else 'incomplete',measured=measured,
        blocked_models=sorted(blocked),finished=datetime.now(timezone.utc).isoformat())
    save(output/'status.json',state)
    audit([output],output/'final-audit')

if __name__=='__main__':main()
