"""One bounded availability probe per advertised free alternative; not a benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run import use_frozen_source
use_frozen_source()
from maritime_nsrl.provider_config import load_env, api_key
from extension.gold_benchmark import check

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--env-file',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    load_env(args.env_file)
    key=api_key('https://openrouter.ai/api/v1')
    if not key:raise ValueError('No API key configured')
    response=requests.get('https://openrouter.ai/api/v1/models',timeout=30)
    response.raise_for_status();catalog=response.json()
    (args.output/'catalog.json').write_text(json.dumps(catalog,indent=2),encoding='utf-8')
    excluded={'google/gemma-4-26b-a4b-it:free','google/gemma-4-31b-it:free',
              'nvidia/nemotron-3.5-super-vl:free','nvidia/nemotron-3.5-lightning:free'}
    models=[m['id'] for m in catalog['data'] if m['id'].endswith(':free') and
        m['id'] not in excluded and m['id']!='openrouter/free' and
        'text' in m.get('architecture',{}).get('output_modalities',[]) and
        all(float(m['pricing'][k])==0 for k in ['prompt','completion'])]
    prompt='Translate this rule to LTL. Allowed proposition: in_bounds = inside the arena. Return only JSON {"ltl":"..."}. Rule: Always stay inside the arena.'
    def probe(model):
        started=time.perf_counter()
        payload=dict(model=model,temperature=0,max_tokens=1024,
                     messages=[dict(role='user',content=prompt)])
        row=dict(model=model,request=payload,started=datetime.now(timezone.utc).isoformat())
        try:
            response=requests.post('https://openrouter.ai/api/v1/chat/completions',
                headers={'Authorization':'Bearer '+key},json=payload,timeout=90,allow_redirects=False)
            row.update(http_status=response.status_code,response=response.text)
            response.raise_for_status();body=response.json()
            choice=body['choices'][0]
            raw=choice['message'].get('content') or ''
            row.update(raw_output=raw,finish_reason=choice.get('finish_reason'),usage=body.get('usage'),
                       status='response_received' if raw else 'empty_output')
            try:
                cleaned=raw.strip()
                if cleaned.startswith('```') and cleaned.endswith('```'):cleaned='\n'.join(cleaned.splitlines()[1:-1])
                formula=json.loads(cleaned).get('ltl','')
                row['formula']=formula
                # [] is the standard alternative spelling of globally.
                row['assessment']=check(formula.replace('[]','G'),'G(in_bounds)')
                row['expected_simple_formula']=row['assessment']['outcome']=='equivalent'
            except (ValueError,AttributeError):row['expected_simple_formula']=False
        except Exception as exc:row.update(status='request_failed',error=type(exc).__name__+': '+str(exc))
        row['latency_seconds']=time.perf_counter()-started
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(probe,model) for model in models]
        for future in as_completed(futures):
            row=future.result()
            with (args.output/'raw.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(row)+'\n')
            print(json.dumps({k:row.get(k) for k in ['model','status','http_status','expected_simple_formula','latency_seconds']}),flush=True)
    print('Availability probes only; not semantic benchmark accuracy.',flush=True)

if __name__=='__main__':main()
