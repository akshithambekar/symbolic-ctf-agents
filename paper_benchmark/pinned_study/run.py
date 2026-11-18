"""Two-arm paid pilot. Never executes model-generated code."""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extension.gold_benchmark import check
from benchmarks.formula_notation import to_spot
from maritime_nsrl.provider_config import load_env, api_key
import requests

MODEL = 'openai/gpt-4o-2024-11-20'
INPUT_PRICE, OUTPUT_PRICE = .0000025, .00001
MAX_TOKENS = 4000

def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)

def reserve_cost(messages):
    # UTF-8 bytes upper-bound text BPE tokens, plus generous framing allowance.
    tokens = sum(len(m['content'].encode('utf-8')) for m in messages) + 32*len(messages) + 1024
    return tokens * INPUT_PRICE + MAX_TOKENS * OUTPUT_PRICE

class Budget:
    def __init__(self, path, limit):
        if not math.isfinite(limit) or not 0 < limit <= 5:
            raise ValueError('This pilot permits a positive ceiling of at most $5')
        self.path, self.limit, self.accounted = path, limit, 0.
        self.pending = None
        self.save()

    def save(self):
        write(self.path, dict(ceiling_usd=self.limit, accounted_usd=self.accounted,
                             pending_reservation_usd=self.pending))

    def reserve(self, messages):
        amount = reserve_cost(messages)
        if self.accounted + amount > self.limit:
            raise RuntimeError('budget_exhausted')
        self.accounted += amount
        self.pending = amount
        self.save()  # Charge reservation before network call; crashes cannot erase it.
        return amount

    def settle(self, usage):
        amount = self.pending
        if isinstance(usage, dict):
            cost = usage.get('cost')
            if cost is None and all(isinstance(usage.get(k), int) for k in ('prompt_tokens','completion_tokens')):
                cost = usage['prompt_tokens']*INPUT_PRICE + usage['completion_tokens']*OUTPUT_PRICE
            if isinstance(cost, (float,int)) and math.isfinite(cost) and 0 <= cost <= amount:
                self.accounted += cost - amount
        self.pending = None
        self.save()

def normalize(raw, method, objects):
    if len(raw) > 64000:
        raise ValueError('Output too long')
    raw = raw.strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = '\n'.join(raw.splitlines()[1:-1]).strip()
    if method == 'custom':
        value = json.loads(raw)
        if not isinstance(value,dict) or not isinstance(value.get('ltl'),str):
            raise ValueError('Expected JSON object with string ltl')
        return value['ltl']
    # literal_eval reads a flat list of strings; it never executes expressions.
    value = ast.literal_eval(raw)
    if not isinstance(value,list) or len(value)>1000 or not all(isinstance(t,str) for t in value):
        raise ValueError('Expected flat prefix string list')
    tokens = iter(value)
    unary = {'NEGATION':'!', 'ALWAYS':'G', 'EVENTUALLY':'F', 'NEXT':'X'}
    binary = {'AND':'&', 'OR':'|', 'IMPLY':'->', 'UNTIL':'U', 'EQUAL':'<->'}
    def parse(depth=0):
        if depth>64: raise ValueError('Expression too deep')
        token = next(tokens)
        if token in unary: return f'{unary[token]}({parse(depth+1)})'
        if token in binary:
            a,b = parse(depth+1),parse(depth+1)
            if token=='EQUAL': return f'(({a}) -> ({b})) & (({b}) -> ({a}))'
            return f'({a}) {binary[token]} ({b})'
        if token.startswith('reach(') and token.endswith(')'):
            return objects[token[6:-1]]
        raise ValueError('Unknown prefix token: '+token)
    try: result = parse()
    except (StopIteration,KeyError) as exc: raise ValueError('Incomplete expression or unknown object') from exc
    if next(tokens,None) is not None: raise ValueError('Unconsumed tokens')
    return result

def summarize(rows):
    groups=defaultdict(list)
    for row in rows: groups[(row['method'],row['domain'],row['split'])].append(row)
    return [dict(method=k[0],domain=k[1],split=k[2],n=len(v),
        valid=sum(bool(r.get('formula')) for r in v),
        correct=sum(r.get('assessment',{}).get('outcome')=='equivalent' for r in v),
        repairs=sum(len(r['attempts'])>1 for r in v),
        latency_seconds=sum(r['latency_seconds'] for r in v)/len(v)) for k,v in sorted(groups.items())]

def validate_predicates(formula, allowed):
    _, atoms = to_spot(formula, 'infix')
    unknown = set(atoms) - set(allowed)
    if unknown:
        raise ValueError('Unsupported predicates: '+str(sorted(unknown)))

def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    frozen_path=args.frozen/'frozen.json'
    manifest=json.loads((args.frozen/'manifest.json').read_text())
    for name,digest in manifest.items():
        if hashlib.sha256((args.frozen/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen artifact hash mismatch: '+name)
    data=json.loads(frozen_path.read_text(encoding='utf-8'))
    # Validate every reference and checker availability before spending money.
    for item in data['items']:
        checked=check(item['reference'],item['reference'])
        if checked['outcome']!='equivalent': raise ValueError('Invalid reference: '+item['id'])
    load_env(args.env_file)
    key=api_key('https://openrouter.ai/api/v1')
    if not key: raise ValueError('No API key configured')
    catalog=requests.get('https://openrouter.ai/api/v1/models',timeout=30).json()
    model=next(m for m in catalog['data'] if m['id']==MODEL)
    if float(model['pricing']['prompt'])>INPUT_PRICE or float(model['pricing']['completion'])>OUTPUT_PRICE:
        raise ValueError('Model pricing exceeds budget assumptions')
    config=dict(model=MODEL,provider='OpenAI',temperature=0,max_tokens=MAX_TOKENS,
        max_repairs=3,methods=['custom','paper_direct'],budget_usd=args.budget_usd,
        design='exploratory pilot; domain-adapted custom and reconstructed paper direct baseline',
        exact_paper_replication=False,generated_python_execution=False,
        frozen_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),
        source_dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT.parent,text=True),
        pricing=model['pricing'])
    shutil.copytree(args.frozen,args.output/'frozen')
    shutil.copytree(Path(__file__).parent,args.output/'source',ignore=shutil.ignore_patterns('__pycache__'))
    write(args.output/'config.json',config)
    budget=Budget(args.output/'budget.json',args.budget_usd)
    rows=[]
    status='running'
    try:
        for item in data['items']:
            for method in config['methods']:
                definition=data['prompts'][item['domain']]
                messages=([dict(role='system',content=definition['custom_system']),dict(role='user',content=item['instruction'])]
                    if method=='custom' else [dict(role='user',content=definition['paper_direct_template'].format(instruction=item['instruction']))])
                row=dict(id=item['id'],method=method,domain=item['domain'],split=item['split'],
                    instruction=item['instruction'],reference=item['reference'],attempts=[],formula=None)
                started=time.perf_counter()
                for retry in range(4):
                    reservation=budget.reserve(messages)
                    payload=dict(model=MODEL,messages=messages.copy(),temperature=0,max_tokens=MAX_TOKENS,
                        provider=dict(order=['OpenAI'],allow_fallbacks=False))
                    attempt=dict(request=payload,reservation_usd=reservation)
                    row['attempts'].append(attempt)
                    tick=time.perf_counter()
                    try:
                        response=requests.post('https://openrouter.ai/api/v1/chat/completions',
                            headers={'Authorization':'Bearer '+key},json=payload,timeout=120,allow_redirects=False)
                        attempt.update(http_status=response.status_code,response=response.text)
                        body=response.json()
                        budget.settle(body.get('usage'))
                        response.raise_for_status()
                        raw=body['choices'][0]['message']['content'] or ''
                        attempt['raw_output']=raw
                    except Exception as exc:
                        if budget.pending is not None: budget.settle(None)
                        attempt['error']=type(exc).__name__+': '+str(exc)
                        row['infrastructure_error']=True
                        break
                    finally:
                        attempt['latency_seconds']=time.perf_counter()-tick
                        with (args.output/'attempts.jsonl').open('a',encoding='utf-8') as handle:
                            handle.write(json.dumps(dict(id=item['id'],method=method,attempt=attempt))+'\n')
                    try:
                        formula=normalize(raw,method,definition['object_mapping'])
                        validate_predicates(formula,definition['mapping'].values())
                        syntax=check(formula,formula)
                        if syntax['outcome']!='equivalent': raise ValueError(str(syntax))
                    except (ValueError,TypeError,SyntaxError) as exc:
                        attempt['validation_error']=str(exc)
                        messages += [dict(role='assistant',content=raw),dict(role='user',content=
                            'Validation failed: '+str(exc)+'. Correct the output without changing the instruction.')]
                        continue
                    row['formula']=formula
                    row['assessment']=check(formula,item['reference'])
                    break
                row['latency_seconds']=time.perf_counter()-started
                rows.append(row)
                with (args.output/'raw.jsonl').open('a',encoding='utf-8') as handle: handle.write(json.dumps(row)+'\n')
                write(args.output/'summary.json',summarize(rows))
                write(args.output/'status.json',dict(status='running',completed=len(rows),scheduled=2*len(data['items']),accounted_usd=budget.accounted))
                if row.get('infrastructure_error'):
                    raise RuntimeError('Provider error; stopped rather than repeatedly spending on failures')
        status='complete'
    except Exception as exc:
        status=type(exc).__name__+': '+str(exc)
    finally:
        write(args.output/'status.json',dict(status=status,completed=len(rows),scheduled=2*len(data['items']),accounted_usd=budget.accounted))
        print(json.dumps(dict(status=status,completed=len(rows),accounted_usd=budget.accounted)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--env-file',type=Path,required=True)
    parser.add_argument('--budget-usd',type=float,default=5.)
    run(parser.parse_args())
