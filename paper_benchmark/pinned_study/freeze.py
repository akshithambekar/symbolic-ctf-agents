"""Freeze shared Pan examples and disjoint human/synthetic pilot instructions."""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extension.gold_benchmark import native_tree,render,prepare,REVISION

COUNTS={'drone-planning':20,'clean-up':12,'pick-and-place':5}
TEST_COUNTS={'drone-planning':20,'clean-up':20,'pick-and-place':5}

def normalized_instruction(value):
    return ' '.join(re.findall(r'[a-z0-9]+',value.casefold()))

def extract_template(source,function):
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
    node=next(n for n in fn.body if isinstance(n,ast.Assign) and
              any(isinstance(t,ast.Name) and t.id=='template' for t in n.targets))
    if not isinstance(node.value,ast.Constant) or not isinstance(node.value.value,str):
        raise ValueError('Expected a static upstream prompt literal')
    return node.value.value

def prefix(tree,mapping):
    if tree[0]=='ap':return [mapping[tree[1]]]
    op={'!':'NEGATION','&':'AND','|':'OR','->':'IMPLY','U':'UNTIL',
        'G':'ALWAYS','F':'EVENTUALLY','X':'NEXT'}[tree[0]]
    return [op]+[v for child in tree[1:] for v in prefix(child,mapping)]

def freeze(source,golden,upstream,output):
    output.mkdir(parents=True,exist_ok=False)
    scratch=output/'domain-profile';scratch.mkdir()
    profile=prepare(golden,scratch)
    rng=random.Random(20261118)
    definitions={};examples={};items=[];source_hashes={}
    template_source=(upstream/'scenegraph_prompt.py').read_text(encoding='utf-8')
    original=extract_template(template_source,'get_translate_prompt')
    # Preserve the upstream task/operator/output contract. Replace the generic
    # robotics examples with the frozen Pan examples rather than adding both.
    header=original.split('Given below are examples')[0]
    for domain,count in COUNTS.items():
        canonical_path=source/(domain+'-canonical.json')
        canonical=json.loads(canonical_path.read_text(encoding='utf-8'))
        seed_path=source/(domain+'.jsonl')
        pool=[json.loads(x) for x in seed_path.read_text(encoding='utf-8').splitlines()]
        domain_profile=profile['domains'][domain]
        mapping=domain_profile['mapping']
        objects={native:f'reach(object_{i})' for i,native in enumerate(mapping)}
        alias_objects={f'object_{i}':alias for i,alias in enumerate(mapping.values())}
        candidates=defaultdict(list)
        for index,row in enumerate(pool):candidates[row['canonical']].append((index,row))
        keys=rng.sample(sorted(candidates),count)
        chosen=[]
        for key in keys:
            index,row=rng.choice(candidates[key]);raw=canonical[key]['raw']
            tree=native_tree(raw,domain)
            chosen.append(dict(source_row=index,instruction=row['natural'],canonical=key,
                original_reference=raw,ltl=render(tree,mapping),
                listed_prefix=prefix(tree,objects)))
        examples[domain]=chosen
        banned={normalized_instruction(e['instruction']) for e in chosen}
        object_text='\n'.join(f'object_{i} = {alias}: '+domain_profile['glossary'].splitlines()[i].split(': ',1)[1]
                              for i,alias in enumerate(mapping.values()))
        direct_examples='\n\n'.join('natural language: '+e['instruction']+'\nLTL: '+str(e['listed_prefix']) for e in chosen)
        custom_examples='\n\n'.join('Instruction: '+e['instruction']+'\nOutput: '+json.dumps({'ltl':e['ltl']}) for e in chosen)
        definitions[domain]=dict(mapping=mapping,object_mapping=alias_objects,glossary=domain_profile['glossary'],
            custom_system=('Translate natural language into JSON {"ltl":"..."}. Return data only, '
                'no Python or Markdown. Use standard LTL over infinite traces, ! & | -> X U F G. '
                'Use only the supplied proposition identifiers. Do not invent constraints. '
                'Return {"error":"reason"} for unexpressible intent.\n'+domain_profile['glossary']+
                '\nThe following examples define this dataset vocabulary and annotation conventions:\n'+custom_examples),
            paper_direct_template=(header+'\nDomain vocabulary (reach means the truth of the mapped predicate):\n'+object_text+
                '\nUse standard LTL over infinite traces.\nDataset-specific examples:\n'+direct_examples+
                '\n\nTransform the following instruction into one flat listed-prefix LTL expression. '
                'Output only the expression, not Python, JSON, or prose.\nInstruction: {instruction}\nLTL: '))
        for split,path in [('human',golden/(domain+'.jsonl')),('synthetic',source/(domain+'-synthetic.jsonl'))]:
            rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
            groups=defaultdict(list)
            for index,row in enumerate(rows):
                if normalized_instruction(row['natural']) in banned:continue
                key=row['canonical']
                raw=row.get('raw_ltl') or canonical[key]['raw']
                groups[key].append((index,row,raw))
            selected=rng.sample(sorted(groups),TEST_COUNTS[domain])
            for key in selected:
                index,row,raw=rng.choice(groups[key])
                items.append(dict(id=f'{domain}-{split}-{index}',domain=domain,split=split,
                    source_row=index,instruction=row['natural'],original_reference=raw,
                    reference=render(native_tree(raw,domain),mapping)))
            source_hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [canonical_path,seed_path]:
            source_hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    value=dict(version='pan-shared-fewshot-gpt4o-v1',source_revision=REVISION,seed=20261118,
        example_counts=COUNTS,examples=examples,prompts=definitions,items=items,
        model='openai/gpt-4o-2024-11-20',source_hashes=source_hashes,
        exact_authors_sample=False,independently_reviewed=False,
        note='Examples are unchanged Pan train_seed utterances paired via Pan canonical references. '
             'Exact LTLCodeGen paper example identities are unavailable. Both methods share these examples. '
             'Prompt/test instruction overlap is excluded; formula-family overlap is intentional and recorded.')
    (output/'frozen.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    (output/'upstream_original_translate_prompt.txt').write_text(original,encoding='utf-8')
    (output/'scenegraph_prompt.py').write_bytes((upstream/'scenegraph_prompt.py').read_bytes())
    (output/'ltl_translator.py').write_bytes((upstream/'ltl_translator.py').read_bytes())
    (output/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf-8')
    return value

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--golden',type=Path,required=True)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();data=freeze(args.source,args.golden,args.upstream,args.output)
    print(json.dumps({'example_counts':data['example_counts'],'pilot_items':len(data['items']),
                      'scheduled_outcomes':2*len(data['items'])}))
