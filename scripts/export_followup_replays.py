"""Extract full recorded snapshots for the existing native simulator viewer."""
import argparse
import json
from pathlib import Path


def export(root):
    output = root/'replays';output.mkdir(exist_ok=False)
    for line in (root/'exposure/traces.jsonl').read_text(encoding='utf-8').splitlines():
        record=json.loads(line);identity=record['identity']
        if identity['training_seed']!=0 or identity['opponent']!='static':continue
        if identity['condition'] not in {'approach_tight','approach_reserve','return_tight'}:continue
        path=output/f"{identity['condition']}-{identity['arm']}.jsonl"
        frames=[record['initial'],*record['frames']]
        path.write_text(''.join(json.dumps(f)+'\n' for f in frames),encoding='utf-8')
    for source in (root/'conflicts').glob('*.jsonl'):
        frames=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
        frames=[f for f in frames if 'positions' in f and 'version' in f]
        (output/f"conflict-{source.name}").write_text(''.join(json.dumps(f)+'\n' for f in frames),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);export(p.parse_args().root)
