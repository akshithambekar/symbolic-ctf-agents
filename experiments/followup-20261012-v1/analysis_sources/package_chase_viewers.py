"""Add canonical inference bundles without changing raw training artifacts."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def package(root):
    output=root/'viewer_policies';output.mkdir(exist_ok=False)
    for source in sorted((root/'chase').iterdir()):
        if not (source/'policy.zip').exists():continue
        raw=json.loads((source/'config.json').read_text(encoding='utf-8'))
        seed=int(source.name[-1]);origin=ROOT/f'experiments/tactical-attacker-20261012-v1/attackers/seed{seed}'
        dest=output/source.name;dest.mkdir()
        shutil.copy2(source/'policy.zip',dest/'policy.zip')
        shutil.copy2(origin/'specification.json',dest/'specification.json')
        arena=dict(raw['arena']);arena['opponent']='chase';arena['opponent_speed']=1.
        (dest/'arena_config.json').write_text(json.dumps(arena,indent=2),encoding='utf-8')
        (dest/'config.json').write_text(json.dumps({'mode':'team','team_size':1,'seed':seed,
            'training_method':'imitation-initialized PPO followed by staged chase continuation',
            'variant':raw['variant'],'source_training_config':str(source/'config.json')},indent=2),encoding='utf-8')
        (dest/'origin.json').write_text(json.dumps({'policy_sha256':hashlib.sha256((source/'policy.zip').read_bytes()).hexdigest(),
            'source_policy':str(source/'policy.zip'),'source_config_sha256':hashlib.sha256((source/'config.json').read_bytes()).hexdigest(),
            'note':'Canonical inference bundle. Standard TeamArena has identical observations/actions/physics; training-only reward terms are unnecessary for policy inference.'},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);package(p.parse_args().root)
