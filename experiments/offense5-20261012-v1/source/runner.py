"""Five-seed matched replication of the selected offensive exposure condition."""
import argparse
import copy
import hashlib
import json
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from maritime_nsrl.attacker_training import train_attacker
from maritime_nsrl.followup import Exposure, ExposureArena
from maritime_nsrl.research_env import ArenaConfig
from maritime_nsrl.runtime_shield_study import csv_rows, save
from maritime_nsrl.viewer import snapshot


def run(output,screen):
    rows=json.loads((screen/'exposure/episodes.json').read_text(encoding='utf-8'))
    off=[r for r in rows if r['condition']=='approach_tight' and r['arm']=='off']
    on=[r for r in rows if r['condition']=='approach_tight' and r['arm']=='original']
    assert len(off)==len(on)==6
    assert sum(r['captures']>0 for r in on)>=sum(r['captures']>0 for r in off)>0
    assert sum(r['out_of_bounds_steps']>0 for r in on)<sum(r['out_of_bounds_steps']>0 for r in off)
    assert sum(r['shield_deadlocks'] for r in on)==0 and sum(r['offensive_events'] for r in on)>0
    output.mkdir(parents=True,exist_ok=False)
    save(output/'protocol.json',{'type':'five-seed matched replication of selected engineered condition',
        'selection':'original gate passed offensive screen; brake was redundant and proactive aborted',
        'selection_data_sha256':hashlib.sha256((screen/'exposure/episodes.json').read_bytes()).hexdigest(),
        'seeds':[0,1,2,3,4],'new_training_seeds':[2,3,4],'reused_selection_checkpoints':[0,1],
        'training':'same 24 oracle episodes,160 cloning epochs,4096 PPO decisions,static/standard,unguided',
        'evaluation_seed_base':960000,'evaluation_episodes_per_seed_opponent_arm':5,
        'opponents':['static','patrol','chase'],'arms':['off','original'],'exposure':asdict(Exposure('approach',2.01,1.)),
        'primary':'bounds-violation episode rate, capture rate, deadlocks, offensive interventions',
        'interpretation':'Selected-condition replication, not an unselected general superiority comparison. Two checkpoints were used in selection; three are new independent training replicates. Fresh evaluation seeds for all.'})
    snap=output/'source';shutil.copytree(ROOT/'src',snap/'src',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__,snap/'runner.py')
    save(output/'source_hashes.json',{p.relative_to(snap).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.rglob('*') if p.is_file()})
    policies=output/'policies';policies.mkdir()
    for seed in (0,1):
        source=ROOT/f'experiments/tactical-attacker-20261012-v1/attackers/seed{seed}'
        dest=policies/f'seed{seed}';dest.mkdir()
        for name in ['policy.zip','arena_config.json','config.json','specification.json','status.json','training_complete.json']:
            shutil.copy2(source/name,dest/name)
        save(dest/'origin.json',{'reused':True,'source':str(source),'policy_sha256':hashlib.sha256((source/'policy.zip').read_bytes()).hexdigest()})
    for seed in (2,3,4):
        checkpoint=train_attacker(ROOT,policies/f'seed{seed}',seed)
        if checkpoint is None:
            save(output/'status.json',{'status':'stopped: new attacker failed static screen','seed':seed})
            return
    rows=[]
    with (output/'traces.jsonl').open('x',encoding='utf-8') as traces:
        for seed in range(5):
            directory=policies/f'seed{seed}';model=PPO.load(directory/'policy.zip',device='cpu')
            original=ArenaConfig(**json.loads((directory/'arena_config.json').read_text(encoding='utf-8')))
            for opponent in ['static','patrol','chase']:
                for ep in range(5):
                    for arm in ['off','original']:
                        config=copy.deepcopy(original);config.opponent=opponent;config.shield=arm=='original'
                        arena=ExposureArena(config,Exposure('approach',2.01,1.))
                        evaluation_seed=960000+seed*100+ep
                        random.seed(evaluation_seed);np.random.seed(evaluation_seed);torch.manual_seed(evaluation_seed)
                        frames=[]
                        try:
                            obs=arena.reset(evaluation_seed);initial=snapshot(arena,[],ep,True,0)
                            while not arena.finished:
                                actions,_=model.predict(obs,deterministic=True)
                                if arena.gate:arena.gate.waypoints=list(map(int,actions))
                                caps=int(arena.game.state['captures'][0]);obs,_,_,info=arena.step(actions)
                                frames.append(snapshot(arena,actions,ep,True,caps))
                            metrics=info['metrics'];metrics.pop('trajectory',None)
                            events=arena.gate.events if arena.gate else []
                            row={'training_seed':seed,'evaluation_seed':evaluation_seed,'arm':arm,**metrics,
                                'offensive_events':sum(e['enemy_territory'] and not e['deadlock'] for e in events)}
                            rows.append(row)
                            traces.write(json.dumps({'identity':{k:row[k] for k in ['training_seed','evaluation_seed','opponent','arm']},
                                'config':asdict(config),'initial':initial,'frames':frames,'events':events})+'\n');traces.flush()
                        finally:arena.close()
                print(f'confirmation seed{seed}/{opponent}',flush=True)
    csv_rows(output/'episodes.csv',rows);save(output/'episodes.json',rows)
    save(output/'status.json',{'status':'completed five-seed selected-condition replication','episodes':len(rows)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--screen',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);run(a.output,a.screen)
