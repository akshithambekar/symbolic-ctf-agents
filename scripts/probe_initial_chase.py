"""Post-run lower-speed reference: distinguish existing ability from learning."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from stable_baselines3 import PPO
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from maritime_nsrl.research_env import ArenaConfig
from maritime_nsrl.attacker_training import evaluate_attacker
from maritime_nsrl.runtime_shield_study import csv_rows, save


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    save(output/'protocol.json',{'type':'post-hoc unmodified-checkpoint reference for easier chase stages',
        'speeds':[.65,.85],'episodes_per_seed_speed':5,'training_seeds':[0,1],
        'evaluation_seeds':'940000 + stage*100 + seed*10; identical to staged capability screens',
        'note':'Not used to change promotion. Passing the capability screen is not by itself evidence that continuation improved performance.'})
    save(output/'source_hashes.json',{'script':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').rglob('*.py')}})
    rows=[]
    for seed in [0,1]:
        source=ROOT/f'experiments/tactical-attacker-20261012-v1/attackers/seed{seed}'
        model=PPO.load(source/'policy.zip',device='cpu')
        cfg=ArenaConfig(**json.loads((source/'arena_config.json').read_text(encoding='utf-8')));cfg.opponent='chase'
        for stage,speed in enumerate([.65,.85]):
            cfg.opponent_speed=speed
            result=evaluate_attacker(model,cfg,output/f'seed{seed}-speed{speed}.csv',5,940000+stage*100+seed*10)
            rows.append({'training_seed':seed,'speed':speed,**result})
    csv_rows(output/'summary.csv',rows);save(output/'summary.json',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);run(a.output)
