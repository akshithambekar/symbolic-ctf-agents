"""Sequential, resumable matched ablations. Run only against a frozen snapshot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

MODES = {
    "neural": (False, False, False),
    "temporal": (False, True, False),
    "reward": (True, False, False),
    "shield": (False, False, True),
    "full": (True, True, True),
}


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def summarize(root):
    """Bootstrap training-seed means, never treat episodes as independent seeds."""
    cells = {}
    for file in sorted(root.glob('*/evaluation.csv')):
        config = json.loads((file.parent / 'run_contract.json').read_text())
        for row in csv.DictReader(file.open(newline='', encoding='utf-8')):
            key = (config['mode'], row['scenario'], row['opponent'])
            cells.setdefault(key, {}).setdefault(config['seed'], []).append(row)
    rng = np.random.default_rng(913)
    results = []
    for (mode, scenario, opponent), seeds in sorted(cells.items()):
        metrics = {}
        extractors = {
            'capture_rate': lambda r: float(float(r['captures']) > 0),
            'common_reward': lambda r: float(r['common_task_reward']),
            'tags_per_episode': lambda r: float(r['tags_received']),
            'episode_tag_rate': lambda r: float(float(r['tags_received']) > 0),
            'rule_violation_steps': lambda r: float(r['rule_violation_steps']),
            'ltlf_satisfaction_rate': lambda r: float(r['ltlf_satisfied'] == 'True'),
            'successful_capture_time': lambda r: float(r['capture_time']) if r['capture_time'] else None,
            'shield_interventions': lambda r: float(r['shield_interventions']),
            'shield_deadlock_rate': lambda r: float(float(r['shield_deadlocks']) > 0),
        }
        for name, extract in extractors.items():
            values = []
            for rows in seeds.values():
                samples = [v for r in rows if (v := extract(r)) is not None]
                if samples: values.append(float(np.mean(samples)))
            interval = None
            if len(values) >= 5:
                draws = rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
                interval = np.quantile(draws, [.025, .975]).tolist()
            metrics[name] = {'mean_of_seed_means': float(np.mean(values)) if values else None,
                             'contributing_seeds': len(values), 'bootstrap_95_ci': interval}
        results.append(dict(mode=mode, scenario=scenario, opponent=opponent,
                            completed_seeds=len(seeds), metrics=metrics))
    save(root / 'comparison.json', results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    plan = json.loads((root / 'protocol.json').read_text())
    snapshot = root / 'snapshot'
    for name, expected in plan['source_hashes'].items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source changed: {name}')
    sys.path.insert(0, str(snapshot / 'src'))
    sys.path.insert(0, str(snapshot / 'vendor' / 'pyquaticus'))
    from maritime_nsrl.experiments import train_team, evaluate_team
    import maritime_nsrl
    if not Path(maritime_nsrl.__file__).resolve().is_relative_to(snapshot):
        raise RuntimeError('Wrong source import')
    start = time.time()
    completed = 0
    try:
        for seed in plan['seeds']:
            for mode, (reward, temporal, shield) in MODES.items():
                run = root / f'{mode}-seed{seed}'
                contract = dict(mode=mode, seed=seed, reward_shaping=reward,
                    temporal_state=temporal, shield=shield, shield_formula='G(in_bounds)',
                    steps=plan['steps'], team_size=3, opponent='chase', scenario='standard',
                    evaluation_episodes=plan['evaluation_episodes'], evaluation_seed=20000,
                    protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest())
                save(root/'status.json', dict(state='running', current_run=run.name,
                    completed_runs=completed, total_runs=len(plan['seeds'])*len(MODES), elapsed_seconds=time.time()-start))
                print(f'Starting {run.name}', flush=True)
                if not (run/'policy.zip').exists():
                    if run.exists():
                        raise RuntimeError(f'Incomplete run {run}; preserve it and resolve before resuming')
                    train_team(snapshot/'specification.json', run, team_size=3,
                        steps=plan['steps'], seed=seed, shield=shield,
                        reward_shaping=reward, temporal_state=temporal)
                if (run/'run_contract.json').exists():
                    if json.loads((run/'run_contract.json').read_text()) != contract:
                        raise ValueError('Run contract changed')
                else:
                    save(run/'run_contract.json', contract)
                if not (run/'evaluation.csv').exists():
                    evaluate_team(run/'policy.zip', run/'evaluation.csv',
                        episodes=plan['evaluation_episodes'], seed=20000)
                summarize(root)
                completed += 1
        save(root/'status.json', dict(state='complete', completed_runs=completed,
                                      elapsed_seconds=time.time()-start))
    except Exception:
        save(root/'status.json', dict(state='failed', completed_runs=completed,
                                     traceback=traceback.format_exc()))
        raise


if __name__ == '__main__': main()
