import csv
import importlib.util
import json
from pathlib import Path


def test_summary_uses_common_reward_and_seed_level_uncertainty(tmp_path):
    spec = importlib.util.spec_from_file_location('matched_suite', Path(__file__).parents[1]/'scripts/matched_suite.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for seed in range(5):
        run = tmp_path/f'neural-seed{seed}'
        run.mkdir()
        (run/'run_contract.json').write_text(json.dumps(dict(mode='neural', seed=seed)))
        row = dict(scenario='standard', opponent='chase', captures=0,
                   common_task_reward=seed, episode_reward=999, tags_received=1,
                   rule_violation_steps=2, ltlf_satisfied=False, capture_time='',
                   shield_interventions=0, shield_deadlocks=0)
        with (run/'evaluation.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    module.summarize(tmp_path)
    metrics = json.loads((tmp_path/'comparison.json').read_text())[0]['metrics']
    assert metrics['common_reward']['mean_of_seed_means'] == 2
    assert metrics['common_reward']['contributing_seeds'] == 5
    assert metrics['common_reward']['bootstrap_95_ci'] is not None
    assert metrics['successful_capture_time']['mean_of_seed_means'] is None
