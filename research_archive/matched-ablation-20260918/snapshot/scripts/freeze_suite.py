"""Create a new immutable-by-hash experiment input snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root/'snapshot'
    snapshot.mkdir()
    ignore = shutil.ignore_patterns('.git', '__pycache__', '.pytest_cache', '*.egg-info')
    for directory in ('src', 'scripts', 'tests', 'vendor/pyquaticus'):
        shutil.copytree(project/directory, snapshot/directory, ignore=ignore)
    for name in ('pyproject.toml', 'README.md'):
        shutil.copy2(project/name, snapshot/name)
    shutil.copy2(project/'results/phase2-role-spec.json', snapshot/'specification.json')
    (snapshot/'dependencies.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    hashes = {p.relative_to(snapshot).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*')) if p.is_file()}
    plan = dict(seeds=[0,1,2,3,4], steps=100000, actual_agent_transitions=100608,
        actual_joint_decisions=33536, team_size=3, training_opponent='chase',
        training_scenario='standard', evaluation_episodes=30, evaluation_seed=20000,
        evaluation_opponents=['static','patrol','chase'],
        evaluation_scenarios=['standard','wide','narrow'],
        modes=['neural','temporal','reward','shield','full'],
        python=sys.version, platform=platform.platform(), source_hashes=hashes,
        notes='100000 requested individual-agent transitions; identical PPO rollout rounding. '
              'No hyperparameter selection on held-out evaluation. Five independent training seeds. '
              'Shield is arena-bounds only. Shared-policy PPO against scripted red opponents.')
    (root/'protocol.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    (root/'README.md').write_text('''# Frozen matched ablation suite

25 fresh training runs: five modes × seeds 0–4. Each requests 100,000 individual
agent transitions (100,608 after PPO rounding; 33,536 joint 3v3 decisions).
Training uses the standard arena and chasing scripted red opponents.
Each policy evaluates 30 held-out episodes per cell across three maps and three
opponents, with the same evaluation seeds 20000–20029: 6,750 episodes total.

`protocol.json` hashes every frozen source/spec/dependency file. Each run saves
its specification, arena/PPO configs, enabled components, actual budget, policy,
training progress and raw evaluation rows. `status.json` records runner status;
`runner.log` and `runner-errors.log` contain process output. `comparison.json`
is updated after each completed evaluation and is partial until all 25 finish.

Common reward = 20 × captures + 2 × pickups − 2 × entries into tagged state
− 0.01 per executed simulator step. Shaping and the shield abort penalty are
excluded. Aborted episodes still end early and are reported via deadlock rate.
Rule violations count simulator steps where any blue boat is out of bounds,
any blue boat is tagged, or designated defender 0 is across the centerline.
This is the fixed role safety contract, not accumulated absorbing DFA failure.
Tag rate is the fraction of episodes with tags; tags per episode is also given.
LTLf satisfaction includes the full role and ordered capture mission.
Capture time excludes unsuccessful episodes; missing successes are null, not zero.
Report seed means with descriptive bootstrap intervals over training seeds,
not pseudoreplicated episodes. Capture-time intervals use only seeds with
successes, and the number contributing is recorded. Five seeds is a starting
point, not a guarantee of statistical power. No superiority claim is assumed.

Resume the frozen runner using the project virtual environment and
`snapshot/scripts/matched_suite.py <this-directory>`. Completed policies and
evaluations are reused. Incomplete training directories cause a visible failure
and require investigation; the runner never silently overwrites them.
''', encoding='utf-8')
    print(root)


if __name__ == '__main__': main()
