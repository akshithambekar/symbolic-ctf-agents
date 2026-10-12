"""Evaluate unchanged saved neural policies with and without the runtime gate."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maritime_nsrl.runtime_shield_study import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--evaluation-seed", type=int, default=710000)
    args = parser.parse_args()
    print(run(ROOT, args.output, args.episodes, tuple(args.training_seeds), args.evaluation_seed))
