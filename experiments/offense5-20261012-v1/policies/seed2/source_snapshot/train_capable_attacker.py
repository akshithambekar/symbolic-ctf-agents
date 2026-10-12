import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maritime_nsrl.attacker_training import train_attacker

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exploratory imitation-initialized PPO attacker; no formal training guidance")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=4096)
    p.add_argument("--demonstrations", type=int, default=24)
    p.add_argument("--bc-epochs", type=int, default=160)
    args = p.parse_args()
    print(train_attacker(ROOT, args.output, args.seed, args.steps, args.demonstrations, args.bc_epochs))
