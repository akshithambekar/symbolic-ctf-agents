"""Run the exploratory 1v1 chasing-defender recovery diagnostic."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))

from maritime_nsrl.chase_recovery import initialize, run

parser=argparse.ArgumentParser()
parser.add_argument("command", choices=["initialize", "run"])
parser.add_argument("--root",type=Path,required=True)
parser.add_argument("--source-spec",type=Path,default=ROOT / "research_archive" / "phase3-confirmatory-20260927-v1" / "specification.json")
args=parser.parse_args()

if args.command == "initialize":
    print(initialize(args.root, ROOT, args.source_spec))
else:
    print(run(args.root))
