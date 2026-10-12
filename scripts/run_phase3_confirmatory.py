"""Initialize, verify, or run one pre-registered Phase 3 confirmatory stage."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))

from maritime_nsrl.confirmatory import STAGES, initialize, preflight, run_stage, resume_stage_after_technical_fix, continue_stage02_after_numerical_fix, resume_stage02_after_interruption

parser=argparse.ArgumentParser()
parser.add_argument("command",choices=["initialize","preflight","run-stage","recover-stage02","continue-stage02","resume-stage02"])
parser.add_argument("--root",type=Path,required=True)
parser.add_argument("--stage",choices=sorted(STAGES))
args=parser.parse_args()
if args.command=="initialize":
    print(initialize(args.root,ROOT,ROOT / "research_archive" / "offensive-phase-20260924-v1" / "snapshot" / "specification.json"))
elif args.command=="preflight": print(preflight(args.root))
elif args.command=="run-stage":
    if not args.stage: parser.error("--stage is required for run-stage")
    print(run_stage(args.root,args.stage))
elif args.command=="recover-stage02":
    print(resume_stage_after_technical_fix(args.root,"02-1v1-patrol"))
elif args.command=="continue-stage02":
    print(continue_stage02_after_numerical_fix(args.root))
else:
    print(resume_stage02_after_interruption(args.root))
