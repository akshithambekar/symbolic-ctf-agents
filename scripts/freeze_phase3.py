"""Freeze a new offensive-performance phase without touching Phase 2."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("output",type=Path)
    args=parser.parse_args(); project=Path(__file__).resolve().parents[1]
    root=args.output.resolve(); root.mkdir(parents=True,exist_ok=False)
    snapshot=root/"snapshot"; snapshot.mkdir()
    ignore=shutil.ignore_patterns(".git","__pycache__",".pytest_cache","*.egg-info")
    for directory in ("src","scripts","tests","vendor/pyquaticus"):
        shutil.copytree(project/directory,snapshot/directory,ignore=ignore)
    for name in ("pyproject.toml","README.md"):
        shutil.copy2(project/name,snapshot/name)
    shutil.copy2(project/"results/live-translation.json",snapshot/"specification.json")
    hashes={p.relative_to(snapshot).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(snapshot.rglob("*")) if p.is_file()}
    protocol={"phase":"exploratory","research_question":"Can safety benefits be retained while recovering capture performance?",
        "exploratory_seeds":[0,1],"short_steps":30720,"short_evaluation_episodes":20,
        "promotion_capture_rate":.10,"training_scenario":"standard",
        "curriculum":["1v1-static","1v1-patrol","1v1-chase","3v3-static","3v3-patrol","3v3-chase"],
        "source_hashes":hashes,"prior_suite":"research_archive/matched-ablation-20260918-v1 (immutable)"}
    (root/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    (root/"README.md").write_text("""# Phase 3: offensive recovery

This is a fresh exploratory phase. The completed Phase 2 suite is an immutable
input and is never modified. First, a scripted oracle tests physical feasibility
and identifies the failing offensive stage. Then three low-cost variants use two
seeds at 30,720 transitions. A variant advances only after at least 10% captures
over 40 held-out episodes at the easier stage. Failed variants stop, preventing
an expensive matrix with no evidence of learnability.

The mission specification retains `G(in_bounds)` and ordered pickup-before-capture,
but removes the absorbing no-tag and global fixed-defender clauses. Tagging remains
a measured outcome and task penalty. `soft_guidance` increases pickup/capture signal,
reduces tag and time penalties, removes absorbing-failure penalties, and keeps the
temporal-state input. This phase is exploratory; it cannot support superiority claims.
""",encoding="utf-8")
    print(root)


if __name__=="__main__": main()
