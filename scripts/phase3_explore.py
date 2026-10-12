"""Small, resumable curriculum screen; promotion requires easier-stage captures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

VARIANTS = {
    "neural": dict(reward_shaping=False, temporal_state=False, config_overrides={}),
    "temporal_soft": dict(reward_shaping=False, temporal_state=True, config_overrides={}),
    "soft_guidance": dict(reward_shaping=True, temporal_state=True, config_overrides={
        "capture_reward":25., "pickup_reward":8., "tag_penalty":1., "time_penalty":.005,
        "potential_scale":3., "acceptance_bonus":8., "failure_penalty":0.}),
}
STAGES = [(1,"static"),(1,"patrol"),(1,"chase"),(3,"static"),(3,"patrol"),(3,"chase")]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    args=parser.parse_args(); root=args.directory.resolve()
    protocol=json.loads((root/"protocol.json").read_text())
    snapshot=root/"snapshot"; sys.path[:0]=[str(snapshot/"src"),str(snapshot/"vendor/pyquaticus")]
    for name,digest in protocol["source_hashes"].items():
        if hashlib.sha256((snapshot/name).read_bytes()).hexdigest()!=digest: raise ValueError(name)
    from maritime_nsrl.experiments import train_team,evaluate_team
    active=set(VARIANTS)
    records=[]
    for stage_index,(team_size,opponent) in enumerate(STAGES):
        if not active: break
        stage=f"stage{stage_index+1}-{team_size}v{team_size}-{opponent}"
        next_active=set()
        for variant in sorted(active):
            captures=episodes=0
            for seed in protocol["exploratory_seeds"]:
                run=root/"runs"/stage/variant/f"seed{seed}"
                opts=VARIANTS[variant]
                if not (run/"policy.zip").exists():
                    train_team(snapshot/"specification.json",run,team_size=team_size,
                        opponent=opponent,scenario="standard",steps=protocol["short_steps"],
                        seed=seed,horizon=240,shield=False,**opts)
                evaluation=run/"evaluation.csv"
                if not evaluation.exists():
                    evaluate_team(run/"policy.zip",evaluation,
                        episodes=protocol["short_evaluation_episodes"],seed=40000,
                        opponents=(opponent,),scenarios=("standard",),
                        trace_output=run/"trajectories.jsonl",trace_stride=2)
                rows=list(csv.DictReader(evaluation.open()))
                captures+=sum(float(r["captures"])>0 for r in rows); episodes+=len(rows)
            rate=captures/episodes
            promoted=rate>=protocol["promotion_capture_rate"]
            records.append({"stage":stage,"variant":variant,"captures":captures,
                            "episodes":episodes,"capture_rate":rate,"promoted":promoted})
            if promoted: next_active.add(variant)
            (root/"exploratory_results.json").write_text(json.dumps(records,indent=2))
        active=next_active


if __name__ == "__main__": main()
