"""Generate the evolving Phase 3 diagnosis and plots from preserved raw data."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("phase",type=Path); parser.add_argument("phase2",type=Path)
    a=parser.parse_args(); phase=a.phase.resolve(); prior=a.phase2.resolve()
    modes=["neural","temporal","reward","shield","full"]
    labels=["Neural","Temporal state","Reward shaping","Shield","Full"]
    old=[]
    for mode in modes:
        rows=[r for f in prior.glob(mode+"-seed*/evaluation.csv") for r in csv.DictReader(f.open())]
        old.append({"capture":np.mean([float(r["captures"])>0 for r in rows]),
            "pickup":np.mean([float(r["pickups"])>0 for r in rows]),
            "reward":np.mean([float(r["common_task_reward"]) for r in rows]),
            "violations":np.mean([float(r["rule_violation_steps"]) for r in rows]),
            "guidance":np.mean([float(r["guidance_reward"]) for r in rows])})
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    axes[0].bar(labels,[100*x["pickup"] for x in old],label="Pickup")
    axes[0].bar(labels,[100*x["capture"] for x in old],label="Capture")
    axes[0].set_ylabel("Episodes (%)"); axes[0].legend(); axes[0].set_title("Offensive completion")
    axes[1].bar(labels,[x["guidance"] for x in old]); axes[1].axhline(0,color="black",lw=.7)
    axes[1].set_title("Mean guidance return")
    axes[2].bar(labels,[x["violations"] for x in old]); axes[2].set_title("Safety violations / episode")
    for ax in axes: ax.tick_params(axis="x",rotation=30)
    fig.tight_layout(); fig.savefig(phase/"phase2_diagnosis.png",dpi=170); plt.close(fig)

    oracle=json.loads((phase/"oracle/summary.json").read_text())
    standard=[r for r in oracle if r["scenario"]=="standard"]
    stages=["Territory","Flag vicinity","Pickup","Return midfield","Capture"]
    keys=["enemy_territory_rate","enemy_flag_rate","pickup_rate","return_midfield_rate","capture_rate"]
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharey=True)
    for ax,row in zip(axes.flat,standard):
        ax.bar(stages,[100*row[k] for k in keys]); ax.tick_params(axis="x",rotation=35)
        ax.set_title(f"{row['team_size']}v{row['team_size']} / {row['opponent']}"); ax.set_ylim(0,105)
    axes[0,0].set_ylabel("Episodes (%)"); axes[1,0].set_ylabel("Episodes (%)")
    fig.tight_layout(); fig.savefig(phase/"oracle_funnel_standard.png",dpi=170); plt.close(fig)

    exploratory="Not complete yet."
    ep=phase/"exploratory_results.json"
    if ep.exists():
        records=json.loads(ep.read_text())
        exploratory="\n".join(f"| {r['stage']} | {r['variant']} | {r['captures']}/{r['episodes']} | {100*r['capture_rate']:.1f}% | {r['promoted']} |" for r in records)
    report=f"""# Phase 3 diagnosis: offensive recovery

## Evidence from the immutable matched study

The main failure occurs before flag acquisition. Pickup rates were {old[0]['pickup']:.1%}
(neural), {old[1]['pickup']:.1%} (temporal state), {old[2]['pickup']:.1%}
(reward shaping), {old[3]['pickup']:.1%} (shield), and {old[4]['pickup']:.1%}
(full guidance). Full guidance never picked up a flag. Reward shaping contributed
{old[2]['guidance']:.2f} mean return in the shaping-only condition and
{old[4]['guidance']:.2f} in the full condition. Against a +2 pickup reward, these
large negative shaping totals can dominate the sparse offensive signal. The full
policy's favorable common reward is partly associated with conservative behavior
and early shield termination; it did not complete the mission.

## Scripted feasibility and failure stages

The discrete action abstraction is sufficient for reliable 1v1 static capture in
all three arenas. A coordinated two-attacker 3v3 policy also captures reliably in
standard static play by using the defender's tag cooldown. Thus zero learned 3v3
captures are not caused by an impossible simulator or missing capture action.

Patrol failures generally happen after pickup on the return leg. In standard 1v1,
the simple oracle reached/picked up the flag in 5/5 episodes but captured in 1/5.
Chase failures happen earlier: it crossed midfield in every episode but never
reached the flag in any tested map or team size. This separates the research
problem into return-path robustness against patrol and approach/evasion against chase.

The oracle is deliberately simple and failure does not prove a configuration is
physically impossible. It only identifies what this action abstraction and tested
heuristic could demonstrate. Full raw trajectories are retained.

## Current exploratory screen

| Stage | Variant | Captures | Rate | Promoted |
|---|---|---:|---:|---|
{exploratory}

The screen starts with 1v1 static using two seeds and 30,720 transitions. Variants
must reach 10% capture over 40 held-out episodes to advance. The initial candidates
are neural, temporal-state-only, and soft guidance without `G(!tagged)` or a fixed
defender. Soft guidance increases pickup/capture reward, reduces tag/time penalties,
and removes absorbing-failure penalties. No confirmatory claim will be made from
this exploratory screen.

## Recommendation gate

Do not schedule a five-seed confirmatory matrix until a candidate captures reliably
in 1v1 static and patrol, shows a nonzero approach/pickup rate against chase, and
does not materially worsen tags or fixed safety violations. Shield variants should
be reconsidered only after action-substitution traces show where interventions occur;
the current arena-bound shield is not an offensive constraint and its deadlocks can
artificially shorten episodes.
"""
    (phase/"DIAGNOSIS.md").write_text(report,encoding="utf-8")


if __name__=="__main__": main()
