"""Short, matched diagnostics for the 1v1 chasing-defender failure mode.

This study deliberately changes neither the mission specification nor the action
set.  It tests whether local approach and evasion feedback can recover flag
acquisition while retaining the successful softened, no-tag-penalty structure.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import numpy as np

from .confirmatory import _write_representatives
from .experiments import evaluate_team, save_json, train_team


STEPS = 20_160
SEEDS = [0, 1]
EVALUATION_EPISODES = 20
FORMULA = "G(in_bounds) & F(has_flag & F(captured))"
PROMOTION_RULE = {
    "defined_before_runs": True,
    "criterion": "Promote only a non-baseline candidate with pooled held-out pickup rate >= 0.10 and mean minimum enemy-flag distance at least 10% below the current no-tag-penalty baseline. This short study is exploratory; promotion recommends, but does not itself launch, a five-seed confirmation.",
    "held_out_episodes_per_seed": EVALUATION_EPISODES,
}


def variants():
    shared = {"capture_reward": 25., "pickup_reward": 8., "tag_penalty": 0.,
              "time_penalty": .005, "potential_scale": 3., "acceptance_bonus": 8.,
              "failure_penalty": 0., "shield": False, "approach_progress_scale": 0.,
              "evasion_scale": 0., "defender_distance_feature": False}
    return {
        "current_no_tag_penalty": {"reward_shaping": True, "temporal_state": True, **shared},
        "dense_flag_progress": {"reward_shaping": True, "temporal_state": True,
                                **{**shared, "approach_progress_scale": 8.}},
        "progress_evasion_distance_feature": {
            "reward_shaping": True, "temporal_state": True,
            **{**shared, "approach_progress_scale": 8., "evasion_scale": 8.,
               "defender_distance_feature": True},
        },
    }


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash_tree(root):
    root = Path(root)
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file() and "__pycache__" not in path.parts}


def initialize(root, source_root, source_spec):
    root, source_root, source_spec = Path(root), Path(source_root), Path(source_spec)
    if root.exists():
        raise FileExistsError(f"Chase-recovery root already exists: {root}")
    root.mkdir(parents=True)
    snapshot = root / "source_snapshot"
    shutil.copytree(source_root / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(source_root / "pyproject.toml", snapshot / "pyproject.toml")
    shutil.copy2(source_spec, root / "specification.json")
    specification = json.loads((root / "specification.json").read_text(encoding="utf-8"))
    if specification["ltl"] != FORMULA:
        raise ValueError(f"Expected softened mission {FORMULA!r}, found {specification['ltl']!r}")
    protocol = {
        "study": "Phase 3 exploratory 1v1 chase recovery",
        "study_type": "exploratory diagnostic; not a confirmatory comparison",
        "created_at": _now(), "source_root": str(source_root), "source_spec_origin": str(source_spec),
        "source_hashes": _hash_tree(snapshot), "formula": FORMULA,
        "team_size": 1, "scenario": "standard", "opponent": "chase", "action_set": "fixed eight waypoint actions",
        "training_steps_per_run": STEPS, "seeds": SEEDS, "evaluation_episodes_per_seed": EVALUATION_EPISODES,
        "evaluation": "deterministic held-out seeds: 71000 + variant_index*1000 + seed*100 + episode",
        "variants": variants(), "promotion_rule": PROMOTION_RULE,
        "matched_controls": ["same PyQuaticus arena, scripted chase opponent, PPO hyperparameters, mission LTL, seeds, budget, and evaluation episodes"],
        "primary_metrics": ["capture rate", "pickup rate", "minimum distance to enemy flag", "tags before pickup", "enemy territory time", "minimum and mean defender distance"],
        "notes": ["No hard shield is enabled.", "Tagging is not an absorbing LTL clause and training tag penalty remains zero.", "The richer action-set idea is deferred so this diagnostic isolates reward and observation effects."],
    }
    save_json(root / "protocol.json", protocol)
    return protocol


def _rate(rows, predicate):
    return float(np.mean([bool(predicate(row)) for row in rows])) if rows else 0.


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in {"", "None", None}]
    return float(np.mean(values)) if values else None


def _aggregate(root):
    entries = []
    for variant in variants():
        for seed in SEEDS:
            path = Path(root) / "diagnostic" / variant / f"seed{seed}" / "evaluation.csv"
            if not path.exists():
                continue
            rows = list(csv.DictReader(path.open(encoding="utf-8")))
            entries.append({
                "variant": variant, "seed": seed, "episodes": len(rows),
                "captures": sum(float(r["captures"]) > 0 for r in rows),
                "capture_rate": _rate(rows, lambda r: float(r["captures"]) > 0),
                "pickup_rate": _rate(rows, lambda r: r.get("pickup_time") not in {"", "None", None}),
                "mean_min_enemy_flag_distance": _mean(rows, "enemy_flag_distance_min"),
                "mean_tags_before_pickup": _mean(rows, "tags_before_pickup"),
                "mean_enemy_territory_time": _mean(rows, "enemy_territory_time"),
                "mean_min_defender_distance": _mean(rows, "defender_distance_min"),
                "mean_defender_distance": _mean(rows, "defender_distance_mean"),
                "mean_common_task_reward": _mean(rows, "common_task_reward"),
            })
    output = Path(root) / "diagnostic"
    with (output / "per_seed_summary.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = list(entries[0]) if entries else ["variant"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(entries)
    summaries=[]
    for variant in variants():
        group=[entry for entry in entries if entry["variant"] == variant]
        if not group:
            continue
        def pooled(name):
            values=[entry[name] for entry in group if entry[name] is not None]
            return float(statistics.mean(values)) if values else None
        summaries.append({"variant":variant,"seeds_completed":len(group),"evaluation_episodes":sum(entry["episodes"] for entry in group),
                          "captures":sum(entry["captures"] for entry in group),
                          "pooled_capture_rate":sum(entry["captures"] for entry in group) / sum(entry["episodes"] for entry in group),
                          "pooled_pickup_rate":pooled("pickup_rate"), "mean_min_enemy_flag_distance":pooled("mean_min_enemy_flag_distance"),
                          "mean_tags_before_pickup":pooled("mean_tags_before_pickup"), "mean_enemy_territory_time":pooled("mean_enemy_territory_time"),
                          "mean_min_defender_distance":pooled("mean_min_defender_distance"), "mean_defender_distance":pooled("mean_defender_distance"),
                          "mean_common_task_reward":pooled("mean_common_task_reward")})
    baseline=next((item for item in summaries if item["variant"] == "current_no_tag_penalty"), None)
    recommendations=[]
    if baseline and baseline["mean_min_enemy_flag_distance"] is not None:
        for item in summaries:
            distance_improvement=(baseline["mean_min_enemy_flag_distance"] - item["mean_min_enemy_flag_distance"]) / max(baseline["mean_min_enemy_flag_distance"], 1e-9)
            recommended=item["variant"] != "current_no_tag_penalty" and item["pooled_pickup_rate"] >= .10 and distance_improvement >= .10
            recommendations.append({"variant": item["variant"], "distance_improvement_vs_baseline": distance_improvement,
                                    "recommended_for_five_seed_confirmation": recommended})
    save_json(output / "diagnostic_summary.json", {"completed_at":_now(),"per_seed":entries,"variants":summaries,
                                                     "promotion_rule":PROMOTION_RULE,"recommendations":recommendations})
    return summaries, recommendations


def run(root):
    root=Path(root)
    if not (root / "protocol.json").exists():
        raise RuntimeError("Initialize the study first")
    directory=root / "diagnostic"
    if directory.exists():
        raise FileExistsError(f"Diagnostic directory already exists: {directory}")
    directory.mkdir()
    save_json(directory / "plan.json", {"created_at":_now(),"variants":variants(),"seeds":SEEDS,"steps":STEPS,
                                          "evaluation_episodes":EVALUATION_EPISODES,"source_hashes":_hash_tree(root / "source_snapshot")})
    events=directory / "execution.jsonl"
    for variant, definition in variants().items():
        for seed in SEEDS:
            run_dir=directory / variant / f"seed{seed}"
            with events.open("a",encoding="utf-8") as handle:
                handle.write(json.dumps({"at":_now(),"event":"started","variant":variant,"seed":seed})+"\n")
            overrides={key:value for key,value in definition.items() if key not in {"shield","reward_shaping","temporal_state"}}
            policy=train_team(root / "specification.json",run_dir,team_size=1,opponent="chase",scenario="standard",shield=False,
                              steps=STEPS,seed=seed,reward_shaping=True,temporal_state=True,config_overrides=overrides)
            trace=run_dir / "trajectories.jsonl"
            variant_index=list(variants()).index(variant)
            evaluate_team(policy,run_dir / "evaluation.csv",episodes=EVALUATION_EPISODES,
                          seed=71000+variant_index*1000+seed*100,opponents=("chase",),scenarios=("standard",),
                          trace_output=trace,trace_stride=1)
            _write_representatives(trace,run_dir / "evaluation.csv",run_dir / "representative_replays")
            with events.open("a",encoding="utf-8") as handle:
                handle.write(json.dumps({"at":_now(),"event":"completed","variant":variant,"seed":seed})+"\n")
    summaries,recommendations=_aggregate(root)
    (directory / "REPORT.md").write_text("# 1v1 chase-recovery diagnostic\n\nThis is an exploratory two-seed diagnostic. It is not evidence of superiority; candidates meeting the pre-registered screen require a fresh five-seed matched confirmation. See `diagnostic_summary.json`, `per_seed_summary.csv`, raw evaluation files, training curves, and replay traces.\n",encoding="utf-8")
    return {"variants":summaries,"recommendations":recommendations}
