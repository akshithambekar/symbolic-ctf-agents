"""Frozen, staged Phase 3 confirmatory study runner.

The runner never overwrites a run.  Stages are deliberately started one at a
time, after reviewing a pre-registered promotion decision from the prior stage.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import numpy as np

from .experiments import evaluate_team, save_json, train_team
from .research_env import ArenaConfig, TeamArena

STEPS = 100_608
EVALUATION_EPISODES = 30
SEEDS = [0, 1, 2, 3, 4]
PROMOTION_RULE = {
    "defined_before_runs": True,
    "criterion": "A method advances only if at least 3 of 5 training seeds have held-out capture rate >= 0.10 and the pooled rate over 150 deterministic held-out episodes is >= 0.10.",
    "held_out_episodes_per_seed": EVALUATION_EPISODES,
}
STAGES = {
    "02-1v1-patrol": {"team_size": 1, "opponent": "patrol", "scenario": "standard", "previous": None},
    "03-1v1-chase": {"team_size": 1, "opponent": "chase", "scenario": "standard", "previous": "02-1v1-patrol"},
    "04-3v3-static": {"team_size": 3, "opponent": "static", "scenario": "standard", "previous": "03-1v1-chase"},
    "05-3v3-patrol": {"team_size": 3, "opponent": "patrol", "scenario": "standard", "previous": "04-3v3-static"},
    "06-3v3-chase": {"team_size": 3, "opponent": "chase", "scenario": "standard", "previous": "05-3v3-patrol"},
}


def method_definitions():
    # Training rewards preserve the successful soft exploratory configuration.
    # Only the tag coefficient changes between the two softened arms.
    shared = {"capture_reward": 25., "pickup_reward": 8., "time_penalty": .005,
              "potential_scale": 3., "acceptance_bonus": 8., "failure_penalty": 0.,
              "shield": False}
    return {
        "neural": {"reward_shaping": False, "temporal_state": False,
                   "tag_penalty": 1., **shared},
        "temporal_state_only": {"reward_shaping": False, "temporal_state": True,
                                "tag_penalty": 1., **shared},
        "softened_guidance": {"reward_shaping": True, "temporal_state": True,
                                "tag_penalty": 1., **shared},
        "softened_guidance_no_tag_penalty": {"reward_shaping": True, "temporal_state": True,
                                               "tag_penalty": 0., **shared},
    }


def _hash_tree(root):
    root = Path(root)
    rows = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        rows[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows


def _now():
    return datetime.now(timezone.utc).isoformat()


def initialize(root, source_root, source_spec):
    root, source_root, source_spec = Path(root), Path(source_root), Path(source_spec)
    if root.exists():
        raise FileExistsError(f"Confirmatory root already exists: {root}")
    root.mkdir(parents=True)
    snapshot = root / "source_snapshot"
    shutil.copytree(source_root / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(source_root / "pyproject.toml", snapshot / "pyproject.toml")
    shutil.copy2(source_spec, root / "specification.json")
    protocol = {
        "study": "Phase 3 confirmatory softened-guidance curriculum",
        "created_at": _now(), "source_root": str(source_root), "source_spec_origin": str(source_spec),
        "source_hashes": _hash_tree(snapshot), "steps_per_training_run": STEPS,
        "seeds": SEEDS, "evaluation_episodes_per_seed": EVALUATION_EPISODES,
        "evaluation": "deterministic; evaluation seeds are 50000 + 1000*stage_index + seed*100 + episode",
        "formula": json.loads((root / "specification.json").read_text(encoding="utf-8"))["ltl"],
        "methods": method_definitions(), "stages": STAGES, "promotion_rule": PROMOTION_RULE,
        "common_evaluation_reward": "20*capture + 2*pickup - 2*tag-event - 0.01 per simulator step; fixed for every method",
        "notes": ["No hard G(!tagged) clause is used in this softened study.",
                  "The no-tag arm differs only by training tag_penalty=0.0.",
                  "3v3 uses shared-policy PPO, not MAPPO or a centralized critic.",
                  "Stages require an explicit command; the runner never promotes automatically."],
    }
    save_json(root / "protocol.json", protocol)
    save_json(root / "status.json", {"created_at": _now(), "status": "initialized"})
    return protocol


def preflight(root):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    smoke = root / "preflight" / "softened_guidance_seed999"
    if smoke.exists():
        raise FileExistsError("Preflight already exists; preserve it rather than replacing it")
    config = method_definitions()["softened_guidance"]
    policy = train_team(root / "specification.json", smoke, team_size=1, opponent="patrol",
                        scenario="standard", shield=False, steps=512, seed=999,
                        reward_shaping=config["reward_shaping"], temporal_state=config["temporal_state"],
                        config_overrides={k:v for k,v in config.items() if k not in {"shield", "reward_shaping", "temporal_state"}})
    evaluate_team(policy, smoke / "evaluation.csv", episodes=2, seed=49000,
                  opponents=("patrol",), scenarios=("standard",),
                  trace_output=smoke / "trajectories.jsonl", trace_stride=1)
    rows = list(csv.DictReader((smoke / "evaluation.csv").open(encoding="utf-8")))
    required = {"common_task_reward", "reached_enemy_territory", "enemy_flag_distance_min",
                "tags_before_pickup", "tags_after_pickup", "temporal_state_visits",
                "requested_action_counts", "executed_action_counts"}
    missing = sorted(required - set(rows[0])) if rows else sorted(required)
    report = {"completed_at": _now(), "passed": not missing and (smoke / "trajectories.jsonl").exists(),
              "missing_metrics": missing, "rows": len(rows), "replay": str(smoke / "trajectories.jsonl"),
              "planned_stage": "02-1v1-patrol", "planned_runs": len(SEEDS)*len(method_definitions()),
              "planned_evaluation_episodes": len(SEEDS)*len(method_definitions())*EVALUATION_EPISODES,
              "training_agent_transitions": len(SEEDS)*len(method_definitions())*STEPS}
    save_json(root / "preflight.json", report)
    if not report["passed"]:
        raise RuntimeError(f"Preflight failed: {report}")
    return report


def _allowed_methods(root, stage):
    previous = STAGES[stage]["previous"]
    if previous is None:
        return list(method_definitions())
    decision = Path(root) / "stages" / previous / "promotion.json"
    if not decision.exists():
        raise RuntimeError(f"Stage {stage} requires reviewed {decision}")
    return json.loads(decision.read_text(encoding="utf-8"))["promoted_methods"]


def _write_representatives(trace_path, metrics_path, output):
    records = [json.loads(line) for line in Path(trace_path).read_text(encoding="utf-8").splitlines() if line]
    metrics = {int(row["evaluation_seed"]): row for row in csv.DictReader(Path(metrics_path).open(encoding="utf-8"))}
    # evaluate_team stores trajectories and episode metrics in separate artifacts.
    # Join them by held-out evaluation seed for representative selection.
    records = [{**record, "metrics": metrics[int(record["evaluation_seed"])]} for record in records]
    output = Path(output); output.mkdir(exist_ok=True)
    predicates = {
        "success": lambda r: float(r["metrics"]["captures"]) > 0,
        "tag_event": lambda r: float(r["metrics"]["tags_received"]) > 0,
        "failure": lambda r: float(r["metrics"]["captures"]) == 0,
        "shield_intervention": lambda r: float(r["metrics"]["shield_interventions"]) > 0,
        "shield_deadlock": lambda r: float(r["metrics"]["shield_deadlocks"]) > 0,
    }
    for label, predicate in predicates.items():
        record = next((r for r in records if predicate(r)), None)
        destination = output / f"{label}.json"
        if record and not destination.exists():
            destination.write_text(json.dumps(record, indent=2), encoding="utf-8")


def _aggregate(stage_dir, stage, aliases=None):
    aliases=aliases or {}
    entries=[]
    for method in method_definitions():
        for seed in SEEDS:
            run_name=aliases.get(f"{method}/seed{seed}", f"seed{seed}")
            path = stage_dir / method / run_name / "evaluation.csv"
            if not path.exists():
                continue
            rows=list(csv.DictReader(path.open(encoding="utf-8")))
            capture=[int(float(r["captures"]) > 0) for r in rows]
            def bool_rate(name): return float(np.mean([str(r[name]).lower()=="true" for r in rows]))
            def present_rate(name): return float(np.mean([str(r[name]).strip() not in {"", "None", "null"} for r in rows]))
            entries.append({"method":method,"seed":seed,"episodes":len(rows),"captures":sum(capture),
                "capture_rate":float(np.mean(capture)),"pickup_rate":present_rate("pickup_time"),
                "return_midfield_rate":bool_rate("return_crossed_midfield"),
                "enemy_territory_rate":bool_rate("reached_enemy_territory"),
                "tag_rate":float(np.mean([float(r["tags_received"]) for r in rows])),
                "violation_rate":float(np.mean([float(r["rule_violation_steps"]) > 0 for r in rows])),
                "ltlf_satisfaction_rate":bool_rate("ltlf_satisfied"),
                "common_task_reward_mean":float(np.mean([float(r["common_task_reward"]) for r in rows]))})
    with (stage_dir / "per_seed_summary.csv").open("x", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=list(entries[0]) if entries else ["method"]); writer.writeheader(); writer.writerows(entries)
    summary=[]
    for method in method_definitions():
        group=[e for e in entries if e["method"]==method]
        if not group: continue
        pooled=sum(e["captures"] for e in group)/sum(e["episodes"] for e in group)
        promoted = sum(e["capture_rate"] >= .10 for e in group) >= 3 and pooled >= .10
        summary.append({"method":method,"seeds_completed":len(group),"evaluation_episodes":sum(e["episodes"] for e in group),
            "total_captures":sum(e["captures"] for e in group),"pooled_capture_rate":pooled,
            "mean_seed_capture_rate":statistics.mean(e["capture_rate"] for e in group),
            "std_seed_capture_rate":statistics.stdev(e["capture_rate"] for e in group) if len(group)>1 else 0.,
            "promotes":promoted})
    save_json(stage_dir / "stage_summary.json", {"stage":stage,"per_seed":entries,"methods":summary,
        "promotion_rule":PROMOTION_RULE})
    return summary


def run_stage(root, stage):
    root = Path(root)
    if not (root / "preflight.json").exists():
        raise RuntimeError("Run and pass preflight before a confirmatory stage")
    if stage not in STAGES: raise ValueError(f"Unknown stage: {stage}")
    allowed=_allowed_methods(root, stage)
    stage_dir=root / "stages" / stage
    if stage_dir.exists(): raise FileExistsError(f"Stage directory already exists: {stage_dir}")
    stage_dir.mkdir(parents=True)
    # Freeze the exact runtime package for this batch.  This is separate from the
    # study-level snapshot because documented technical fixes may precede a stage.
    runtime_snapshot=stage_dir / "source_snapshot"
    shutil.copytree(Path(__file__).resolve().parent, runtime_snapshot,
                    ignore=shutil.ignore_patterns("__pycache__"))
    stage_config=STAGES[stage]
    save_json(stage_dir / "stage_plan.json", {"created_at":_now(),"stage":stage,"config":stage_config,
        "methods":allowed,"seeds":SEEDS,"steps_per_run":STEPS,"episodes_per_seed":EVALUATION_EPISODES,
        "source_hashes":_hash_tree(runtime_snapshot),"promotion_rule":PROMOTION_RULE})
    events=stage_dir / "execution.jsonl"
    for method in allowed:
        definition=method_definitions()[method]
        for seed in SEEDS:
            run=stage_dir / method / f"seed{seed}"
            events.parent.mkdir(parents=True,exist_ok=True)
            with events.open("a",encoding="utf-8") as f: f.write(json.dumps({"at":_now(),"event":"started","method":method,"seed":seed})+"\n")
            try:
                overrides={k:v for k,v in definition.items() if k not in {"shield","reward_shaping","temporal_state"}}
                policy=train_team(root / "specification.json",run,team_size=stage_config["team_size"],
                    opponent=stage_config["opponent"],scenario=stage_config["scenario"],shield=False,
                    steps=STEPS,seed=seed,reward_shaping=definition["reward_shaping"],
                    temporal_state=definition["temporal_state"],config_overrides=overrides)
                trace=run / "trajectories.jsonl"
                evaluate_team(policy,run / "evaluation.csv",episodes=EVALUATION_EPISODES,
                    seed=50000+1000*int(stage[:2])+100*seed,opponents=(stage_config["opponent"],),
                    scenarios=(stage_config["scenario"],),trace_output=trace,trace_stride=1)
                _write_representatives(trace,run / "evaluation.csv",run / "representative_replays")
                with events.open("a",encoding="utf-8") as f: f.write(json.dumps({"at":_now(),"event":"completed","method":method,"seed":seed})+"\n")
            except Exception as exc:
                save_json(run / "technical_failure.json", {"at":_now(),"method":method,"seed":seed,"error":repr(exc)})
                with events.open("a",encoding="utf-8") as f: f.write(json.dumps({"at":_now(),"event":"failed","method":method,"seed":seed,"error":repr(exc)})+"\n")
                raise
    summary=_aggregate(stage_dir,stage)
    decision={"created_at":_now(),"stage":stage,"promotion_rule":PROMOTION_RULE,
        "promoted_methods":[x["method"] for x in summary if x["promotes"]],"summary":summary,
        "review_note":"This deterministic rule is pre-registered; inspect raw metrics before manually starting another stage."}
    save_json(stage_dir / "promotion.json",decision)
    (stage_dir / "REPORT.md").write_text("# Phase 3 confirmatory stage\n\n"
        "This stage is confirmatory only after all five planned seeds completed. See `stage_summary.json`, `per_seed_summary.csv`, raw evaluation CSV files, and replay-compatible trajectories.\n",
        encoding="utf-8")
    return decision


def resume_stage_after_technical_fix(root, stage):
    """Resume only missing work after a documented orchestration failure.

    Completed policies/evaluations are never retrained or overwritten.
    """
    root=Path(root); stage_dir=root / "stages" / stage
    if not stage_dir.exists(): raise FileNotFoundError(stage_dir)
    plan=json.loads((stage_dir / "stage_plan.json").read_text(encoding="utf-8"))
    if stage != "02-1v1-patrol":
        raise RuntimeError("This targeted recovery is restricted to the documented Stage 2 replay post-processing failure")
    deviation=stage_dir / "technical_deviation.json"
    if not deviation.exists():
        save_json(deviation,{"at":_now(),"type":"postprocessing_fix","scope":"Representative replay selection only",
            "effect_on_completed_policy_and_evaluation":"none","failure":"KeyError('metrics') because evaluate_team stores metrics separately from trajectories",
            "action":"Join evaluation.csv rows to trajectory records by evaluation_seed; do not retrain completed seed0."})
    method_names=plan["methods"]
    events=stage_dir / "execution.jsonl"
    for method in method_names:
        definition=method_definitions()[method]
        for seed in SEEDS:
            run=stage_dir / method / f"seed{seed}"
            policy=run / "policy.zip"; evaluation=run / "evaluation.csv"; trace=run / "trajectories.jsonl"
            if evaluation.exists() and trace.exists():
                _write_representatives(trace,evaluation,run / "representative_replays")
                with events.open("a",encoding="utf-8") as f: f.write(json.dumps({"at":_now(),"event":"recovered_postprocessing","method":method,"seed":seed})+"\n")
                continue
            if policy.exists():
                evaluate_team(policy,evaluation,episodes=EVALUATION_EPISODES,
                    seed=50000+1000*int(stage[:2])+100*seed,opponents=(STAGES[stage]["opponent"],),
                    scenarios=(STAGES[stage]["scenario"],),trace_output=trace,trace_stride=1)
                _write_representatives(trace,evaluation,run / "representative_replays")
                with events.open("a",encoding="utf-8") as f: f.write(json.dumps({"at":_now(),"event":"recovered_evaluation","method":method,"seed":seed})+"\n")
                continue
            if run.exists():
                raise RuntimeError(f"Incomplete training run requires explicit review, not automatic rerun: {run}")
            with events.open("a",encoding="utf-8") as f: f.write(json.dumps({"at":_now(),"event":"started_after_technical_fix","method":method,"seed":seed})+"\n")
            overrides={k:v for k,v in definition.items() if k not in {"shield","reward_shaping","temporal_state"}}
            policy=train_team(root / "specification.json",run,team_size=STAGES[stage]["team_size"],
                opponent=STAGES[stage]["opponent"],scenario=STAGES[stage]["scenario"],shield=False,
                steps=STEPS,seed=seed,reward_shaping=definition["reward_shaping"],
                temporal_state=definition["temporal_state"],config_overrides=overrides)
            evaluate_team(policy,evaluation,episodes=EVALUATION_EPISODES,
                seed=50000+1000*int(stage[:2])+100*seed,opponents=(STAGES[stage]["opponent"],),
                scenarios=(STAGES[stage]["scenario"],),trace_output=trace,trace_stride=1)
            _write_representatives(trace,evaluation,run / "representative_replays")
            with events.open("a",encoding="utf-8") as f: f.write(json.dumps({"at":_now(),"event":"completed","method":method,"seed":seed})+"\n")
    summary=_aggregate(stage_dir,stage)
    save_json(stage_dir / "promotion.json",{"created_at":_now(),"stage":stage,"promotion_rule":PROMOTION_RULE,
        "promoted_methods":[x["method"] for x in summary if x["promotes"]],"summary":summary,
        "review_note":"Post-processing recovery did not retrain or overwrite completed policies/evaluations."})
    return summary


def continue_stage02_after_numerical_fix(root):
    """Continue Stage 2 after the documented PyQuaticus NaN episode failure.

    The failed original directory remains immutable. The rerun has the same logical
    seed and only differs by a guard that ends a corrupted simulator episode.
    """
    root=Path(root); stage="02-1v1-patrol"; stage_dir=root / "stages" / stage
    plan=json.loads((stage_dir / "stage_plan.json").read_text(encoding="utf-8"))
    registry_path=stage_dir / "run_registry.json"
    registry=json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    old=stage_dir / "softened_guidance" / "seed0"
    if not old.exists(): raise RuntimeError("Expected documented failed softened_guidance/seed0 run")
    failure=old / "technical_failure.json"
    if not failure.exists():
        save_json(failure,{"at":_now(),"type":"simulator_numerical_failure",
            "training_progress":48768,"error":"PyQuaticus invalid geometry propagated NaN logits to PPO",
            "rerun_policy":"one documented technical rerun under seed0_technical_rerun1; original directory preserved."})
    logical="softened_guidance/seed0"
    registry.setdefault(logical,"seed0_technical_rerun1")
    save_json(registry_path,registry)
    deviation=stage_dir / "technical_deviation_numerical_guard.json"
    if not deviation.exists():
        save_json(deviation,{"at":_now(),"scope":"TeamArena numerical fail-closed guard",
            "effect":"Only non-finite PyQuaticus game states now terminate their episode and retain the last valid observation.",
            "effect_on_prior_completed_runs":"none; completed neural and temporal runs are preserved.",
            "rerun":"softened_guidance logical seed 0 only, same training/evaluation seed, new immutable retry directory."})
    events=stage_dir / "execution.jsonl"
    for method in plan["methods"]:
        definition=method_definitions()[method]
        for seed in SEEDS:
            logical=f"{method}/seed{seed}"
            run_name=registry.get(logical,f"seed{seed}")
            run=stage_dir / method / run_name
            policy=run / "policy.zip"; evaluation=run / "evaluation.csv"; trace=run / "trajectories.jsonl"
            if evaluation.exists() and trace.exists():
                _write_representatives(trace,evaluation,run / "representative_replays")
                continue
            if run.exists() and not policy.exists():
                raise RuntimeError(f"Incomplete run needs review before continuation: {run}")
            if not run.exists():
                with events.open("a",encoding="utf-8") as f: f.write(json.dumps({"at":_now(),"event":"started_after_numerical_fix","method":method,"seed":seed,"run":run_name})+"\n")
                overrides={k:v for k,v in definition.items() if k not in {"shield","reward_shaping","temporal_state"}}
                policy=train_team(root / "specification.json",run,team_size=1,opponent="patrol",scenario="standard",
                    shield=False,steps=STEPS,seed=seed,reward_shaping=definition["reward_shaping"],
                    temporal_state=definition["temporal_state"],config_overrides=overrides)
            evaluate_team(policy,evaluation,episodes=EVALUATION_EPISODES,seed=50000+1000*2+100*seed,
                opponents=("patrol",),scenarios=("standard",),trace_output=trace,trace_stride=1)
            _write_representatives(trace,evaluation,run / "representative_replays")
            with events.open("a",encoding="utf-8") as f: f.write(json.dumps({"at":_now(),"event":"completed_after_numerical_fix","method":method,"seed":seed,"run":run_name})+"\n")
    summary=_aggregate(stage_dir,stage,registry)
    save_json(stage_dir / "promotion.json",{"created_at":_now(),"stage":stage,"promotion_rule":PROMOTION_RULE,
        "promoted_methods":[x["method"] for x in summary if x["promotes"]],"summary":summary,
        "run_registry":registry,"review_note":"One documented simulator numerical rerun; all original artifacts retained."})
    return summary


def resume_stage02_after_interruption(root):
    """Resume after an externally interrupted no-tag-penalty seed.

    The partially trained original is retained and is never treated as a result.
    """
    root=Path(root); stage="02-1v1-patrol"; stage_dir=root / "stages" / stage
    registry_path=stage_dir / "run_registry.json"
    registry=json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    logical="softened_guidance_no_tag_penalty/seed2"
    original=stage_dir / "softened_guidance_no_tag_penalty" / "seed2"
    if not original.exists() or (original / "policy.zip").exists():
        raise RuntimeError("Expected the documented incomplete no-tag-penalty seed2 run")
    failure=original / "technical_failure.json"
    if not failure.exists():
        last=0
        metrics=original / "training_episode_metrics.jsonl"
        if metrics.exists():
            last=json.loads(metrics.read_text(encoding="utf-8").splitlines()[-1])["timesteps"]
        save_json(failure,{"at":_now(),"type":"external_process_interruption","training_progress":last,
            "evidence":"No active Stage 2 Python process and no application traceback after the final metric log.",
            "rerun_policy":"one documented exact-seed rerun under seed2_interrupted_rerun1; original directory preserved."})
    registry.setdefault(logical,"seed2_interrupted_rerun1")
    save_json(registry_path,registry)
    deviation=stage_dir / "technical_deviation_interruption.json"
    if not deviation.exists():
        save_json(deviation,{"at":_now(),"scope":"orchestration only","effect_on_environment_and_policy_configuration":"none",
            "effect":"One incomplete logical seed is rerun with the same seed and evaluation seeds in a separate immutable directory."})
    # Reuse the documented numerical-guard continuation, which respects run_registry.
    return continue_stage02_after_numerical_fix(root)
