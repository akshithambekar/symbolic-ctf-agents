"""Exploratory paired evaluation of unchanged neural policies with runtime shielding."""
import csv
import hashlib
import importlib.metadata
import json
import random
import shutil
import subprocess
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import pyquaticus
import torch
from stable_baselines3 import PPO

from .research_env import ArenaConfig, SCENARIOS, TeamArena
from .shield import PredictiveShield
from .specifications import read_artifact


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def baseline_config(values, scenario, opponent, shield):
    config = ArenaConfig(**values)
    if config.guided or config.reward_shaping or config.temporal_state or config.shield:
        raise ValueError("Runtime-only study requires an unguided, unshielded training configuration")
    config.scenario, config.opponent, config.shield = scenario, opponent, shield
    config.shield_formula = "G(in_bounds)"
    config.trace_stride = 0
    return config


class AuditedShield(PredictiveShield):
    """Record existing gate decisions without additional simulator predictions."""
    def __init__(self, arena):
        super().__init__("G(in_bounds)")
        self.arena = arena
        self.checks = 0
        self.events = []
        self.waypoints = []
        self.initialize(arena.labels())

    def transition(self, game, candidates, ground):
        choices = list(candidates)
        positions = game.state["agent_position"][:self.arena.n]
        carrying = bool(any(game.state["agent_has_flag"][:self.arena.n]))
        enemy = bool(any(positions[:, 0] > self.arena.size[0] / 2))
        distance = float(min(np.linalg.norm(p - game.state["flag_home"][1]) for p in positions))
        before_time = float(game.current_time)
        self.checks += 1
        successor, result, index = super().transition(game, choices, ground)
        if index is None or index > 0:
            self.events.append({"check": self.checks, "time": before_time,
                "pre_positions": positions.tolist(), "enemy_territory": enemy,
                "carrying_flag": carrying, "enemy_flag_distance": distance,
                "near_enemy_flag": distance <= 2 * game.catch_radius,
                "proposed_waypoints": self.waypoints,
                "proposed_controls": {k: np.asarray(v).tolist() for k, v in choices[0].items()},
                "substituted_controls": None if index is None else {
                    k: np.asarray(v).tolist() for k, v in choices[index].items()},
                "candidate_index": index, "deadlock": index is None,
                "reason": "no tested live successor" if index is None else "proposed successor rejected"})
        return successor, result, index


def summarize(rows):
    captures = [r["capture_time"] for r in rows if r["captures"] > 0 and r["capture_time"] is not None]
    distances = [r["enemy_flag_distance_min"] for r in rows if r["enemy_flag_distance_min"] is not None]
    checks = sum(r["shield_checks"] for r in rows)
    return {"episodes": len(rows), "captures": sum(r["captures"] > 0 for r in rows),
        "capture_rate": float(np.mean([r["captures"] > 0 for r in rows])),
        "pickup_rate": float(np.mean([r["pickups"] > 0 for r in rows])),
        "enemy_territory_rate": float(np.mean([r["reached_enemy_territory"] for r in rows])),
        "flag_vicinity_rate": float(np.mean([r["reached_enemy_flag"] for r in rows])),
        "mean_min_flag_distance": float(np.mean(distances)) if distances else None,
        "mean_common_reward": float(np.mean([r["common_task_reward"] for r in rows])),
        "tag_rate": float(np.mean([r["tags_received"] > 0 for r in rows])),
        "mean_tags": float(np.mean([r["tags_received"] for r in rows])),
        "out_of_bounds_episode_rate": float(np.mean([r["out_of_bounds_steps"] > 0 for r in rows])),
        "mean_out_of_bounds_steps": float(np.mean([r["out_of_bounds_steps"] for r in rows])),
        "mean_rule_violation_steps": float(np.mean([r["rule_violation_steps"] for r in rows])),
        "ltlf_satisfaction_rate": float(np.mean([r["ltlf_satisfied"] for r in rows])),
        "mean_capture_time_successes": float(np.mean(captures)) if captures else None,
        "mean_duration": float(np.mean([r["duration"] for r in rows])),
        "mean_evaluation_wall_seconds": float(np.mean([r["evaluation_wall_seconds"] for r in rows])),
        "deadlock_rate": float(np.mean([r["shield_deadlocks"] > 0 for r in rows])),
        "numerical_termination_rate": float(np.mean([r["numerical_terminations"] > 0 for r in rows])),
        "interventions": sum(r["shield_interventions"] for r in rows), "checks": checks,
        "intervention_frequency": sum(r["shield_interventions"] for r in rows) / checks if checks else 0.,
        "offensive_interventions": sum(r["audit_offensive_interventions"] for r in rows),
        "near_flag_interventions": sum(r["audit_near_flag_interventions"] for r in rows),
        "carrying_interventions": sum(r["audit_carrying_interventions"] for r in rows)}


def csv_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(root, output, episodes=10, training_seeds=(0, 1), evaluation_seed=710000, policy_root=None,
        scenarios=tuple(SCENARIOS), opponents=("static", "patrol", "chase"),
        start_modes=("standard",), deterministic=True, boundary_margin=2.5, boundary_speed=0.):
    if episodes < 1 or not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("Positive episodes and distinct training seeds required")
    root, output = Path(root), Path(output)
    source = Path(policy_root) if policy_root is not None else root / "research_archive/phase3-confirmatory-20260927-v1/stages/02-1v1-patrol/neural"
    if not scenarios or not opponents or not start_modes: raise ValueError("Nonempty evaluation cells required")
    inputs = []
    for seed in training_seeds:
        directory = source / f"seed{seed}"
        values = json.loads((directory / "arena_config.json").read_text(encoding="utf-8"))
        if policy_root is not None and json.loads((directory / "status.json").read_text(encoding="utf-8"))["status"] != "promoted":
            raise ValueError("Custom attacker checkpoint has not passed its capability screen")
        baseline_config(values, "standard", "static", False)
        spec = read_artifact(directory / "specification.json")
        if spec["ltl"] != values["formula"]:
            raise ValueError("Saved mission/config provenance mismatch")
        inputs.append((seed, directory, values))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    sources = list((root / "src").rglob("*.py")) + [root / "scripts/run_runtime_shield_study.py"]
    hashes = {}
    for path in sources:
        relative = path.relative_to(root)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative).replace("\\", "/")] = digest(path)
    simulator = Path(pyquaticus.__file__).parent
    save(output / "source_hashes.json", hashes)
    save(output / "simulator_hashes.json", {str(p.relative_to(simulator)): digest(p) for p in simulator.rglob("*.py")})
    save(output / "protocol.json", {"status": "exploratory", "design": "paired saved neural policies; evaluation-only shield",
        "training_seeds": list(training_seeds), "evaluation_seeds": list(range(evaluation_seed, evaluation_seed + episodes)),
        "scenarios": list(scenarios), "opponents": list(opponents), "start_modes": list(start_modes),
        "config_layout": "with-start-mode", "policy_source": str(source),
        "boundary_start_margin": boundary_margin,
        "boundary_start_speed": boundary_speed,
        "arms": ["unshielded", "runtime_bounds_shield"], "deterministic": deterministic,
        "shield_formula": "G(in_bounds)", "retraining": False,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "torch", "stable-baselines3", "gymnasium", "lark")},
        "simulator_path": str(simulator), "source_root": str(root),
        "limitations": ["Two training seeds, no confirmatory significance claim", "Saved neural checkpoints; inspect inputs for training method",
            "One-step local bounds gate, not full LTL mission enforcement", "Deadlocks shorten exposure and can inflate common reward"]})
    rows = []
    started = time.perf_counter()
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as raw, (output / "traces.jsonl").open("w", encoding="utf-8") as traces:
        for seed, directory, values in inputs:
            archived_input = output / "inputs" / f"seed{seed}"
            archived_input.mkdir(parents=True)
            for name in ("config.json", "arena_config.json", "specification.json", "policy.zip", "training_complete.json", "ppo_config.json"):
                if (directory / name).exists(): shutil.copy2(directory / name, archived_input / name)
            save(archived_input / "hashes.json", {p.name: digest(p) for p in archived_input.iterdir() if p.is_file()})
            model = PPO.load(archived_input / "policy.zip", device="cpu")
            for scenario, opponent, start_mode in product(scenarios, opponents, start_modes):
                for enabled in (False, True):
                    arm = "runtime_bounds_shield" if enabled else "unshielded"
                    config = baseline_config(values, scenario, opponent, enabled)
                    config.start_mode = start_mode
                    config.boundary_start_margin = boundary_margin
                    config.boundary_start_speed = boundary_speed
                    config_dir = output / "configs" / f"seed{seed}"
                    config_dir.mkdir(parents=True, exist_ok=True)
                    save(config_dir / f"{scenario}-{opponent}-{start_mode}-{arm}.json", asdict(config))
                    arena = TeamArena(config)
                    try:
                        for ep in range(episodes):
                            episode_started = time.perf_counter()
                            paired_seed = evaluation_seed + ep
                            random.seed(paired_seed); np.random.seed(paired_seed); torch.manual_seed(paired_seed)
                            observation = arena.reset(seed=paired_seed)
                            if enabled: arena.gate = AuditedShield(arena)
                            trajectory = []
                            done = False
                            while not done:
                                actions, _ = model.predict(observation, deterministic=deterministic)
                                if enabled: arena.gate.waypoints = np.asarray(actions).astype(int).tolist()
                                observation, _, done, info = arena.step(actions)
                                trajectory.append({"time": float(arena.game.current_time),
                                    "positions": arena.game.state["agent_position"].tolist(),
                                    "has_flag": list(map(bool, arena.game.state["agent_has_flag"])),
                                    "tagged": list(map(bool, arena.game.state["agent_is_tagged"])),
                                    "proposed_waypoints": np.asarray(actions).astype(int).tolist(),
                                    "temporal_state": arena.monitor.state,
                                    "interventions": arena.metrics["shield_interventions"],
                                    "deadlocks": arena.metrics["shield_deadlocks"]})
                            metrics = info["metrics"]
                            metrics.pop("trajectory", None)
                            events = arena.gate.events if enabled else []
                            interventions = [e for e in events if not e["deadlock"]]
                            row = {"training_seed": seed, "evaluation_seed": paired_seed,
                                "arm": arm, "start_mode": start_mode, **metrics, "evaluation_wall_seconds": time.perf_counter() - episode_started,
                                "shield_checks": arena.gate.checks if enabled else 0,
                                "audit_offensive_interventions": sum(e["enemy_territory"] or e["carrying_flag"] for e in interventions),
                                "audit_near_flag_interventions": sum(e["near_enemy_flag"] for e in interventions),
                                "audit_carrying_interventions": sum(e["carrying_flag"] for e in interventions)}
                            rows.append(row)
                            raw.write(json.dumps(row, allow_nan=False) + "\n"); raw.flush()
                            traces.write(json.dumps({"training_seed": seed, "evaluation_seed": paired_seed,
                                "scenario": scenario, "opponent": opponent, "start_mode": start_mode, "arm": arm,
                                "trajectory": trajectory, "shield_events": events}, allow_nan=False) + "\n"); traces.flush()
                    finally: arena.close()
                    print(f"seed={seed} {scenario}/{opponent}/{start_mode}/{arm}: {episodes} episodes", flush=True)
    csv_rows(output / "episodes.csv", rows)
    groups = []
    for seed, scenario, opponent, start_mode, arm in product(training_seeds, scenarios, opponents, start_modes, ("unshielded", "runtime_bounds_shield")):
        group = [r for r in rows if (r["training_seed"], r["scenario"], r["opponent"], r["start_mode"], r["arm"]) == (seed, scenario, opponent, start_mode, arm)]
        groups.append({"training_seed": seed, "scenario": scenario, "opponent": opponent, "start_mode": start_mode, "arm": arm, **summarize(group)})
    csv_rows(output / "per_seed_summary.csv", groups)
    paired = []
    for off in rows:
        if off["arm"] != "unshielded": continue
        on = next(r for r in rows if r["arm"] != "unshielded" and all(r[k] == off[k] for k in ("training_seed", "scenario", "opponent", "start_mode", "evaluation_seed")))
        paired.append({k: off[k] for k in ("training_seed", "scenario", "opponent", "start_mode", "evaluation_seed") } | {
            f"delta_{k}": on[k] - off[k] for k in ("captures", "pickups", "common_task_reward", "tags_received", "out_of_bounds_steps", "duration")})
    csv_rows(output / "paired_deltas.csv", paired)
    totals = {arm: summarize([r for r in rows if r["arm"] == arm]) for arm in ("unshielded", "runtime_bounds_shield")}
    save(output / "summary.json", {"status": "completed exploratory", "elapsed_seconds": time.perf_counter() - started, "arms": totals})
    report(output, groups, totals)
    return totals


def report(output, groups, totals):
    lines = ["# Runtime-only bounds shielding: exploratory paired study", "",
        "The same saved unshielded neural checkpoints were evaluated with and without the existing runtime bounds gate. No policy retraining, temporal observation changes, or reward shaping was introduced. Each arm used the same held-out seeds, maps, opponents, simulator, and horizon.", "",
        "| Arm | Episodes | Captures | Pickups (%) | Tags (%) | OOB (%) | Common reward | Duration | Deadlocks (%) | Interventions/checks |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, s in totals.items():
        lines.append(f"| {arm} | {s['episodes']} | {s['captures']} | {100*s['pickup_rate']:.2f} | {100*s['tag_rate']:.2f} | {100*s['out_of_bounds_episode_rate']:.2f} | {s['mean_common_reward']:.3f} | {s['mean_duration']:.2f} | {100*s['deadlock_rate']:.2f} | {s['interventions']}/{s['checks']} |")
    lines += ["", "## Interpretation limits", "", "This is a two-training-seed exploratory study, not a superiority test. Evaluation episodes are paired within checkpoints; they are not independent training replicates. The gate enforces predicted bounds safety only. Tags and mission satisfaction remain measured outcomes, not guaranteed properties. A deadlock ends an episode early, reducing exposure to both tags and time penalties; common reward must be interpreted alongside duration and deadlock rate. Rejecting an action near the flag is evidence of intervention context, not proof that a feasible capture was prevented.", "", "See per_seed_summary.csv, paired_deltas.csv, episodes.jsonl, traces.jsonl, inputs/, configs/, and source hashes for the full record."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, label in zip(axes, ("capture_rate", "out_of_bounds_episode_rate", "deadlock_rate"), ("Capture rate", "Out-of-bounds episode rate", "Deadlock rate")):
        for offset, arm in enumerate(totals):
            for seed in sorted({g["training_seed"] for g in groups}):
                values = [g[key] for g in groups if g["training_seed"] == seed and g["arm"] == arm]
                ax.scatter(offset, np.mean(values), label=f"seed {seed}" if offset == 0 else None)
        ax.set_xticks([0, 1], ["Off", "Runtime shield"]); ax.set_ylabel(label); ax.set_ylim(-.02, 1.02)
    axes[0].legend(); fig.suptitle("Exploratory paired runtime shielding (each point = one training seed)")
    fig.tight_layout(); fig.savefig(output / "comparison.png", dpi=160); plt.close(fig)
