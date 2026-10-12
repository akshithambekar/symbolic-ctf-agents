"""Short scripted per-player tactic screen; never claims learned coordination."""
import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maritime_nsrl.research_env import ArenaConfig
from maritime_nsrl.runtime_shield_study import csv_rows, save
from maritime_nsrl.tactics import TacticalArena, TacticalPolicy, TacticsConfig, role_contract
from maritime_nsrl.viewer import snapshot


def run(output, episodes=3):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    snapshot_dir = output / "source_snapshot"
    shutil.copytree(ROOT / "src", snapshot_dir / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(__file__, snapshot_dir / "evaluate_tactics.py")
    save(output / "source_hashes.json", {p.relative_to(snapshot_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in snapshot_dir.rglob("*") if p.is_file()})
    save(output / "protocol.json", {"type": "exploratory scripted tactical validation", "team_size": 3,
        "modes": ["none", "fixed", "conditional", "dynamic"], "opponents": ["static", "patrol", "chase", "raid"],
        "episodes_per_cell": episodes, "evaluation_seeds": list(range(880000, 880000+episodes)),
        "action_abstraction": "role-specific waypoint targets; distinct from the 1v1 learned attacker",
        "hard_shield": False, "role_constraints": "monitored, not formally enforced", "learned_coordination": False})
    rows = []
    for mode in ("none", "fixed", "conditional", "dynamic"):
        artifact = role_contract(mode)
        for opponent in ("static", "patrol", "chase", "raid"):
            tactics = TacticsConfig(defender_mode=mode, attack_style="escort" if mode == "dynamic" else "split_flank",
                red_behavior="raid" if opponent == "raid" else "standard")
            config = ArenaConfig(team_size=3, guided=False, reward_shaping=False, temporal_state=False,
                shield=False, formula=artifact["ltl"], opponent="static" if opponent == "raid" else opponent)
            cell = output / f"{mode}-{opponent}"
            cell.mkdir()
            save(cell / "profile.json", {"arena": asdict(config), "tactics": asdict(tactics), "specification": artifact})
            arena, policy = TacticalArena(config, tactics), TacticalPolicy()
            try:
                for episode in range(episodes):
                    arena.reset(880000 + episode)
                    previous_roles = arena.roles()
                    counters = dict(intrusion_decisions=0, covered_intrusion_decisions=0, role_violation_decisions=0,
                        role_switches=0, defender_assignments=[0, 0, 0])
                    with (cell / f"episode-{episode}.jsonl").open("w", encoding="utf-8") as trace:
                        while not arena.finished:
                            roles = arena.roles()
                            labels = arena.labels()
                            intrusion = "opponent_push" in labels
                            covered = "defender_behind_center" in labels
                            counters["intrusion_decisions"] += int(intrusion)
                            counters["covered_intrusion_decisions"] += int(intrusion and covered)
                            counters["role_violation_decisions"] += int((mode == "fixed" or mode in {"conditional", "dynamic"} and intrusion) and not covered)
                            counters["role_switches"] += sum(a != b for a, b in zip(roles, previous_roles))
                            for i, role in enumerate(roles): counters["defender_assignments"][i] += int(role == "defender")
                            previous_roles = roles
                            actions = policy(arena)
                            before_caps = int(arena.game.state["captures"][0])
                            _, _, _, info = arena.step(actions)
                            frame = snapshot(arena, actions, episode, True, before_caps)
                            frame["roles"] = arena.roles(); frame["roles_before_action"] = roles
                            trace.write(json.dumps(frame) + "\n")
                    metrics = info["metrics"]; metrics.pop("trajectory", None)
                    rows.append({"mode": mode, "tactical_opponent": opponent, "evaluation_seed": 880000+episode, **metrics, **counters})
            finally: arena.close()
            print(f"tactics {mode}/{opponent}: {episodes} episodes", flush=True)
    csv_rows(output / "episodes.csv", rows)
    save(output / "episodes.json", rows)
    summaries = []
    for mode in ("none", "fixed", "conditional", "dynamic"):
        for opponent in ("static", "patrol", "chase", "raid"):
            group = [r for r in rows if r["mode"] == mode and r["tactical_opponent"] == opponent]
            intrusion = sum(r["intrusion_decisions"] for r in group)
            summaries.append({"mode": mode, "opponent": opponent, "episodes": len(group),
                "captures": sum(r["captures"] > 0 for r in group), "pickups": sum(r["pickups"] > 0 for r in group),
                "red_captures": sum(r["opponent_captures"] for r in group), "mean_tags": float(np.mean([r["tags_received"] for r in group])),
                "role_violation_decisions": sum(r["role_violation_decisions"] for r in group),
                "intrusion_decisions": intrusion,
                "intrusion_coverage": sum(r["covered_intrusion_decisions"] for r in group)/intrusion if intrusion else None,
                "contract_satisfied_episodes": sum(r["ltlf_satisfied"] for r in group)})
    csv_rows(output / "summary.csv", summaries); save(output / "summary.json", summaries)
    lines = ["# Scripted per-player tactics: exploratory validation", "", "This is an explicit heuristic library, not trained MARL coordination. Role-specific waypoint semantics are separate from the unchanged eight-action 1v1 PPO attacker. Conditional contracts react to grounded opponent intrusion; dynamic assignments exclude flag carriers. Roles and constraints are recorded, not hard-shielded.", "",
        "| Roles | Opponent | Captures | Pickups | Red captures | Role violations (decisions) | Intrusion coverage |", "|---|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        coverage = "not exposed" if s["intrusion_coverage"] is None else f"{100*s['intrusion_coverage']:.1f}%"
        lines.append(f"| {s['mode']} | {s['opponent']} | {s['captures']}/{s['episodes']} | {s['pickups']}/{s['episodes']} | {s['red_captures']} | {s['role_violation_decisions']} | {coverage} |")
    lines += ["", "Three episodes per cell are feasibility/debugging probes, not a ranking or superiority claim. Static/patrol/chase defenders do not ordinarily invade blue territory; the separately labeled raiding policy tests the defensive trigger. The current conditional contract is instantaneous: a defender outside the home half when intrusion begins violates it, even if the controller later returns. Dynamic assignment is a heuristic and does not guarantee that any agent can return quickly enough."]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summaries


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()
    if args.episodes < 1: p.error("episodes must be positive")
    run(args.output, args.episodes)
