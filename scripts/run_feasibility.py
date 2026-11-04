"""Compare simple capture controllers, action intervals, and chase speeds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from maritime_nsrl.team_env import ArenaConfig, TeamArena


def choose_action(arena, policy):
    state = arena.game.state
    blue = state["agent_position"][0]
    red = state["agent_position"][1]
    if state["agent_has_flag"][0]:
        return 1
    if policy == "direct":
        return 0
    # The waypoint menu's lower flank gives the attacker a lateral approach.
    # Switch back toward the flag once the boat has cleared the defender's lane.
    if blue[0] < 0.78 * arena.size[0] and blue[1] < red[1] - 5:
        return 2
    if blue[0] < 0.78 * arena.size[0] and blue[1] > red[1] + 5:
        return 3
    return 0


def run_case(speed, repeat, policy, seed):
    config = ArenaConfig(
        formula="G(in_bounds) & F(has_flag & F(captured))",
        guided=False,
        reward_shaping=False,
        temporal_state=False,
        opponent="chase",
        opponent_speed=speed,
        repeat=repeat,
        trace_stride=0,
    )
    arena = TeamArena(config)
    try:
        arena.reset(seed)
        minimum_margin = float("inf")
        while not arena.finished:
            pos = arena.game.state["agent_position"]
            minimum_margin = min(minimum_margin, float(np.linalg.norm(pos[0] - pos[1])))
            _, _, _, info = arena.step([choose_action(arena, policy)])
        metrics = info["metrics"]
        return {
            "speed": speed,
            "repeat": repeat,
            "policy": policy,
            "seed": seed,
            "pickups": metrics["pickups"],
            "captures": metrics["captures"],
            "pickup_time": metrics["pickup_time"],
            "capture_time": metrics["capture_time"],
            "time_remaining": config.horizon - float(arena.game.current_time),
            "minimum_defender_distance": minimum_margin,
            "tags": metrics["tags_received"],
        }
    finally:
        arena.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--out", type=Path, default=Path("runs/feasibility-1v1"))
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--repeats", type=int, nargs="+", default=[5, 2, 1])
    args = parser.parse_args()
    if args.episodes < 1 or any(not 0 < speed <= 1 for speed in args.speeds):
        parser.error("episodes and speeds must be positive and speeds at most 1")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(args.speeds) * len(args.repeats) * 2 * args.episodes
    for speed in args.speeds:
        for repeat in args.repeats:
            for policy in ("direct", "flank"):
                for episode in range(args.episodes):
                    # Identical seed per paired policy/control condition.
                    seed = args.seed + episode
                    row = run_case(speed, repeat, policy, seed)
                    rows.append(row)
                    print(f"{len(rows)}/{total} {row}", flush=True)
    fields = list(rows[0])
    with (args.out / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for speed in args.speeds:
        for repeat in args.repeats:
            for policy in ("direct", "flank"):
                group = [r for r in rows if r["speed"] == speed and r["repeat"] == repeat and r["policy"] == policy]
                summary[f"speed={speed},repeat={repeat},policy={policy}"] = {
                    "episodes": len(group),
                    "pickups": sum(r["pickups"] for r in group),
                    "captures": sum(r["captures"] for r in group),
                    "tags": sum(r["tags"] for r in group),
                    "mean_minimum_defender_distance": float(np.mean([r["minimum_defender_distance"] for r in group])),
                }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
