"""Compare simple return actions from deterministic PPO pickup states."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from maritime_nsrl.team_env import ArenaConfig, TeamArena


RETURN_ACTIONS = {"home": 1, "lower_lane": 4, "upper_lane": 5, "evade": 6}


def pickup_snapshot(arena):
    state = arena.game.state
    return {
        "time": float(arena.game.current_time),
        "blue_position": state["agent_position"][0].astype(float).tolist(),
        "blue_heading": float(state["agent_heading"][0]),
        "blue_speed": float(state["agent_speed"][0]),
        "red_position": state["agent_position"][1].astype(float).tolist(),
        "red_heading": float(state["agent_heading"][1]),
        "red_speed": float(state["agent_speed"][1]),
        "tagged": bool(state["agent_is_tagged"][0]),
    }


def run_case(model, config, seed, return_name):
    arena = TeamArena(config)
    pickup = None
    try:
        observation = arena.reset(seed)
        while not arena.finished:
            if arena.game.state["agent_has_flag"][0]:
                action = RETURN_ACTIONS[return_name]
            else:
                predicted, _ = model.predict(observation, deterministic=True)
                action = int(np.asarray(predicted).reshape(-1)[0])
            observation, _, _, info = arena.step([action])
            if pickup is None and arena.metrics["pickups"]:
                pickup = pickup_snapshot(arena)
        metrics = info["metrics"]
        return {
            "seed": seed,
            "return_action": return_name,
            "pickups": metrics["pickups"],
            "captures": metrics["captures"],
            "pickup_time": metrics["pickup_time"],
            "capture_time": metrics["capture_time"],
            "tags_after_pickup": metrics["tags_after_pickup"],
            "return_crossed_midfield": metrics["return_crossed_midfield"],
            "pickup_snapshot": json.dumps(pickup, sort_keys=True) if pickup else None,
        }
    finally:
        arena.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("runs/approach-ppo-upper-return-1v1/policy.zip"))
    parser.add_argument("--arena-config", type=Path, default=Path("runs/approach-ppo-upper-return-1v1/arena_config.json"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--returns", nargs="+", choices=RETURN_ACTIONS, default=list(RETURN_ACTIONS))
    parser.add_argument("--out", type=Path, default=Path("runs/ppo-pickup-return-screen-1v1"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if not args.policy.is_file() or not args.arena_config.is_file():
        parser.error("policy and arena config files must exist")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    config = ArenaConfig(**json.loads(args.arena_config.read_text(encoding="utf-8")))
    config.repeat = 1
    model = PPO.load(args.policy, device="cpu")
    rows = []
    total = len(args.returns) * args.episodes
    for return_name in args.returns:
        for episode in range(args.episodes):
            seed = args.seed + episode
            row = run_case(model, config, seed, return_name)
            rows.append(row)
            print(f"{len(rows)}/{total} {row}", flush=True)

    fields = list(rows[0])
    with (args.out / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[row["return_action"]].append(row)
    summary = {
        "policy": str(args.policy),
        "episodes_per_return_action": args.episodes,
        "seed_start": args.seed,
        "repeat": config.repeat,
        "opponent_speed": config.opponent_speed,
        "results": {
            name: {
                "episodes": len(group),
                "pickup_episodes": sum(row["pickups"] > 0 for row in group),
                "capture_episodes": sum(row["captures"] > 0 for row in group),
                "captures_given_pickup_episode": (
                    sum(row["captures"] > 0 for row in group)
                    / sum(row["pickups"] > 0 for row in group)
                    if any(row["pickups"] > 0 for row in group)
                    else None
                ),
                "tags_after_pickup": sum(row["tags_after_pickup"] for row in group),
                "return_crossed_midfield": sum(row["return_crossed_midfield"] for row in group),
            }
            for name, group in sorted(groups.items())
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
