"""Compare return actions after the same flank approach picks up the flag."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from maritime_nsrl.team_env import ArenaConfig, TeamArena


RETURN_ACTIONS = {
    "home": 1,
    "lower_lane": 4,
    "upper_lane": 5,
    "evade": 6,
}


def choose_action(arena, return_action):
    state = arena.game.state
    blue = state["agent_position"][0]
    red = state["agent_position"][1]
    if state["agent_has_flag"][0]:
        return return_action
    if blue[0] < 0.78 * arena.size[0] and blue[1] < red[1] - 5:
        return 2
    if blue[0] < 0.78 * arena.size[0] and blue[1] > red[1] + 5:
        return 3
    return 0


def run_case(speed, return_name, seed):
    config = ArenaConfig(
        formula="G(in_bounds) & F(has_flag & F(captured))",
        guided=False,
        reward_shaping=False,
        temporal_state=False,
        opponent="chase",
        opponent_speed=speed,
        repeat=5,
        trace_stride=0,
    )
    arena = TeamArena(config)
    try:
        arena.reset(seed)
        while not arena.finished:
            _, _, _, info = arena.step([choose_action(arena, RETURN_ACTIONS[return_name])])
        metrics = info["metrics"]
        return {
            "speed": speed,
            "repeat": config.repeat,
            "return_action": return_name,
            "seed": seed,
            "pickups": metrics["pickups"],
            "captures": metrics["captures"],
            "pickup_time": metrics["pickup_time"],
            "capture_time": metrics["capture_time"],
            "tags_before_pickup": metrics["tags_before_pickup"],
            "tags_after_pickup": metrics["tags_after_pickup"],
            "time_remaining": config.horizon - float(arena.game.current_time),
            "return_crossed_midfield": metrics["return_crossed_midfield"],
            "minimum_defender_distance": metrics["defender_distance_min"],
        }
    finally:
        arena.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.6, 0.7])
    parser.add_argument("--returns", nargs="+", choices=RETURN_ACTIONS, default=list(RETURN_ACTIONS))
    parser.add_argument("--out", type=Path, default=Path("runs/carry-return-1v1"))
    args = parser.parse_args()
    if args.episodes < 1 or any(not 0 < speed <= 1 for speed in args.speeds):
        parser.error("episodes and speeds must be positive and speeds at most 1")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(args.speeds) * len(args.returns) * args.episodes
    for speed in args.speeds:
        for return_name in args.returns:
            for episode in range(args.episodes):
                row = run_case(speed, return_name, args.seed + episode)
                rows.append(row)
                print(f"{len(rows)}/{total} {row}", flush=True)

    fields = list(rows[0])
    with (args.out / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["speed"], row["return_action"])].append(row)
    summary = {
        f"speed={speed},return_action={return_name}": {
            "episodes": len(group),
            "pickups": sum(row["pickups"] for row in group),
            "captures": sum(row["captures"] for row in group),
            "captures_given_pickup": (
                sum(row["captures"] for row in group)
                / sum(row["pickups"] for row in group)
                if sum(row["pickups"] for row in group)
                else None
            ),
            "tags_after_pickup": sum(row["tags_after_pickup"] for row in group),
            "return_crossed_midfield": sum(
                row["return_crossed_midfield"] for row in group
            ),
        }
        for (speed, return_name), group in sorted(groups.items())
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
