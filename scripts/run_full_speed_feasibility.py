"""Screen action maneuvers and a bounded planner at competition speed."""

import argparse
import copy
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from maritime_nsrl.team_env import ArenaConfig, TeamArena


ACTION_NAMES = {
    0: "flag",
    1: "home",
    2: "lower_flank",
    3: "upper_flank",
    4: "lower_lane",
    5: "upper_lane",
    6: "evade",
    7: "hold",
}


def state_score(arena, previous_captures):
    state = arena.game.state
    if state["captures"][0] > previous_captures:
        return 1000.0
    if state["agent_is_tagged"][0]:
        return -1000.0
    blue = state["agent_position"][0]
    red = state["agent_position"][1]
    if state["agent_has_flag"][0]:
        progress = np.linalg.norm(blue - state["flag_home"][0])
    else:
        progress = np.linalg.norm(blue - state["flag_home"][1])
    defender_distance = np.linalg.norm(blue - red)
    # Favor task progress and maintain a modest separation from the defender.
    return -1.1 * float(progress) + 0.35 * float(defender_distance)


def simulate_candidate(arena, action, captures):
    np_state, py_state = np.random.get_state(), random.getstate()
    try:
        child = copy.deepcopy(arena)
        child.step([action])
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
    if child.metrics["tags_received"] or child.metrics["captures"] > captures:
        child.finished = True
    return child


def bounded_planner_action(arena, width=3):
    """Choose a first action by searching 8 actions for two decisions.

    It keeps only the best `width` first-step states, then checks all eight
    actions from each. Every candidate uses the local simulator transition.
    """
    first_states = []
    captures = int(arena.game.state["captures"][0])
    for action in ACTION_NAMES:
        child = simulate_candidate(arena, action, captures)
        score = state_score(child, captures)
        first_states.append((score, action, child))
    first_states.sort(key=lambda item: item[0], reverse=True)

    best_score, best_action = -float("inf"), first_states[0][1]
    for _, first_action, child in first_states[:width]:
        if child.finished:
            score = state_score(child, captures)
            if score > best_score:
                best_score, best_action = score, first_action
            continue
        for second_action in ACTION_NAMES:
            second = simulate_candidate(child, second_action, captures)
            score = state_score(second, captures)
            if score > best_score:
                best_score, best_action = score, first_action
    return best_action


def run_episode(speed, start_mode, policy, seed, planner_width):
    config = ArenaConfig(
        formula="G(in_bounds) & F(has_flag & F(captured))",
        guided=False,
        reward_shaping=False,
        temporal_state=False,
        opponent="chase",
        opponent_speed=speed,
        repeat=1,
        trace_stride=0,
        start_mode=start_mode,
        random_starts=True,
    )
    arena = TeamArena(config)
    try:
        arena.reset(seed)
        initial_has_flag = bool(arena.game.state["agent_has_flag"][0])
        stop_reason = "horizon"
        while not arena.finished:
            action = (
                bounded_planner_action(arena, planner_width)
                if policy == "bounded_planner"
                else int(policy)
            )
            _, _, _, info = arena.step([action])
            if arena.metrics["captures"]:
                stop_reason = "capture"
                break
            if arena.metrics["tags_received"]:
                stop_reason = "first_tag"
                break
        metrics = arena.metrics
        return {
            "speed": speed,
            "start_mode": start_mode,
            "policy": "bounded_planner" if policy == "bounded_planner" else ACTION_NAMES[int(policy)],
            "seed": seed,
            "initial_has_flag": initial_has_flag,
            "pickups": metrics["pickups"],
            "captures": metrics["captures"],
            "pickup_time": metrics["pickup_time"],
            "capture_time": metrics["capture_time"],
            "tags_before_pickup": metrics["tags_before_pickup"],
            "tags_after_pickup": metrics["tags_after_pickup"],
            "tagged": bool(arena.game.state["agent_is_tagged"][0]),
            "duration": float(arena.game.current_time),
            "stop_reason": stop_reason,
        }
    finally:
        arena.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=51000)
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.9, 1.0])
    parser.add_argument(
        "--starts",
        nargs="+",
        choices=["standard", "near_flag", "carry"],
        default=["standard", "near_flag", "carry"],
    )
    parser.add_argument("--maneuvers", type=int, nargs="+", choices=range(8))
    parser.add_argument("--planner-width", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("runs/full-speed-feasibility-1v1"))
    args = parser.parse_args()
    if args.episodes < 1 or args.planner_width < 1:
        parser.error("episodes and planner width must be positive")
    if any(not 0 < speed <= 1 for speed in args.speeds):
        parser.error("speeds must be in (0, 1]")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    maneuvers = args.maneuvers
    policy_count = len(maneuvers) + 1 if maneuvers else 5
    total = len(args.speeds) * len(args.starts) * policy_count * args.episodes
    for speed in args.speeds:
        for start in args.starts:
            selected_maneuvers = maneuvers or (
                [1, 4, 5, 6] if start == "carry" else [0, 2, 3, 6]
            )
            policies = list(selected_maneuvers) + ["bounded_planner"]
            for policy in policies:
                for episode in range(args.episodes):
                    row = run_episode(
                        speed, start, policy, args.seed + episode, args.planner_width
                    )
                    rows.append(row)
                    print(f"{len(rows)}/{total} {row}", flush=True)

    with (args.out / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["speed"], row["start_mode"], row["policy"])].append(row)
    summary = {
        "experiment": "bounded two-decision planner and stage-specific maneuver set",
        "conditions": {
            "arena": "standard 160x80",
            "opponent": "chase",
            "opponent_speeds": args.speeds,
            "action_repeat": 1,
            "episode_horizon_seconds": 240,
            "approach_episodes_stop_on": "first tag, capture, or horizon",
            "carry_episodes_stop_on": "capture, first tag, or horizon",
            "standard_start": "TeamArena default seeded random starts",
            "near_flag_start": (
                "attacker at x=0.87W, y=0.84H plus/minus 0.025H; "
                "defender at x=0.70W, y=0.68H with paired vertical offset"
            ),
            "carry_start": (
                "carrier at x=0.62W, y=0.84H plus/minus 0.025H; "
                "defender at x=0.82W, y=0.68H with paired vertical offset"
            ),
            "planner": "two simulator decisions, first-step beam width 3, eight actions",
            "seed_pairing": "same seed across policies within each speed and start condition",
            "simulator_capture": (
                "PyQuaticus awards capture when the carrier crosses onto its own side; "
                "the screen does not establish a return to the home base"
            ),
        },
        "episodes_per_condition": args.episodes,
        "seed_start": args.seed,
        "planner_width": args.planner_width,
        "opponent": "chase",
        "repeat": 1,
        "results": {
            f"speed={speed},start={start},policy={policy}": {
                "episodes": len(group),
                "initial_carry_episodes": sum(row["initial_has_flag"] for row in group),
                "pickup_episodes": sum(row["pickups"] > 0 for row in group),
                "capture_episodes": sum(row["captures"] > 0 for row in group),
                "tags_after_pickup": sum(row["tags_after_pickup"] for row in group),
            }
            for (speed, start, policy), group in sorted(groups.items())
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
