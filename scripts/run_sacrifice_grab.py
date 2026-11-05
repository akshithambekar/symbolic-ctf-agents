"""Evaluate the two-boat decoy and timed-grab script against 2v2 chase."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from maritime_nsrl.team_env import ArenaConfig, TeamArena


DECOY, GRABBER = 0, 1
RED_START = 2
ACTION = {
    "flag": 0,
    "home": 1,
    "lower_flank": 2,
    "upper_flank": 3,
    "upper_lane": 5,
    "hold": 7,
}


def route_points(arena, grabber, carrying):
    width, height = arena.size
    flag = arena.game.state["flag_home"][1].astype(float)
    upper = np.array([0.40 * width, 0.85 * height])
    current = arena.game.state["agent_position"][grabber].astype(float)
    if carrying:
        return [current, upper]
    flank = np.array([0.91 * width, 0.88 * height])
    return [current, flank, flag, upper]


def clip_to_red_side(points, midfield):
    """Return the route segments at or beyond midfield."""
    clipped = []
    for start, end in zip(points, points[1:]):
        if start[0] >= midfield and end[0] >= midfield:
            clipped.append((start, end))
        elif (start[0] < midfield <= end[0]) or (end[0] < midfield <= start[0]):
            fraction = (midfield - start[0]) / (end[0] - start[0])
            crossing = start + fraction * (end - start)
            clipped.append((crossing, end) if end[0] >= midfield else (start, crossing))
    return clipped


def point_segment_distance(point, start, end):
    delta = end - start
    length_sq = float(delta @ delta)
    fraction = 0.0 if length_sq == 0 else float(np.clip((point - start) @ delta / length_sq, 0, 1))
    return float(np.linalg.norm(point - (start + fraction * delta)))


def timing_check(arena):
    state = arena.game.state
    width = float(arena.size[0])
    positions = state["agent_position"]
    speeds = np.asarray(arena.game.max_speeds, dtype=float)
    blue_speed = float(speeds[GRABBER])
    threats = [
        index
        for index in range(RED_START, RED_START + arena.n)
        if not state["agent_is_tagged"][index]
    ]
    points = route_points(arena, GRABBER, bool(state["agent_has_flag"][GRABBER]))
    segments = clip_to_red_side(points, width / 2)
    grab_time = sum(float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:])) / blue_speed
    race_slacks, target_slacks = [], []
    for red in threats:
        red_speed = float(speeds[red] * arena.config.opponent_speed)
        route_distance = (
            min(
                point_segment_distance(positions[red], start, end)
                for start, end in segments
            )
            if segments
            else float("inf")
        )
        travel_time = max(route_distance - arena.game.catch_radius, 0.0) / red_speed
        cooldown_remaining = max(
            arena.game.tagging_cooldown - float(state["agent_tagging_cooldown"][red]),
            0.0,
        )
        red_eta = max(travel_time, cooldown_remaining)
        race_slacks.append(red_eta - grab_time)
        if not state["agent_is_tagged"][DECOY]:
            target_slacks.append(
                float(
                    np.linalg.norm(positions[red] - positions[GRABBER])
                    - np.linalg.norm(positions[red] - positions[DECOY])
                )
            )
    race_margin = min(race_slacks, default=float("inf"))
    target_margin = min(target_slacks, default=float("inf"))
    return {
        "red_threats": threats,
        "grab_route_seconds": grab_time,
        "race_margin_seconds": race_margin,
        "target_margin_m": target_margin,
        "race_ok": race_margin > 0,
        "target_ok": target_margin > 0,
    }


def guard_stays_home(arena):
    """Identify a defender near its flag while its teammate pursues the decoy."""
    state = arena.game.state
    if state["agent_on_sides"][DECOY]:
        return False
    red = range(RED_START, RED_START + arena.n)
    untagged = [
        index
        for index in red
        if not state["agent_is_tagged"][index]
    ]
    if len(untagged) < 2:
        return False
    flag = state["flag_home"][1]
    decoy = state["agent_position"][DECOY]
    grabber = state["agent_position"][GRABBER]
    distances = {
        index: float(np.linalg.norm(state["agent_position"][index] - decoy))
        for index in untagged
    }
    guard = min(
        untagged,
        key=lambda index: float(np.linalg.norm(state["agent_position"][index] - flag)),
    )
    chaser = next(index for index in untagged if index != guard)
    return (
        np.linalg.norm(state["agent_position"][guard] - flag) < 35
        and distances[chaser] < np.linalg.norm(state["agent_position"][chaser] - grabber)
    )


def choose_actions(arena, phase):
    state = arena.game.state
    decoy_tagged = bool(state["agent_is_tagged"][DECOY])
    if phase == "sacrifice" and not decoy_tagged:
        return ACTION["flag"], ACTION["upper_lane"]
    if phase == "grab" or state["agent_has_flag"][GRABBER]:
        decoy_action = ACTION["home"] if decoy_tagged else ACTION["lower_flank"]
        if state["agent_has_flag"][GRABBER]:
            grab_action = ACTION["upper_lane"]
        else:
            pos = state["agent_position"][GRABBER]
            grab_action = (
                ACTION["flag"]
                if pos[0] >= 0.86 * arena.size[0] and pos[1] >= 0.82 * arena.size[1]
                else ACTION["upper_flank"]
            )
        return decoy_action, grab_action
    if decoy_tagged:
        return ACTION["home"], ACTION["hold"]
    return ACTION["lower_flank"], ACTION["upper_lane"]


def run_episode(seed, record_decisions=True):
    config = ArenaConfig(
        team_size=2,
        formula="G(in_bounds) & F(has_flag & F(captured))",
        guided=False,
        reward_shaping=False,
        temporal_state=False,
        opponent="chase",
        opponent_speed=1.0,
        repeat=1,
        random_starts=True,
        horizon=240.0,
    )
    arena = TeamArena(config)
    decision_rows = []
    try:
        arena.reset(seed)
        phase = "draw"
        sacrifice_used = False
        stop_reason = "horizon"
        while not arena.finished:
            now = float(arena.game.current_time)
            state = arena.game.state
            if state["agent_is_tagged"][DECOY]:
                phase = "recover"
            elif phase == "recover" and not state["agent_is_tagged"][DECOY]:
                phase = "draw"
            if phase == "draw" and guard_stays_home(arena) and not sacrifice_used:
                phase = "sacrifice"
            check = timing_check(arena)
            if phase != "grab" and check["race_ok"] and check["target_ok"]:
                phase = "grab"
            actions = choose_actions(arena, phase)
            if phase == "sacrifice" and actions[DECOY] == ACTION["flag"]:
                sacrifice_used = True
            if record_decisions:
                if check["race_ok"] and check["target_ok"]:
                    reason = "go"
                elif not check["race_ok"] and not check["target_ok"]:
                    reason = "race_and_targeting"
                elif not check["race_ok"]:
                    reason = "race"
                else:
                    reason = "targeting"
                decision_rows.append(
                    {
                        "seed": seed,
                        "time_s": now,
                        "phase": phase,
                        "action_decoy": actions[DECOY],
                        "action_grabber": actions[GRABBER],
                        "grab_route_s": round(check["grab_route_seconds"], 3),
                        "race_margin_s": round(check["race_margin_seconds"], 3),
                        "target_margin_m": round(check["target_margin_m"], 3),
                        "race_ok": check["race_ok"],
                        "target_ok": check["target_ok"],
                        "gate_result": reason,
                        "red_threat_ids": ",".join(map(str, check["red_threats"])),
                        "decoy_tagged": bool(state["agent_is_tagged"][DECOY]),
                        "grabber_has_flag": bool(state["agent_has_flag"][GRABBER]),
                        "blue_positions": json.dumps(
                            state["agent_position"][:2].astype(float).tolist(), separators=(",", ":")
                        ),
                        "red_positions": json.dumps(
                            state["agent_position"][2:4].astype(float).tolist(), separators=(",", ":")
                        ),
                    }
                )
            arena.step(actions)
            if arena.metrics["captures"]:
                stop_reason = "capture"
                break
            if arena.game.current_time >= config.horizon:
                break
        result = {
            "seed": seed,
            "captures": int(arena.metrics["captures"]),
            "pickups": int(arena.metrics["pickups"]),
            "tags_received": int(arena.metrics["tags_received"]),
            "capture_time_s": arena.metrics["capture_time"],
            "duration_s": float(arena.game.current_time),
            "stop_reason": stop_reason,
            "sacrifice_used": sacrifice_used,
            "final_blue_positions": arena.game.state["agent_position"][:2].astype(float).tolist(),
            "minimum_defender_distance_m": arena.metrics["defender_distance_min"],
            "decisions": len(decision_rows),
            "gate_go_decisions": sum(row["gate_result"] == "go" for row in decision_rows),
            "gate_race_failures": sum(not row["race_ok"] for row in decision_rows),
            "gate_target_failures": sum(not row["target_ok"] for row in decision_rows),
        }
        return result, decision_rows
    finally:
        arena.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=62000)
    parser.add_argument("--out", type=Path, default=Path("runs/sacrifice-grab-2v2"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    rows, decisions = [], []
    for episode in range(args.episodes):
        seed = args.seed + episode
        row, logs = run_episode(seed)
        row["fresh_replay_captures"] = None
        if row["captures"]:
            replay, _ = run_episode(seed, record_decisions=False)
            row["fresh_replay_captures"] = replay["captures"]
            row["fresh_replay_capture_time_s"] = replay["capture_time_s"]
            row["fresh_replay_match"] = (
                replay["captures"] == row["captures"]
                and replay["capture_time_s"] == row["capture_time_s"]
            )
        else:
            row["fresh_replay_capture_time_s"] = None
            row["fresh_replay_match"] = None
        rows.append(row)
        decisions.extend(logs)
        print(f"{episode + 1}/{args.episodes} {row}", flush=True)

    with (args.out / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "decisions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    summary = {
        "experiment": "repeat-1 2v2 timed decoy and grab against full-speed pure-pursuit chase",
        "episodes": args.episodes,
        "seed_start": args.seed,
        "horizon_s": 240,
        "repeat": 1,
        "opponent_speed": 1.0,
        "capture_episodes": sum(row["captures"] > 0 for row in rows),
        "fresh_replay_captures": sum(row["fresh_replay_captures"] or 0 for row in rows),
        "verified_replay_pairs": sum(row["fresh_replay_match"] is True for row in rows),
        "sacrifice_episodes": sum(row["sacrifice_used"] for row in rows),
        "gate_reasons": {
            "go": sum(row["gate_go_decisions"] for row in rows),
            "race_failed": sum(not log["race_ok"] for log in decisions),
            "targeting_failed": sum(not log["target_ok"] for log in decisions),
            "both_failed": sum(
                not log["race_ok"] and not log["target_ok"]
                for log in decisions
            ),
        },
        "results": rows,
        "decision_log": (
            "decisions.csv; each decision records route time, race margin, "
            "target margin, and gate result"
        ),
        "limits": [
            "The race margin is a geometric constant-speed screen against the current route and includes cooldown recovery, not a full future pursuit simulation.",
            "The script uses the imported chase bot and its 30-second tagging cooldown.",
            "A capture is the simulator's midfield capture event and does not establish competition-rule fidelity.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
