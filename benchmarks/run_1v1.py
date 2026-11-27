"""Run the resolved PyQuaticus 1v1 checkpoint benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maritime_nsrl.experiments import save_json
from maritime_nsrl.opponents import steer
from maritime_nsrl.single_agent_env import MaritimeEnv
from maritime_nsrl.translation import load_spec

SUITE_PATH = ROOT / "benchmarks/1v1/v1.json"
DEFAULT_CHECKPOINT = ROOT / "runs/fixed-start-orbit-ppo-20261119/training/policy.zip"
DEFAULT_SPEC = ROOT / "runs/ppo-nov/specification.json"
DEFAULT_RUN_META = ROOT / "runs/fixed-start-orbit-ppo-20261119/config.json"
DEFAULT_EXPERIMENT = ROOT / "runs/fixed-start-orbit-ppo-20261119/experiment.py"
FAMILIES = {"flag_orbit": 14, "midfield_pace": 12, "interior_pace": 10,
            "patrol": 8, "brief_pursuit": 5, "reactive": 1}


def validate_suite(suite):
    cases = suite["cases"]
    if len(cases) != 50 or [c["id"] for c in cases] != list(range(1, 51)):
        raise ValueError("Suite requires stable, ordered case IDs 1 through 50")
    if Counter(c["family"] for c in cases) != Counter(FAMILIES):
        raise ValueError("Incorrect family counts")
    if sum(c["family"] == "reactive" for c in cases) != 1 or cases[-1]["id"] != 50:
        raise ValueError("Exactly case 50 must be the unrestricted reactive case")
    for c in cases:
        if c["seed"] != 51000 + c["id"]:
            raise ValueError(f"Case {c['id']} has an invalid seed")
        x, y = c["blue_start"]
        if not (0 <= x < 80 and 0 <= y <= 80):
            raise ValueError(f"Case {c['id']} has an invalid Blue start")
        if not 0.8 <= c["red_speed_fraction"] <= 1.0:
            raise ValueError(f"Case {c['id']} speed is outside 80–100%")
        route = c["red_route"]
        points = [route["center"]] if route["type"] == "orbit" else route["waypoints"]
        if route.get("start") is not None: points.append(route["start"])
        if route["type"] == "orbit":
            if route["radius"] not in (10, 14, 16) or not 0 <= route["start_angle_degrees"] < 360:
                raise ValueError(f"Case {c['id']} has invalid orbit geometry")
        elif len(points) < 2:
            raise ValueError(f"Case {c['id']} needs a route with at least two points")
        if any(not (0 <= p[0] <= 160 and 0 <= p[1] <= 80) for p in points):
            raise ValueError(f"Case {c['id']} has an out-of-bounds route")
    if suite["deadline_sim_seconds"] != 240:
        raise ValueError("Unexpected benchmark deadline")


def heading_to(target, origin):
    delta = np.asarray(target, dtype=float) - np.asarray(origin, dtype=float)
    return float(np.degrees(np.arctan2(delta[0], delta[1])) % 360)


def orbit_start(route):
    a = math.radians(route["start_angle_degrees"])
    center = np.asarray(route["center"], dtype=float)
    radius = route["radius"]
    pos = center + radius * np.array([math.cos(a), math.sin(a)])
    heading = ((180-route["start_angle_degrees"]) if route["direction"] == "cw"
               else -route["start_angle_degrees"]) % 360
    return pos.tolist(), heading


def segment_distance(point, a, b):
    p, a, b = np.asarray(point), np.asarray(a), np.asarray(b)
    v = b-a
    t = np.clip(np.dot(p-a, v) / max(np.dot(v, v), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t*v)))


class ScenarioRed:
    """Deterministic patrol with a bounded, cooldown-based optional pursuit."""
    def __init__(self, route, speed_fraction, reactive=False):
        self.route = route
        self.speed_fraction = speed_fraction
        self.reactive = reactive
        self.waypoint = 1
        self.route_initialized = False
        self.pursuing = False
        self.pursuit_started = None
        self.cooldown_until = 0.0

    def position(self, game):
        return np.asarray(game.state["agent_position"][1], dtype=float)

    def blue(self, game):
        return np.asarray(game.state["agent_position"][0], dtype=float)

    def route_distance(self, game):
        r, red = self.route, self.position(game)
        if r["type"] == "orbit":
            return abs(float(np.linalg.norm(red - r["center"])) - r["radius"])
        pts = [np.asarray(p) for p in r["waypoints"]]
        return min(segment_distance(red, a, b) for a, b in zip(pts, pts[1:] + pts[:1]))

    def route_target(self, game):
        r, red = self.route, self.position(game)
        if r["type"] == "orbit":
            center = np.asarray(r["center"], dtype=float)
            angle = math.atan2(red[1]-center[1], red[0]-center[0])
            delta = 0.35 * (1 if r["direction"] == "ccw" else -1)
            return center + r["radius"] * np.array([math.cos(angle+delta), math.sin(angle+delta)])
        points = [np.asarray(p, dtype=float) for p in r["waypoints"]]
        if r["type"] in ("pace", "pursuit"):
            direction = r.get("initial_leg", r.get("default_leg", "N"))
            if not self.route_initialized:
                self.waypoint = 1 if direction == "N" else 0
                self.route_initialized = True
            elif np.linalg.norm(red-points[self.waypoint]) < 4:
                self.waypoint = 1-self.waypoint
        elif np.linalg.norm(red-points[self.waypoint]) < 4:
            self.waypoint = (self.waypoint+1) % len(points)
        return points[self.waypoint]

    def action(self, game, index=1, learner_index=0):
        now = float(game.current_time)
        red, blue = self.position(game), self.blue(game)
        blue_in_red = blue[0] >= 80
        blue_tagged = bool(game.state["agent_is_tagged"][0])
        threshold = 0 if self.reactive else self.route.get("trigger_distance", 16)
        eligible = blue_in_red and not blue_tagged and (self.reactive or np.linalg.norm(red-blue) <= threshold)
        if self.pursuing:
            broken = (not blue_in_red or blue_tagged or
                      now-self.pursuit_started >= self.route.get("max_duration", 8) or
                      self.route_distance(game) >= self.route.get("leash_distance", 18))
            if broken:
                self.pursuing = False
                self.cooldown_until = now + (0 if self.reactive else self.route.get("cooldown", 12))
        if not self.pursuing and eligible and now >= self.cooldown_until:
            self.pursuing = True
            self.pursuit_started = now
        if self.pursuing:
            return steer(game, 1, blue, self.speed_fraction)
        target = self.route_target(game)
        return steer(game, 1, target, self.speed_fraction)


class CaseEvents:
    def __init__(self, case):
        self.case = case
        self.events = []
        self.milestones = {}
        self.closest_before_pickup = None
        self.previous = None
        self.pickup_time = None
        self.possession_started = None
        self.pickups = 0
        self.tags_before_pickup = 0
        self.tags_while_carrying = 0
        self.losses = []
        self.oob_events = 0
        self.oob_active = False
        self.last_route_sample = -1.0
        self.route_samples = []
        self.possession_durations = []
        self.controller = None

    def prime(self, game):
        s = game.state
        self.previous = {"bp": np.asarray(s["agent_position"][0], dtype=float).copy(),
                         "carrying": bool(s["agent_has_flag"][0]),
                         "tagged": bool(s["agent_is_tagged"][0]),
                         "captures": s["captures"].copy(), "grabs": s["grabs"].copy(),
                         "tags": s["tags"].copy(), "oob": bool(s["agent_oob"][0])}

    def observe(self, game):
        s, now = game.state, float(game.current_time)
        bp = np.asarray(s["agent_position"][0], dtype=float).copy()
        rp = np.asarray(s["agent_position"][1], dtype=float).copy()
        dist = float(np.linalg.norm(bp - np.asarray([140., 40.])))
        carrying = bool(s["agent_has_flag"][0])
        tagged = bool(s["agent_is_tagged"][0])
        oob = bool(s["agent_oob"][0])
        if now - self.last_route_sample >= 1.0 - 1e-8:
            self.last_route_sample = now
            self.route_samples.append({"time": now, "red_position": rp.tolist(),
                "red_heading": float(s["agent_heading"][1]), "red_speed": float(s["agent_speed"][1]),
                "red_on_blue_side": bool(s["agent_on_sides"][1]),
                "pursuing": bool(self.controller and self.controller.pursuing)})
        if self.previous is None:
            self.prime(game)
        if bp[0] >= 80 and "entered_red_territory" not in self.milestones:
            self.milestones["entered_red_territory"] = now
            self.events.append({"time": now, "event": "entered_red_territory", "position": bp.tolist()})
        if dist <= 15 and "approached_red_flag_15m" not in self.milestones:
            self.milestones["approached_red_flag_15m"] = now
            self.events.append({"time": now, "event": "approached_red_flag_15m", "position": bp.tolist()})
        if not carrying:
            self.closest_before_pickup = dist if self.closest_before_pickup is None else min(self.closest_before_pickup, dist)
        grabs = int(s["grabs"][0] - self.previous["grabs"][0])
        tags = int(s["tags"][0] - self.previous["tags"][0])
        if tags:
            if carrying: self.tags_while_carrying += tags
            else: self.tags_before_pickup += tags
            self.events.append({"time": now, "event": "tag", "count": tags, "stage": "carrying" if carrying else "before_pickup", "position": rp.tolist()})
        if grabs:
            self.pickups += grabs
            if not self.previous["carrying"] and carrying:
                self.possession_started = now
            if self.pickup_time is None:
                self.pickup_time = now
                self.milestones["picked_up_flag"] = now
            self.events.append({"time": now, "event": "pickup", "count": grabs, "position": bp.tolist()})
        if self.previous["bp"][0] >= 80 and bp[0] < 80 and carrying and "crossed_home_territory_carrying" not in self.milestones:
            self.milestones["crossed_home_territory_carrying"] = now
            self.events.append({"time": now, "event": "crossed_home_territory_carrying", "position": bp.tolist()})
        lost = self.previous["carrying"] and not carrying
        captures = int(s["captures"][0] - self.previous["captures"][0])
        if captures and "captured" not in self.milestones:
            self.milestones["captured"] = now
            self.events.append({"time": now, "event": "capture", "position": bp.tolist()})
            if self.possession_started is not None:
                self.possession_durations.append(now-self.possession_started)
        if lost and not captures:
            stage = "after_home_crossing" if "crossed_home_territory_carrying" in self.milestones else "before_home_crossing"
            self.losses.append({"time": now, "position": bp.tolist(), "stage": stage,
                                "escape_time_seconds": now-self.possession_started if self.possession_started is not None else None})
            if self.possession_started is not None:
                self.possession_durations.append(now-self.possession_started)
            self.events.append({"time": now, "event": "flag_loss", "position": bp.tolist(), "stage": stage})
            self.possession_started = None
        if oob and not self.oob_active:
            self.oob_events += 1
            self.events.append({"time": now, "event": "out_of_bounds", "position": bp.tolist()})
        self.oob_active = oob
        self.previous = {"bp": bp, "carrying": carrying, "tagged": tagged,
                         "captures": s["captures"].copy(), "grabs": s["grabs"].copy(),
                         "tags": s["tags"].copy(), "oob": oob}


def scenario_for(case, recorder):
    route = case["red_route"]
    if route["type"] == "orbit":
        red, red_heading = orbit_start(route)
    else:
        red = route.get("start", route["waypoints"][0])
        if route["type"] == "pace" or route["type"] == "pursuit":
            next_target = route["waypoints"][1 if route.get("initial_leg", route.get("default_leg")) == "N" else 0]
        else:
            next_target = route["waypoints"][1]
        red_heading = heading_to(next_target, red)
    controller = ScenarioRed(route, case["red_speed_fraction"], case["family"] == "reactive")
    recorder.controller = controller
    return {"init_dict": {"agent_position": [case["blue_start"], red],
                           "agent_heading": [90.0, red_heading], "agent_speed": [0.0, 0.0]},
            "opponent_controller": controller, "step_observer": recorder.observe}


def run_case(model, spec, case, settings):
    env = MaritimeEnv(spec, guided=settings["guided"], opponent="static",
                      horizon=settings["horizon"], gamma=settings["gamma"],
                      repeat=settings["repeat"], dynamics=settings["dynamics"])
    rec = CaseEvents(case)
    try:
        obs, _ = env.reset(seed=case["seed"], options={"scenario": scenario_for(case, rec)})
        rec.prime(env.game)
        state = env.game.state
        actual = np.asarray(state["agent_position"], dtype=float)
        wanted = np.asarray([case["blue_start"], scenario_for(case, rec)["init_dict"]["agent_position"][1]])
        if not np.allclose(actual, wanted, atol=1e-6):
            raise RuntimeError(f"case {case['id']} reset spawn mismatch: {actual} != {wanted}")
        route = case["red_route"]
        dist_to_flag = float(np.linalg.norm(actual[1]-[140,40]))
        if route["type"] == "orbit" and abs(dist_to_flag-route["radius"]) > 1e-5:
            raise RuntimeError(f"case {case['id']} orbit spawn radius mismatch")
        done = False
        decisions = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)
            decisions += 1
        metrics = info.get("metrics")
        if metrics is None:
            raise RuntimeError(f"case {case['id']} ended without terminal metrics")
        captures = int(env.game.state["captures"][0])
        if captures != int(metrics["flag_captures"]):
            raise RuntimeError(f"case {case['id']} simulator capture/reward metric mismatch")
        end_reason = "capture" if captures else "deadline" if float(env.game.current_time) >= settings["horizon"] else "simulator_termination"
        escape = rec.possession_durations[-1] if rec.possession_durations else None
        return {"case_id": case["id"], "family": case["family"], "seed": case["seed"],
                "capture": bool(captures), "capture_time_seconds": rec.milestones.get("captured"),
                "milestones": rec.milestones, "closest_red_flag_distance_before_first_pickup_m": rec.closest_before_pickup,
                "pickup_count": rec.pickups, "escape_time_seconds": escape,
                "tags_before_pickup": rec.tags_before_pickup, "tags_while_carrying": rec.tags_while_carrying,
                "possession_escape_times_seconds": rec.possession_durations,
                "flag_losses": rec.losses, "out_of_bounds_events": rec.oob_events,
                "episode_end_reason": end_reason, "simulated_seconds": float(env.game.current_time),
                "decisions": decisions, "events": rec.events, "red_route_samples_1s": rec.route_samples}
    finally:
        env.close()


def summarize(rows):
    def pack(subset):
        n = len(subset)
        counts = {key: sum(key in r["milestones"] for r in subset) for key in
                  ("entered_red_territory", "approached_red_flag_15m", "picked_up_flag", "crossed_home_territory_carrying", "captured")}
        return {"cases": n, "captures": sum(r["capture"] for r in subset), "capture_rate": sum(r["capture"] for r in subset)/n if n else 0,
                "milestone_counts": counts,
                "conditional_conversion": {"approach_given_entry": counts["approached_red_flag_15m"]/counts["entered_red_territory"] if counts["entered_red_territory"] else None,
                    "pickup_given_approach": counts["picked_up_flag"]/counts["approached_red_flag_15m"] if counts["approached_red_flag_15m"] else None,
                    "home_crossing_given_pickup": counts["crossed_home_territory_carrying"]/counts["picked_up_flag"] if counts["picked_up_flag"] else None,
                    "capture_given_pickup": counts["captured"]/counts["picked_up_flag"] if counts["picked_up_flag"] else None}}
    families = {name: pack([r for r in rows if r["family"] == name]) for name in FAMILIES}
    return {"overall": pack(rows), "by_family": families,
            "case_50": next((r for r in rows if r["case_id"] == 50), None)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--specification", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--case-id", type=int, help="Run one case as a checkpoint smoke test")
    args = ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    suite = json.loads(SUITE_PATH.read_text())
    validate_suite(suite)
    metadata = json.loads(DEFAULT_RUN_META.read_text())
    # The custom run's experiment records that these settings match the standard PPO contract.
    settings = {"guided": False, "opponent": "scenario-defined", "dynamics": "heron",
                "horizon": 240.0, "gamma": .99, "repeat": 5, "device": "cpu",
                "observation_shape": [19], "action_count": 7, "deterministic": True}
    torch.set_num_threads(1)
    model = PPO.load(args.checkpoint, device="cpu")
    spec = load_spec(args.specification)
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    provenance = {"checkpoint_path": str(args.checkpoint.resolve()), "checkpoint_sha256": sha,
                  "checkpoint_run_metadata": metadata, "run_experiment_source_sha256": hashlib.sha256(DEFAULT_EXPERIMENT.read_bytes()).hexdigest(),
                  "specification_path": str(args.specification.resolve()),
                  "specification_sha256": hashlib.sha256(args.specification.read_bytes()).hexdigest(),
                  "effective_environment": settings, "suite_path": str(SUITE_PATH.resolve()),
                  "suite_sha256": hashlib.sha256(SUITE_PATH.read_bytes()).hexdigest(),
                  "simulator_rules": suite["arena"], "pursuit_defaults": suite["pursuit_defaults"]}
    args.output.mkdir(parents=True)
    save_json(args.output / "provenance.json", provenance)
    selected_cases = suite["cases"]
    if args.case_id is not None:
        if not 1 <= args.case_id <= len(selected_cases): ap.error("--case-id must be from 1 to 50")
        selected_cases = [selected_cases[args.case_id-1]]
    rows = []
    started = time.perf_counter()
    for case in selected_cases:
        row = run_case(model, spec, case, settings)
        rows.append(row)
        with (args.output / "cases.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")
        print(f"CASE {case['id']:02d} {case['family']} capture={int(row['capture'])} time={row['capture_time_seconds']} pickups={row['pickup_count']}", flush=True)
    result = {"suite_version": "v1", "checkpoint_identity": provenance, "settings": settings,
              "runtime_seconds": time.perf_counter()-started, "case_results": rows,
              **summarize(rows)}
    save_json(args.output / "summary.json", result)
    print(json.dumps(result["overall"], indent=2), flush=True)

if __name__ == "__main__":
    main()
