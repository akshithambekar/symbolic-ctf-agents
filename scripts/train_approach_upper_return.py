"""Train an attacker to reach the flag, then evaluate it with a scripted return."""

import argparse
import copy
import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from maritime_nsrl.experiments import save_json
from maritime_nsrl.team_env import ArenaConfig, SharedTeamVecEnv, TeamArena


UPPER_RETURN_ACTION = 5


class PickupTerminalArena(TeamArena):
    """End training episodes at pickup so PPO learns only the approach leg."""

    def step(self, actions):
        observation, rewards, done, info = super().step(actions)
        if not done and self.game.state["agent_has_flag"][0]:
            self.finished = True
            done = True
            metrics = copy.deepcopy(self.metrics)
            metrics["training_terminated_on_pickup"] = True
            info = {
                "metrics": {
                    **metrics,
                    "duration": float(self.game.current_time),
                    "ltlf_satisfied": self.monitor.satisfied,
                    "team_size": self.n,
                    "scenario": self.config.scenario,
                    "opponent": self.config.opponent,
                    "opponent_speed": self.config.opponent_speed,
                    "trajectory": None,
                }
            }
        return observation, rewards, done, info


class EpisodeLog(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.rows = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "metrics" not in info:
                continue
            row = {"episode": len(self.rows) + 1, "timesteps": self.num_timesteps, **info["metrics"]}
            self.rows.append(row)
            with self.path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if row.get("training_terminated_on_pickup"):
                print(
                    f"Training episode {len(self.rows)}: pickup at {row['duration']:.0f}s",
                    flush=True,
                )
            elif len(self.rows) % 100 == 0:
                print(f"Training episodes completed: {len(self.rows)}", flush=True)
        return True


def flank_action(arena):
    state = arena.game.state
    blue = state["agent_position"][0]
    red = state["agent_position"][1]
    if state["agent_has_flag"][0]:
        return UPPER_RETURN_ACTION
    if blue[0] < 0.78 * arena.size[0] and blue[1] < red[1] - 5:
        return 2
    if blue[0] < 0.78 * arena.size[0] and blue[1] > red[1] + 5:
        return 3
    return 0


def evaluate_policy(model, config, seeds):
    arena = TeamArena(config)
    rows = []
    try:
        for seed in seeds:
            observation = arena.reset(seed)
            while not arena.finished:
                if arena.game.state["agent_has_flag"][0]:
                    action = UPPER_RETURN_ACTION
                else:
                    action, _ = model.predict(observation, deterministic=True)
                observation, _, _, info = arena.step(action)
            rows.append({"policy": "ppo_plus_upper_return", "seed": seed, **info["metrics"]})
    finally:
        arena.close()
    return rows


def evaluate_flank(config, seeds):
    arena = TeamArena(config)
    rows = []
    try:
        for seed in seeds:
            arena.reset(seed)
            while not arena.finished:
                _, _, _, info = arena.step([flank_action(arena)])
            rows.append({"policy": "flank_plus_upper_return", "seed": seed, **info["metrics"]})
    finally:
        arena.close()
    return rows


def policy_summary(rows):
    pickup_episodes = sum(row["pickups"] > 0 for row in rows)
    captures = sum(row["captures"] > 0 for row in rows)
    pickup_events = sum(row["pickups"] for row in rows)
    return {
        "episodes": len(rows),
        "pickup_episodes": pickup_episodes,
        "pickup_events": pickup_events,
        "captures": captures,
        "pickup_rate": pickup_episodes / len(rows),
        "capture_rate": captures / len(rows),
        "captures_given_pickup_episode": captures / pickup_episodes if pickup_episodes else None,
        "mission_completions": sum(row["ltlf_satisfied"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/approach-ppo-upper-return-1v1"))
    parser.add_argument("--steps", type=int, default=50_176)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evaluation-episodes", type=int, default=50)
    parser.add_argument("--evaluation-seed", type=int, default=20_000)
    parser.add_argument("--opponent-speed", type=float, default=0.6)
    args = parser.parse_args()
    if args.steps < 256 or args.steps % 256 or args.evaluation_episodes < 1:
        parser.error("steps must be a positive multiple of 256 and evaluations must be positive")
    if not 0 < args.opponent_speed <= 1:
        parser.error("opponent speed must be in (0, 1]")
    existing = {path.name for path in args.output.iterdir()} if args.output.exists() else set()
    if existing - {"decisions.tsv"}:
        parser.error(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    config = ArenaConfig(
        team_size=1,
        formula="G(in_bounds) & F(has_flag & F(captured))",
        guided=True,
        reward_shaping=True,
        temporal_state=True,
        opponent="chase",
        opponent_speed=args.opponent_speed,
        repeat=5,
        pickup_reward=20.0,
        tag_penalty=0.0,
        trace_stride=0,
    )
    save_json(args.output / "arena_config.json", asdict(config))
    save_json(args.output / "config.json", {
        "experiment": "PPO approach and pickup, scripted upper-lane return",
        "training_termination": "first flag pickup",
        "training_opponent_speed": args.opponent_speed,
        "scripted_return_action": UPPER_RETURN_ACTION,
        "training_steps": args.steps,
        "seed": args.seed,
        "evaluation_seed_start": args.evaluation_seed,
        "evaluation_episodes": args.evaluation_episodes,
        "tag_penalty": config.tag_penalty,
        "pickup_reward": config.pickup_reward,
    })
    save_json(args.output / "ppo_config.json", {
        "algorithm": "PPO",
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 5,
        "gamma": config.gamma,
        "ent_coef": 0.01,
        "net_arch": [128, 128],
        "device": "cpu",
    })

    torch.set_num_threads(1)
    vec = SharedTeamVecEnv(PickupTerminalArena(config))
    vec.seed(args.seed)
    callback = EpisodeLog(args.output / "training_episode_metrics.jsonl")
    try:
        model = PPO(
            "MlpPolicy",
            vec,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=5,
            gamma=config.gamma,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [128, 128]},
            seed=args.seed,
            device="cpu",
            verbose=0,
        )
        model.set_logger(configure(str(args.output), ["csv"]))
        model.learn(total_timesteps=args.steps, callback=callback)
        model.save(args.output / "policy.zip")
        save_json(args.output / "training_budget.json", {
            "actual_transitions": model.num_timesteps,
            "ppo_updates": model._n_updates,
            "completed_episodes": len(callback.rows),
            "pickup_episodes": sum(row.get("pickups", 0) > 0 for row in callback.rows),
        })
    finally:
        vec.close()

    model = PPO.load(args.output / "policy.zip", device="cpu")
    evaluation_config = ArenaConfig(**asdict(config))
    seeds = range(args.evaluation_seed, args.evaluation_seed + args.evaluation_episodes)
    model_rows = evaluate_policy(model, evaluation_config, seeds)
    baseline_rows = evaluate_flank(
        evaluation_config,
        range(args.evaluation_seed, args.evaluation_seed + args.evaluation_episodes),
    )
    rows = model_rows + baseline_rows
    fields = sorted({key for row in rows for key in row})
    with (args.output / "evaluation_episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "training": {
            "episodes": len(callback.rows),
            "pickup_episodes": sum(row.get("pickups", 0) > 0 for row in callback.rows),
            "pickup_episode_rate": (
                sum(row.get("pickups", 0) > 0 for row in callback.rows) / len(callback.rows)
                if callback.rows else None
            ),
        },
        "held_out_evaluation": {
            "ppo_plus_upper_return": policy_summary(model_rows),
            "flank_plus_upper_return": policy_summary(baseline_rows),
            "paired_seeds": args.evaluation_episodes,
            "seed_start": args.evaluation_seed,
        },
    }
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
