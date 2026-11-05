"""Train and compare a runner-only PPO policy with a scripted 3v3 team.

The pilot trains against chase opponents and evaluates on held-out patrol and
static styles. Both arms use the same deterministic teammate controllers.
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from maritime_nsrl.team_env import ArenaConfig, TeamArena


def team_actions(arena, runner_action=None):
    """Runner attacks; interceptor protects home; support takes the wide lane."""
    state = arena.game.state
    positions = state["agent_position"]
    width = arena.size[0]
    actions = [0, 1, 2]

    runner = 0
    if state["agent_has_flag"][runner]:
        actions[runner] = 5 if positions[runner, 1] < arena.size[1] / 2 else 4
        if positions[runner, 0] < 0.42 * width:
            actions[runner] = 1
    elif positions[runner, 0] > 0.82 * width:
        actions[runner] = 0
    elif positions[runner, 1] > arena.size[1] * 0.55:
        actions[runner] = 2
    else:
        actions[runner] = 3

    # Interceptor holds the home lane until an enemy crosses midfield, then
    # tracks the nearest invading opponent with the shared intercept action.
    invaders = positions[3:]
    if np.any(invaders[:, 0] < width / 2):
        actions[1] = 8

    # Support attacks the opposite lane and returns through its matching lane.
    if state["agent_has_flag"][2]:
        actions[2] = 4 if positions[2, 1] < arena.size[1] / 2 else 5
        if positions[2, 0] < 0.42 * width:
            actions[2] = 1
    elif positions[2, 0] > 0.82 * width:
        actions[2] = 0
    else:
        actions[2] = 3 if positions[0, 1] > arena.size[1] / 2 else 2

    if runner_action is not None:
        actions[runner] = int(np.asarray(runner_action).item())
    return np.asarray(actions, dtype=np.int64)


class RunnerVecEnv(VecEnv):
    """Expose one PPO-controlled runner while keeping two teammates scripted."""

    def __init__(self, config):
        self.arena = TeamArena(config)
        self.pending_action = None
        super().__init__(1, self.arena.observation_space, self.arena.action_space)

    def reset(self):
        seed = self._seeds[0]
        self._reset_seeds()
        observation = self.arena.reset(seed)
        return observation[:1]

    def step_async(self, actions):
        self.pending_action = actions

    def step_wait(self):
        observation, rewards, done, info = self.arena.step(
            team_actions(self.arena, self.pending_action[0])
        )
        info = dict(info)
        if done:
            info["terminal_observation"] = observation[0].copy()
            observation = self.arena.reset()
        return (
            observation[:1],
            np.asarray([rewards[0]], dtype=np.float32),
            np.asarray([done], dtype=bool),
            [info],
        )

    def close(self):
        self.arena.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.arena, attr_name)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.arena, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self.arena, method_name)(*args, **kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]


def evaluate(model, config, opponent, seeds):
    from dataclasses import replace

    arena = TeamArena(replace(config, opponent=opponent, guided=False,
                              reward_shaping=False, temporal_state=False))
    rows = []
    try:
        for seed in seeds:
            obs = arena.reset(seed)
            while not arena.finished:
                action = None
                if model is not None:
                    action, _ = model.predict(obs[:1], deterministic=True)
                obs, _, _, info = arena.step(team_actions(arena, action))
            metrics = info["metrics"]
            grabs = metrics["pickups"]
            opponent_grabs = metrics["opponent_pickups"]
            captures = metrics["captures"]
            opponent_captures = metrics["opponent_captures"]
            if captures != opponent_captures:
                outcome = "win" if captures > opponent_captures else "loss"
            elif grabs != opponent_grabs:
                outcome = "win" if grabs > opponent_grabs else "loss"
            elif metrics["collisions_for"] != metrics["collisions_against"]:
                outcome = "win" if metrics["collisions_for"] < metrics["collisions_against"] else "loss"
            else:
                outcome = "tie"
            rows.append({
                "policy": "runner_ppo" if model is not None else "scripted_team",
                "evaluated_side": "blue",
                "opponent": opponent,
                "seed": seed,
                "duration": metrics["duration"],
                "score_differential": captures - opponent_captures,
                "outcome": outcome,
                "win_points": 1.0 if outcome == "win" else 0.5 if outcome == "tie" else 0.0,
                "captures_for": captures,
                "captures_against": opponent_captures,
                "grabs": grabs,
                "opponent_grabs": opponent_grabs,
                "grab_to_capture": captures / grabs if grabs else 0.0,
                "collisions": metrics["collisions"],
                "collisions_for": metrics["collisions_for"],
                "collisions_against": metrics["collisions_against"],
                "tags_received": metrics["tags_received"],
            })
    finally:
        arena.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/competition-ppo-pilot"))
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 64 or args.episodes < 1:
        parser.error("steps must be at least 64 and episodes positive")
    args.output.mkdir(parents=True, exist_ok=True)

    config = ArenaConfig(
        team_size=3,
        opponent="chase",
        formula="F(captured)",
        guided=False,
        reward_shaping=False,
        temporal_state=False,
        role_actions=True,
        horizon=600,
        max_score=3,
        tagging_cooldown=60,
        repeat=5,
    )
    (args.output / "arena_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (args.output / "pilot_config.json").write_text(json.dumps({
        "algorithm": "Stable-Baselines3 PPO",
        "training_steps": args.steps,
        "training_seed": args.seed,
        "training_opponent": "chase",
        "held_out_opponents": ["patrol", "static"],
        "evaluation_seeds": [args.seed + i for i in range(args.episodes)],
        "paired_arms": ["scripted_team", "runner_ppo"],
        "evaluated_side": "blue only",
        "ltl_guidance": False,
    }, indent=2) + "\n")
    env = RunnerVecEnv(config)
    try:
        model = PPO(
            "MlpPolicy", env, seed=args.seed, n_steps=64, batch_size=64,
            learning_rate=3e-4, gamma=0.99, verbose=1,
        )
        model.learn(total_timesteps=args.steps)
        model.save(args.output / "runner_ppo")
    finally:
        env.close()

    # Chase is the sole training style. Patrol and static remain held out.
    seeds = [args.seed + i for i in range(args.episodes)]
    rows = []
    for opponent in ("patrol", "static"):
        for policy in (None, model):
            rows.extend(evaluate(policy, config, opponent, seeds))
    with (args.output / "evaluation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for policy_name in ("scripted_team", "runner_ppo"):
        for opponent in ("patrol", "static"):
            group = [r for r in rows if r["policy"] == policy_name and r["opponent"] == opponent]
            summary[f"{policy_name} vs {opponent}"] = {
                "episodes": len(group),
                "mean_score_differential": float(np.mean([r["score_differential"] for r in group])),
                "win_rate": float(np.mean([r["win_points"] for r in group])),
                "wins": sum(r["outcome"] == "win" for r in group),
                "ties": sum(r["outcome"] == "tie" for r in group),
                "losses": sum(r["outcome"] == "loss" for r in group),
                "captures_for": int(sum(r["captures_for"] for r in group)),
                "captures_against": int(sum(r["captures_against"] for r in group)),
                "grab_to_capture": float(sum(r["captures_for"] for r in group) / max(sum(r["grabs"] for r in group), 1)),
                "collisions": int(sum(r["collisions"] for r in group)),
            }
            paired = []
            for seed in seeds:
                scripted = next(r for r in rows if r["policy"] == "scripted_team" and r["opponent"] == opponent and r["seed"] == seed)
                learned = next(r for r in rows if r["policy"] == "runner_ppo" and r["opponent"] == opponent and r["seed"] == seed)
                paired.append((learned["score_differential"] - scripted["score_differential"],
                               learned["win_points"] - scripted["win_points"]))
            summary[f"paired runner_ppo minus scripted_team vs {opponent}"] = {
                "episodes": len(paired),
                "mean_score_differential_change": float(np.mean([x[0] for x in paired])),
                "mean_win_points_change": float(np.mean([x[1] for x in paired])),
            }
    (args.output / "evaluation.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
