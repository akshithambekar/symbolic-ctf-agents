"""Run an episode-budgeted probe-lure-switch PPO trial and held-out evaluation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from maritime_nsrl.experiments import save_json
from maritime_nsrl.probe_lure import INSTRUCTION, mission_phase
from maritime_nsrl.research_env import ArenaConfig, TeamArena, SharedTeamVecEnv
from maritime_nsrl.translation import translate


class EpisodeBudget(BaseCallback):
    def __init__(self, output, episodes):
        super().__init__()
        self.output, self.episodes, self.rows = output, episodes, []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "metrics" in info:
                row = {"episode": len(self.rows) + 1, "timesteps": self.num_timesteps, **info["metrics"]}
                self.rows.append(row)
                with (self.output / "training_episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(f"Training episode {len(self.rows)}/{self.episodes}: "
                      f"captures={row['captures']} mission={row['ltlf_satisfied']}", flush=True)
        return len(self.rows) < self.episodes


def summary(rows):
    wins = sum(row["captures"] > row["opponent_captures"] for row in rows)
    losses = sum(row["captures"] < row["opponent_captures"] for row in rows)
    return {
        "episodes": len(rows), "captures": sum(row["captures"] for row in rows),
        "wins": wins, "losses": losses, "draws_or_timeouts": len(rows) - wins - losses,
        "pickups": sum(row["pickups"] for row in rows),
        "full_mission_completions": sum(row["ltlf_satisfied"] for row in rows),
        "numerical_terminations": sum(row["numerical_terminations"] for row in rows),
        "win_rate": wins / len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--tag-penalty", type=float, default=2.0)
    parser.add_argument("--probe-action", action="store_true")
    parser.add_argument("--warmup-episodes", type=int, default=0)
    args = parser.parse_args()
    if min(args.episodes, args.evaluation_episodes) <= 0:
        parser.error("Episode budgets must be positive")
    if args.warmup_episodes < 0 or args.tag_penalty < 0:
        parser.error("Warm-up episodes and tag penalty must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    spec = translate(INSTRUCTION, logic_backend="dfa")
    config = ArenaConfig(
        team_size=1, opponent="chase", formula=spec["ltl"], trace_stride=1,
        tag_penalty=args.tag_penalty, probe_action=args.probe_action,
    )
    save_json(args.output / "specification.json", spec)
    save_json(args.output / "arena_config.json", asdict(config))
    save_json(args.output / "config.json", {
        "mode": "team", "formula": spec["ltl"], "team_size": 1, "seed": 7,
        "training_episodes": args.episodes, "evaluation_seed_start": 10000,
        "warmup_episodes": args.warmup_episodes,
        "win_definition": "blue captures exceed red captures at episode end",
        "budget_note": "Callback stops at exact episode budget; final partial PPO rollout is discarded.",
    })
    torch.set_num_threads(1)
    vec = SharedTeamVecEnv(TeamArena(config))
    callback = EpisodeBudget(args.output, args.episodes)
    try:
        model = PPO("MlpPolicy", vec, learning_rate=3e-4, n_steps=256, batch_size=64,
                    n_epochs=5, gamma=config.gamma, ent_coef=0.01,
                    policy_kwargs={"net_arch": [128, 128]}, seed=7, device="cpu", verbose=0)
        model.set_logger(configure(str(args.output), ["csv"]))
        if args.warmup_episodes:
            warmup = args.output / "warmup"
            warmup.mkdir()
            config.curriculum_phase = 2
            save_json(warmup / "arena_config.json", asdict(config))
            save_json(warmup / "specification.json", spec)
            save_json(warmup / "config.json", {
                "mode": "team", "formula": spec["ltl"], "team_size": 1,
                "training_episodes": args.warmup_episodes, "curriculum_phase": 2,
            })
            warmup_callback = EpisodeBudget(warmup, args.warmup_episodes)
            model.learn(total_timesteps=args.warmup_episodes * 1000, callback=warmup_callback)
            model.save(warmup / "policy.zip")
            save_json(warmup / "summary.json", {
                **summary(warmup_callback.rows),
                "probe_and_pursuit_successes": sum(
                    r.get("curriculum_success", False) for r in warmup_callback.rows
                ),
            })
            vec.close()
            config.curriculum_phase = 0
            vec = SharedTeamVecEnv(TeamArena(config))
            model.set_env(vec, force_reset=True)
        model.learn(total_timesteps=args.episodes * 1000, callback=callback)
        model.save(args.output / "policy.zip")
        save_json(args.output / "training_budget.json", {
            "actual_transitions": model.num_timesteps, "ppo_updates": model._n_updates,
            "completed_episodes": len(callback.rows),
        })
    finally:
        vec.close()
    model = PPO.load(args.output / "policy.zip", device="cpu")
    rows = []
    arena = TeamArena(config)
    try:
        for episode in range(args.evaluation_episodes):
            obs = arena.reset(seed=10000 + episode)
            furthest_phase = 0
            while not arena.finished:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, info = arena.step(action)
                if not arena.monitor.impossible:
                    furthest_phase = max(furthest_phase, mission_phase(arena.monitor))
            row = {"seed": 10000 + episode, "furthest_phase": furthest_phase, **info["metrics"]}
            rows.append(row)
            with (args.output / "evaluation_episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if (episode + 1) % 10 == 0:
                print(f"Evaluation {episode + 1}/{args.evaluation_episodes}: "
                      f"captures={sum(r['captures'] for r in rows)}", flush=True)
    finally:
        arena.close()
    report = {"training": summary(callback.rows), "held_out_evaluation": summary(rows)}
    save_json(args.output / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
