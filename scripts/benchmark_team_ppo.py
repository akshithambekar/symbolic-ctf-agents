"""Measure headless parameter-shared PPO throughput for 3v3 TeamArena."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from maritime_nsrl.team_env import ArenaConfig, SharedTeamVecEnv, TeamArena


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def machine_model():
    try:
        return subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def timed(method, key, timings):
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        try:
            return method(*args, **kwargs)
        finally:
            timings[key] += time.perf_counter() - start
    return wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/team-ppo-10k-benchmark"))
    parser.add_argument("--decisions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--warmup-decisions", type=int, default=384)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.decisions <= 0 or args.warmup_decisions <= 0 or args.threads <= 0:
        parser.error("decisions, warmup-decisions, and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    timings = {k: 0.0 for k in (
        "environment_step_wall_s", "simulator_step_wall_s", "reset_wall_s",
        "policy_inference_wall_s", "ppo_update_wall_s",
    )}
    counts = {"team_decisions": 0, "individual_blue_samples": 0, "simulator_steps": 0}

    # Competition-length 10-minute episodes, standard reset, 0.5 s decision cadence.
    config = ArenaConfig(
        team_size=3, guided=False, reward_shaping=False, temporal_state=False,
        opponent="chase", opponent_speed=1.0, scenario="standard",
        random_starts=True, horizon=600.0, max_score=3,
        tagging_cooldown=60.0, repeat=5,
    )
    arena = TeamArena(config)
    arena.reset(seed=args.seed)
    arena.reset = timed(arena.reset, "reset_wall_s", timings)
    arena.step = timed(arena.step, "environment_step_wall_s", timings)
    game_step = arena.game.step

    def count_game_step(*step_args, **step_kwargs):
        start = time.perf_counter()
        try:
            return game_step(*step_args, **step_kwargs)
        finally:
            timings["simulator_step_wall_s"] += time.perf_counter() - start
            counts["simulator_steps"] += 1
    arena.game.step = count_game_step
    vec = SharedTeamVecEnv(arena)

    model = PPO(
        "MlpPolicy", vec, learning_rate=3e-4, n_steps=256, batch_size=64,
        n_epochs=5, gamma=0.99, ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]}, seed=args.seed,
        device="cpu", verbose=0,
    )
    policy_forward = model.policy.forward

    def timed_forward(*forward_args, **forward_kwargs):
        start = time.perf_counter()
        try:
            return policy_forward(*forward_args, **forward_kwargs)
        finally:
            timings["policy_inference_wall_s"] += time.perf_counter() - start
    model.policy.forward = timed_forward
    train = model.train

    def timed_train(*train_args, **train_kwargs):
        start = time.perf_counter()
        try:
            return train(*train_args, **train_kwargs)
        finally:
            timings["ppo_update_wall_s"] += time.perf_counter() - start
    model.train = timed_train

    # Untimed warmup also forces a complete PPO update and initializes kernels.
    model.learn(total_timesteps=args.warmup_decisions * config.team_size, reset_num_timesteps=True)
    updates_after_warmup = model._n_updates
    for key in timings:
        timings[key] = 0.0
    counts = {k: 0 for k in counts}
    target_samples = args.decisions * config.team_size
    start = time.perf_counter()
    model.learn(total_timesteps=target_samples, reset_num_timesteps=True)
    training_wall = time.perf_counter() - start
    counts["team_decisions"] = model.num_timesteps // config.team_size
    counts["individual_blue_samples"] = model.num_timesteps
    vec.close()

    summary = {
        "benchmark": "headless full-team 3v3 PPO local CPU throughput",
        "seed": args.seed,
        "requested_team_decisions": args.decisions,
        "actual_counts": counts,
        "training_wall_s": training_wall,
        "throughput_team_decisions_per_s": counts["team_decisions"] / training_wall,
        "projection_250000_team_decisions_wall_s": 250000 * training_wall / counts["team_decisions"],
        "phase_wall_s": {
            "environment_step_including_simulator_s": timings["environment_step_wall_s"],
            "reset_s": timings["reset_wall_s"],
            "policy_inference_s": timings["policy_inference_wall_s"],
            "ppo_update_s": timings["ppo_update_wall_s"],
            "unattributed_training_overhead_s": max(0.0, training_wall - sum((
                timings["environment_step_wall_s"], timings["reset_wall_s"],
                timings["policy_inference_wall_s"], timings["ppo_update_wall_s"],
            ))),
        },
        "nested_environment_breakdown_wall_s": {
            "simulator_step_s": timings["simulator_step_wall_s"],
            "environment_bookkeeping_s": max(
                0.0, timings["environment_step_wall_s"] - timings["simulator_step_wall_s"]
            ),
        },
        "ppo_optimizer_epochs": model._n_updates - updates_after_warmup,
        "ppo_train_calls": counts["team_decisions"] // 256,
        "warmup_team_decisions": args.warmup_decisions,
        "setup_warmup_checkpoint_and_evaluation_in_training_wall": False,
        "configuration": {
            "team_size": 3, "policy_sharing": "one PPO policy shared across three correlated agent lanes",
            "opponent": "fixed PyQuaticus chase controllers, speed fraction 1.0",
            "arena_config": config.__dict__, "horizon_game_seconds": config.horizon,
            "action_repeat_simulator_steps": config.repeat,
            "nominal_decision_interval_game_seconds": 0.1 * config.repeat,
            "rollout_workers": 0, "vector_env_lanes": 3,
            "architecture": "MlpPolicy [128, 128]", "n_steps": 256,
            "batch_size": 64, "n_epochs": 5, "device": "cpu",
            "torch_threads": args.threads, "torch_interop_threads": 1,
            "rendering": False,
        },
        "machine": {
            "platform": platform.platform(), "processor": platform.processor(),
            "machine": platform.machine(), "model": machine_model(),
            "cpu_count": os.cpu_count(),
        },
        "versions": {name: version(name) for name in ("pyquaticus", "stable-baselines3", "numpy", "torch", "gymnasium")},
        "repository_commit": git_commit(),
        "scope_notes": [
            "Measures the existing TeamArena and SharedTeamVecEnv path with real SB3 PPO updates.",
            "One joint action supplies all three Blue agents; SB3 counts three individual samples per joint decision.",
            "Red behavior is the repository chase controller at full speed, not a competition league policy.",
            "PyQuaticus simulator rules and TeamArena observations/actions remain the repository's current abstractions.",
            "The policy is parameter-shared independent PPO, not three independently optimized role policies or centralized-critic MAPPO.",
            "PPO rounds the requested 30,000 individual samples to complete 768-sample rollouts.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    command = (
        ".venv/bin/python scripts/benchmark_team_ppo.py "
        f"--decisions {args.decisions} --seed {args.seed} "
        f"--warmup-decisions {args.warmup_decisions} --threads {args.threads} "
        f"--output {args.output}"
    )
    (args.output / "reproduce.sh").write_text("#!/bin/sh\nset -eu\ncd \"$(git rev-parse --show-toplevel)\"\n" + command + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
