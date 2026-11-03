"""Checkpointed PPO curriculum with dev selection and untouched full-mission tests."""

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from maritime_nsrl.probe_lure import INSTRUCTION, MILESTONES, mission_phase
from maritime_nsrl.research_env import ArenaConfig, SharedTeamVecEnv, TeamArena
from maritime_nsrl.translation import translate


PHASES = (2, 3, 4, 5, 6, 0)
REPOSITORY = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_files():
    return sorted((REPOSITORY / "src/maritime_nsrl").glob("*.py")) + [
        Path(__file__).resolve(), REPOSITORY / "uv.lock", REPOSITORY / "pyproject.toml",
    ]


def source_hashes():
    return {
        str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files()
    }


def save_bundle(directory, model, config, spec, metadata):
    directory.mkdir(parents=True, exist_ok=False)
    model.save(directory / "policy.zip")
    write_json(directory / "arena_config.json", asdict(config))
    write_json(directory / "specification.json", spec)
    write_json(directory / "config.json", {
        "mode": "team", "team_size": 1, "formula": config.formula, **metadata,
    })


def wilson(successes, total):
    if not total:
        return [0.0, 1.0]
    z = 1.959963984540054
    fraction = successes / total
    denominator = 1 + z * z / total
    center = (fraction + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(rows):
    total = len(rows)
    wins = sum(row["captures"] > row["opponent_captures"] for row in rows)
    losses = sum(row["captures"] < row["opponent_captures"] for row in rows)
    missions = sum(bool(row["ltlf_satisfied"]) and row["captures"] > 0 for row in rows)
    return {
        "episodes": total,
        "captures": sum(row["captures"] for row in rows),
        "pickups": sum(row["pickups"] for row in rows),
        "wins": wins, "losses": losses, "draws_or_timeouts": total - wins - losses,
        "win_rate": wins / total if total else 0.0,
        "full_mission_completions": missions,
        "full_mission_rate": missions / total if total else 0.0,
        "full_mission_wilson95": wilson(missions, total),
        "numerical_terminations": sum(row["numerical_terminations"] for row in rows),
        "milestone_completion_counts": {
            name: sum(row["furthest_phase"] > index for row in rows)
            for index, name in enumerate(MILESTONES)
        },
    }


def evaluate(model, config, episodes, seed_start, path):
    arena = TeamArena(replace(config, curriculum_phase=0, trace_stride=1))
    rows = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            obs = arena.reset(seed)
            furthest = 0
            while not arena.finished:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, info = arena.step(action)
                if not arena.monitor.impossible:
                    furthest = max(furthest, mission_phase(arena.monitor))
            row = {"seed": seed, **info["metrics"]}
            row["furthest_phase"] = max(furthest, row.get("furthest_phase", 0))
            rows.append(row)
    finally:
        arena.close()
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return rows, summarize(rows)


def selection_score(rows):
    summary = summarize(rows)
    return (
        summary["full_mission_completions"],
        summary["wins"],
        sum(row["furthest_phase"] for row in rows),
    )


def stage_success_rate(rows, phase):
    if phase in (0, 6):
        return summarize(rows)["full_mission_rate"]
    return sum(row["furthest_phase"] >= phase for row in rows) / len(rows)


class EpisodeLog(BaseCallback):
    def __init__(self, path, state, phase):
        super().__init__()
        self.path, self.state, self.phase = path, state, phase

    def _on_step(self):
        for info in self.locals["infos"]:
            if "metrics" in info:
                self.state["episode_count"] += 1
                row = {
                    "episode": self.state["episode_count"],
                    "timesteps": self.num_timesteps, "curriculum_phase": self.phase,
                    **info["metrics"],
                }
                with self.path.open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        return True


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--chunk-steps", type=int, default=4096)
    parser.add_argument("--stage-steps", type=int, default=32768)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--promotion-rate", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=float, default=240.0)
    parser.add_argument("--tag-penalty", type=float, default=0.5)
    parser.add_argument("--opponent-speed", type=float, default=1.0)
    parser.add_argument("--milestone-reward", type=float, default=2.0)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--start-phase", type=int, choices=PHASES, default=2)
    parser.add_argument("--variants", action="store_true")
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="Pause after this many completed chunks; zero runs the stage budgets")
    args = parser.parse_args(argv)
    if min(args.chunk_steps, args.stage_steps, args.validation_episodes, args.eval_episodes) <= 0:
        parser.error("Step and episode budgets must be positive")
    if args.chunk_steps % 256 or args.stage_steps % args.chunk_steps:
        parser.error("Chunks must be multiples of 256; stage steps must be multiples of chunk steps")
    if not 0 < args.promotion_rate <= 1 or args.horizon <= 0:
        parser.error("Promotion rate must be in (0,1] and horizon must be positive")
    if min(args.tag_penalty, args.milestone_reward) < 0:
        parser.error("Reward coefficients must be nonnegative")
    if args.max_chunks < 0:
        parser.error("Maximum chunks must be nonnegative")
    if not 0 < args.opponent_speed <= 1:
        parser.error("Opponent speed must be in (0,1]")
    if not all(math.isfinite(value) for value in
               [args.horizon, args.tag_penalty, args.milestone_reward]):
        parser.error("Horizon and reward coefficients must be finite")
    if args.evaluate_only and not args.resume:
        parser.error("Evaluation-only requires --resume")
    return args


def prepare(args):
    spec = translate(INSTRUCTION, logic_backend="dfa")
    config = ArenaConfig(
        team_size=1, opponent="chase", formula=spec["ltl"], probe_action=True,
        tag_penalty=args.tag_penalty, probe_milestone_reward=args.milestone_reward,
        horizon=args.horizon, trace_stride=0,
        capture_reward=0.0, acceptance_bonus=24.0,
        opponent_speed=args.opponent_speed,
    )
    settings = {key: value for key, value in vars(args).items()
                if key not in {"output", "resume", "evaluate_only", "max_chunks"}}
    initial_hash = None
    if args.initialize_from:
        settings["initialize_from"] = str(args.initialize_from.resolve())
        initial_hash = hashlib.sha256(args.initialize_from.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1, "settings": settings, "arena_config": asdict(config),
        "source_sha256": source_hashes(),
        "curriculum_phases": PHASES[PHASES.index(args.start_phase):],
        "validation_seed_start": 20000, "test_seed_start": 30000,
        "resume_semantics": "PPO and optimizer restored; simulator episodes restart with a derived seed",
        "selection": "Dev full-mission completions, then wins, then total furthest phase",
        "initial_checkpoint_sha256": initial_hash,
        "packages": {name: importlib.metadata.version(name)
                     for name in ["stable-baselines3", "torch", "numpy", "pyquaticus"]},
    }
    # JSON round-trip makes tuples and saved lists compare consistently.
    manifest = json.loads(json.dumps(manifest))
    output = args.output
    if args.resume:
        stored = json.loads((output / "manifest.json").read_text())
        if stored != manifest:
            raise ValueError("Resume manifest differs: use the original settings and source snapshot")
        state = json.loads((output / "state.json").read_text())
        log = output / "training_episodes.jsonl"
        if state["log_bytes"] and (not log.exists() or log.stat().st_size < state["log_bytes"]):
            raise ValueError("Training log is shorter than its checkpointed state")
        if log.exists():
            with log.open("r+b") as stream:
                stream.truncate(state["log_bytes"])
        else:
            log.touch()
        return spec, config, state
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    snapshot = output / "source_snapshot"
    for path in source_files():
        target = snapshot / path.relative_to(REPOSITORY)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    state = {
        "status": "running", "stage_index": 0, "stage_steps": 0, "total_steps": 0,
        "episode_count": 0, "log_bytes": 0, "checkpoint": None,
        "best_checkpoint": None, "best_score": [-1, -1, -1], "chunks": 0,
    }
    write_json(output / "state.json", state)
    (output / "training_episodes.jsonl").touch()
    return spec, config, state


def final_evaluation(args, config, state):
    if state["best_checkpoint"] is None:
        return
    output = args.output
    best = output / state["best_checkpoint"]
    model = PPO.load(best / "policy.zip", device="cpu")
    evaluation = output / f"test-{time.time_ns()}"
    evaluation.mkdir()
    primary_config = replace(config, opponent_speed=1.0)
    _, primary = evaluate(
        model, primary_config, args.eval_episodes, 30000, evaluation / "primary.jsonl"
    )
    report = {"selected_checkpoint": str(best), "training_status": state["status"],
              "training_opponent_speed": config.opponent_speed,
              "primary_standard_full_speed": primary}
    if args.variants:
        variants = [("standard", config.opponent_speed), ("standard", 0.8),
                    ("wide", 1.0), ("narrow", 1.0)]
        for scenario, speed in dict.fromkeys(variants):
            if (scenario, speed) == ("standard", 1.0):
                continue
            variant = replace(config, scenario=scenario, opponent_speed=speed)
            key = f"{scenario}-speed{speed}"
            _, report[key] = evaluate(
                model, variant, args.eval_episodes, 30000, evaluation / f"{key}.jsonl"
            )
    write_json(evaluation / "summary.json", report)
    write_json(output / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


def run(args):
    torch.set_num_threads(1)
    spec, config, state = prepare(args)
    if args.evaluate_only or state["status"] != "running":
        final_evaluation(args, config, state)
        return
    model = None
    vec = None
    invocation_chunks = 0
    phases = PHASES[PHASES.index(args.start_phase):]
    try:
        if args.initialize_from and not args.resume:
            initial = PPO.load(args.initialize_from, device="cpu")
            initial.set_random_seed(args.seed)
            selected = Path("selected") / "step-000000000"
            directory = args.output / selected
            save_bundle(directory, initial,
                        replace(config, curriculum_phase=0, opponent_speed=1.0), spec, {
                            "seed": args.seed, "total_steps": 0,
                            "initialized_from": str(args.initialize_from.resolve()),
                            "initial_checkpoint_sha256": hashlib.sha256(
                                args.initialize_from.read_bytes()).hexdigest(),
                            "training_opponent_speed": config.opponent_speed,
                        })
            rows, summary = evaluate(
                initial, config, args.validation_episodes, 20000,
                directory / "validation.jsonl",
            )
            state["best_score"] = list(selection_score(rows))
            state["best_checkpoint"] = str(selected)
            write_json(directory / "validation_summary.json", summary)
            write_json(args.output / "state.json", state)
            print(f"steps=0 initialized dev_full={summary['full_mission_rate']:.1%} "
                  f"captures={summary['captures']}", flush=True)
        while state["stage_index"] < len(phases):
            phase = phases[state["stage_index"]]
            stage_config = replace(config, curriculum_phase=phase)
            if vec is None:
                vec = SharedTeamVecEnv(TeamArena(stage_config))
                reset_seed = args.seed + state["chunks"] * 1009
                vec.seed(reset_seed)
                if state["checkpoint"]:
                    model = PPO.load(args.output / state["checkpoint"] / "policy.zip",
                                     env=vec, device="cpu")
                    model.set_random_seed(reset_seed)
                elif args.initialize_from:
                    model = PPO.load(args.initialize_from, env=vec, device="cpu")
                    model.set_random_seed(reset_seed)
                else:
                    model = PPO(
                        "MlpPolicy", vec, learning_rate=3e-4, n_steps=256, batch_size=64,
                        n_epochs=5, gamma=config.gamma, ent_coef=0.01,
                        policy_kwargs={"net_arch": [128, 128]}, seed=reset_seed,
                        device="cpu", verbose=0,
                    )
                model.set_logger(configure(str(args.output), ["csv"]))
                if args.chunk_steps % model.n_steps:
                    raise ValueError("Chunk budget must be a multiple of the loaded PPO rollout length")
            model.learn(total_timesteps=args.chunk_steps, reset_num_timesteps=False,
                        callback=EpisodeLog(args.output / "training_episodes.jsonl", state, phase))
            state["chunks"] += 1
            invocation_chunks += 1
            state["stage_steps"] += args.chunk_steps
            state["total_steps"] += args.chunk_steps
            checkpoint = Path("checkpoints") / f"step-{state['total_steps']:09d}-{time.time_ns()}"
            directory = args.output / checkpoint
            save_bundle(directory, model, stage_config, spec, {
                "seed": args.seed, "total_steps": state["total_steps"], "curriculum_phase": phase,
            })
            rows, summary = evaluate(
                model, config, args.validation_episodes, 20000, directory / "validation.jsonl"
            )
            rate = stage_success_rate(rows, phase)
            write_json(directory / "validation_summary.json", {**summary, "stage_success_rate": rate})
            score = selection_score(rows)
            if score > tuple(state["best_score"]):
                selected = Path("selected") / f"step-{state['total_steps']:09d}"
                save_bundle(args.output / selected, model,
                            replace(config, curriculum_phase=0, opponent_speed=1.0), spec,
                            {"seed": args.seed, "selected_from": str(checkpoint),
                             "training_opponent_speed": config.opponent_speed,
                             "validation_score": list(score)})
                state["best_score"] = list(score)
                state["best_checkpoint"] = str(selected)
            state["checkpoint"] = str(checkpoint)
            state["log_bytes"] = (args.output / "training_episodes.jsonl").stat().st_size
            print(f"steps={state['total_steps']} phase={phase} dev_stage={rate:.1%} "
                  f"dev_full={summary['full_mission_rate']:.1%} captures={summary['captures']}",
                  flush=True)
            if rate >= args.promotion_rate:
                state["stage_index"] += 1
                state["stage_steps"] = 0
                vec.close()
                vec = None
                if state["stage_index"] == len(phases):
                    state["status"] = "complete"
            elif state["stage_steps"] >= args.stage_steps:
                state["status"] = "stage_budget_exhausted"
            write_json(args.output / "state.json", state)
            if state["status"] != "running":
                break
            if args.max_chunks and invocation_chunks >= args.max_chunks:
                print("Paused at a completed checkpoint; continue with --resume.", flush=True)
                return
    except KeyboardInterrupt:
        print("Interrupted; --resume restores the last completed checkpoint.", flush=True)
        return
    finally:
        if vec is not None:
            vec.close()
    final_evaluation(args, config, state)


if __name__ == "__main__":
    run(arguments())
