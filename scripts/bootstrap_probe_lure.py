"""Initialize PPO by imitating successful probe-lure demonstrations.

This is imitation-assisted initialization, not PPO training or evidence of
full-speed performance. Validation uses seeds 20000+; final test seeds are unused.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from stable_baselines3 import PPO

from maritime_nsrl.probe_lure import INSTRUCTION, ProbeLureTeacher
from maritime_nsrl.research_env import ArenaConfig, SharedTeamVecEnv, TeamArena
from maritime_nsrl.translation import translate
from train_probe_lure_full import (
    REPOSITORY, evaluate, save_bundle, selection_score, source_hashes, write_json,
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demonstrations", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--opponent-speed", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if min(args.demonstrations, args.max_attempts, args.epochs, args.validation_episodes) <= 0:
        parser.error("Demonstration, epoch, attempt, and episode budgets must be positive")
    if args.max_attempts < args.demonstrations:
        parser.error("Maximum attempts must cover the requested successful demonstrations")
    if args.max_attempts > 19000 or args.validation_episodes > 10000:
        parser.error("Budgets must preserve the disjoint demonstration, dev, and final-test seed ranges")
    if args.epochs > 200:
        parser.error("This bootstrap trial permits at most 200 supervised epochs")
    if not 0 < args.opponent_speed <= 1:
        parser.error("Opponent speed must be in (0,1]")
    return args


def collect_demonstrations(config, output, count, max_attempts):
    teacher = ProbeLureTeacher()
    arena = TeamArena(config)
    observation_size = arena.observation_space.shape[0]
    observations, actions, successful_seeds = [], [], []
    attempts = 0
    try:
        with (output / "teacher_episodes.jsonl").open("w") as stream:
            for seed in range(1000, 1000 + max_attempts):
                attempts += 1
                obs = arena.reset(seed)
                episode_observations, episode_actions = [], []
                while not arena.finished:
                    action = teacher(arena)
                    episode_observations.append(obs[0].copy())
                    episode_actions.append(int(action[0]))
                    obs, _, _, info = arena.step(action)
                metrics = info["metrics"]
                success = metrics["captures"] > 0 and metrics["ltlf_satisfied"]
                stream.write(json.dumps({
                    "seed": seed, "retained": bool(success), **metrics,
                }, allow_nan=False) + "\n")
                stream.flush()
                if success:
                    observations.extend(episode_observations)
                    actions.extend(episode_actions)
                    successful_seeds.append(seed)
                print(f"Teacher attempts={attempts} successful={len(successful_seeds)}/{count}",
                      flush=True)
                if len(successful_seeds) == count:
                    break
    finally:
        arena.close()
    features = np.asarray(observations, dtype=np.float32).reshape(-1, observation_size)
    labels = np.asarray(actions, dtype=np.int64)
    np.savez_compressed(output / "demonstrations.npz", observations=features, actions=labels,
                        episode_seeds=np.asarray(successful_seeds, dtype=np.int64))
    return features, labels, successful_seeds, attempts


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    spec = translate(INSTRUCTION, logic_backend="dfa")
    config = ArenaConfig(
        team_size=1, formula=spec["ltl"], opponent="chase", opponent_speed=args.opponent_speed,
        horizon=240, probe_action=True, curriculum_phase=0, tag_penalty=0.5,
        capture_reward=0, acceptance_bonus=24, probe_milestone_reward=2,
    )
    hashes = source_hashes()
    relative_script = str(Path(__file__).resolve().relative_to(REPOSITORY))
    hashes[relative_script] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1, "status": "collecting_demonstrations",
        "method": "imitation-assisted PPO initialization; no PPO updates in this script",
        "teacher_scope": "Teacher verified only against a slowed speed-0.6 defender; full-speed success is unproven",
        "settings": {**vars(args), "output": str(args.output.resolve())},
        "arena_config": asdict(config), "source_sha256": hashes,
        "demonstration_seed_start": 1000, "validation_seed_start": 20000,
        "final_test_seeds_used": False,
        "optimizer": {"learning_rate": 3e-4, "batch_size": 256,
                      "objective": "mean negative teacher-action log probability"},
        "ppo": {"n_steps": 256, "batch_size": 64, "n_epochs": 5,
                "gamma": config.gamma, "ent_coef": 0.01, "net_arch": [128, 128]},
        "packages": {name: importlib.metadata.version(name)
                     for name in ["stable-baselines3", "torch", "numpy", "pyquaticus"]},
    }
    write_json(args.output / "manifest.json", manifest)
    for relative in hashes:
        source = REPOSITORY / relative
        target = args.output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(args.output / "specification.json", spec)
    write_json(args.output / "arena_config.json", asdict(config))
    obs, actions, seeds, attempts = collect_demonstrations(
        config, args.output, args.demonstrations, args.max_attempts
    )
    manifest.update({
        "teacher_attempts": attempts, "successful_demonstrations": len(seeds),
        "successful_demonstration_seeds": seeds, "training_samples": len(actions),
        "dataset_sha256": hashlib.sha256((args.output / "demonstrations.npz").read_bytes()).hexdigest(),
    })
    if len(seeds) < args.demonstrations:
        manifest["status"] = "insufficient_successful_demonstrations"
        write_json(args.output / "manifest.json", manifest)
        raise RuntimeError(f"Collected {len(seeds)}/{args.demonstrations} successful demonstrations "
                           f"within {args.max_attempts} attempts; partial dataset preserved")
    manifest["status"] = "cloning"
    write_json(args.output / "manifest.json", manifest)
    vec = SharedTeamVecEnv(TeamArena(config))
    vec.seed(args.seed)
    try:
        model = PPO(
            "MlpPolicy", vec, learning_rate=3e-4, n_steps=256, batch_size=64,
            n_epochs=5, gamma=config.gamma, ent_coef=0.01,
            policy_kwargs={"net_arch": [128, 128]}, seed=args.seed,
            device="cpu", verbose=0,
        )
        features = torch.as_tensor(obs, dtype=torch.float32)
        labels = torch.as_tensor(actions, dtype=torch.long)
        best_score = None
        best_checkpoint = None
        records = []
        for epoch in range(1, args.epochs + 1):
            model.policy.set_training_mode(True)
            order = torch.randperm(len(labels))
            total_loss = 0.0
            for batch in order.split(256):
                distribution = model.policy.get_distribution(features[batch])
                loss = -distribution.log_prob(labels[batch]).mean()
                model.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                model.policy.optimizer.step()
                total_loss += float(loss.detach()) * len(batch)
            mean_loss = total_loss / len(labels)
            if epoch % 20 == 0 or epoch == args.epochs:
                directory = args.output / f"epoch-{epoch:04d}"
                save_bundle(directory, model, config, spec, {
                    "method": "imitation-assisted PPO initialization", "epoch": epoch,
                    "seed": args.seed, "teacher_opponent_speed": args.opponent_speed,
                    "teacher_scope": manifest["teacher_scope"], "ppo_environment_steps": 0,
                })
                rows, summary = evaluate(model, config, args.validation_episodes, 20000,
                                         directory / "validation.jsonl")
                score = selection_score(rows)
                if best_score is None or score > best_score:
                    best_score, best_checkpoint = score, directory
                record = {"epoch": epoch, "mean_action_loss": mean_loss,
                          "selection_score": list(score), **summary}
                records.append(record)
                write_json(directory / "validation_summary.json", record)
                write_json(args.output / "validation_history.json", records)
                print(f"epoch={epoch} loss={mean_loss:.4f} dev_full={summary['full_mission_rate']:.1%} "
                      f"captures={summary['captures']}", flush=True)
        selected = args.output / "selected"
        selected.mkdir()
        for name in ["policy.zip", "arena_config.json", "specification.json", "config.json",
                     "validation.jsonl", "validation_summary.json"]:
            shutil.copyfile(best_checkpoint / name, selected / name)
        manifest.update({
            "status": "complete", "selected_checkpoint": str(best_checkpoint.relative_to(args.output)),
            "selected_score": list(best_score),
            "model_sha256": hashlib.sha256((selected / "policy.zip").read_bytes()).hexdigest(),
            "ppo_environment_steps": 0,
        })
        write_json(args.output / "manifest.json", manifest)
        write_json(args.output / "summary.json", {
            "method": manifest["method"], "teacher_attempts": attempts,
            "successful_demonstrations": len(seeds), "training_samples": len(actions),
            "selected_checkpoint": str(selected),
            "validation": json.loads((selected / "validation_summary.json").read_text()),
            "final_test_seeds_used": False,
        })
    finally:
        vec.close()


if __name__ == "__main__":
    run(arguments())
