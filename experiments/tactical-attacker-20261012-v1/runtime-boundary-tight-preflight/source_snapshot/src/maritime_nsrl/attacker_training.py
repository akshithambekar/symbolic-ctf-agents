"""Imitation-initialized, task-reward PPO attacker; no LTL training guidance."""
import csv
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from .experiments import EpisodeLog
from .oracle import CaptureOracle
from .research_env import ArenaConfig, TeamArena, SharedTeamVecEnv
from .runtime_shield_study import csv_rows, save
from .specifications import read_artifact


def evaluate_attacker(model, config, output, episodes, seed):
    arena = TeamArena(config)
    rows = []
    with Path(output).with_suffix(".traces.jsonl").open("w", encoding="utf-8") as traces:
        try:
            for ep in range(episodes):
                obs = arena.reset(seed + ep)
                trajectory = []
                while not arena.finished:
                    actions, _ = model.predict(obs, deterministic=True)
                    obs, _, _, info = arena.step(actions)
                    trajectory.append({"time": float(arena.game.current_time), "positions": arena.game.state["agent_position"].tolist(),
                        "has_flag": list(map(bool, arena.game.state["agent_has_flag"])), "actions": list(map(int, actions))})
                metrics = info["metrics"]; metrics.pop("trajectory", None)
                rows.append({"evaluation_seed": seed + ep, **metrics})
                traces.write(json.dumps({"evaluation_seed": seed + ep, "trajectory": trajectory}) + "\n")
        finally: arena.close()
    csv_rows(output, rows)
    result = {"episodes": len(rows), "capture_rate": float(np.mean([r["captures"] > 0 for r in rows])),
        "pickup_rate": float(np.mean([r["pickups"] > 0 for r in rows])),
        "enemy_territory_rate": float(np.mean([r["reached_enemy_territory"] for r in rows])),
        "out_of_bounds_episode_rate": float(np.mean([r["out_of_bounds_steps"] > 0 for r in rows])),
        "mean_common_reward": float(np.mean([r["common_task_reward"] for r in rows]))}
    save(Path(output).with_suffix(".summary.json"), result)
    return result


def train_attacker(root, output, seed=0, steps=4096, demonstration_episodes=24, bc_epochs=160):
    root, output = Path(root), Path(output)
    if steps < 256 or demonstration_episodes < 1 or bc_epochs < 1: raise ValueError("Invalid training budget")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(seed)
    config = ArenaConfig(team_size=1, guided=False, reward_shaping=False, temporal_state=False, shield=False,
        formula="G(in_bounds) & F(has_flag & F(captured))", opponent="static", capture_reward=25.,
        pickup_reward=8., tag_penalty=0., time_penalty=.005)
    source = root / "research_archive/phase3-confirmatory-20260927-v1/stages/02-1v1-patrol/neural/seed0/specification.json"
    spec = read_artifact(source)
    assert spec["ltl"] == config.formula
    save(output / "specification.json", spec)
    save(output / "arena_config.json", asdict(config))
    save(output / "config.json", {"mode": "team", "team_size": 1, "seed": seed, "steps": steps,
        "guided": False, "reward_shaping": False, "temporal_state": False, "shield": False,
        "formula": config.formula, "opponent": "static", "scenario": "standard", "horizon": config.horizon,
        "training_method": "oracle behavior cloning followed by task-reward PPO", "demonstration_episodes": demonstration_episodes,
        "bc_epochs": bc_epochs, "promotion_capture_rate": .6, "promotion_episodes": 10,
        "interpretation": "new exploratory attacker; not the original scratch-trained PPO baseline"})
    snapshot = output / "source_snapshot"
    shutil.copytree(root / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(root / "scripts/train_capable_attacker.py", snapshot / "train_capable_attacker.py")
    save(output / "source_hashes.json", {p.relative_to(snapshot).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in snapshot.rglob("*") if p.is_file()})
    arena, teacher = TeamArena(config), CaptureOracle(1)
    observations, actions, demonstration_rows = [], [], []
    try:
        for ep in range(demonstration_episodes):
            obs = arena.reset(seed=100000 + seed*1000 + ep)
            while not arena.finished:
                selected = teacher(arena)
                observations.append(obs[0].copy()); actions.append(int(selected[0]))
                obs, _, _, info = arena.step(selected)
            metrics = info["metrics"]; metrics.pop("trajectory", None)
            demonstration_rows.append({"demonstration_seed": 100000 + seed*1000 + ep, **metrics})
    finally: arena.close()
    csv_rows(output / "demonstration_episodes.csv", demonstration_rows)
    teacher_rate = float(np.mean([r["captures"] > 0 for r in demonstration_rows]))
    print(f"seed {seed}: teacher capture rate {teacher_rate:.3f}; {len(actions)} demonstrated decisions", flush=True)
    if teacher_rate < .6: raise RuntimeError("Oracle capability screen failed; do not train on unusable demonstrations")
    x, y = np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.int64)
    np.savez_compressed(output / "demonstrations.npz", observations=x, actions=y)
    vec = SharedTeamVecEnv(TeamArena(config))
    ppo_values = {"learning_rate": 1e-4, "n_steps": 256, "batch_size": 64, "n_epochs": 4,
        "gamma": .99, "ent_coef": .003, "policy_kwargs": {"net_arch": [128, 128]}, "seed": seed, "device": "cpu"}
    save(output / "ppo_config.json", ppo_values)
    model = PPO("MlpPolicy", vec, **ppo_values)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=.001)
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=8)
    weights = torch.tensor(np.where(counts > 0, 1 / np.sqrt(np.maximum(counts, 1)), 0), dtype=torch.float32)
    curve = []
    try:
        model.policy.set_training_mode(True)
        for epoch in range(bc_epochs):
            order = rng.permutation(len(y))
            losses = []
            for start in range(0, len(y), 256):
                indices = order[start:start+256]
                logits = model.policy.get_distribution(torch.as_tensor(x[indices])).distribution.logits
                loss = F.cross_entropy(logits, torch.as_tensor(y[indices]), weight=weights)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(float(loss.detach()))
            if epoch % 10 == 0 or epoch == bc_epochs-1:
                curve.append({"epoch": epoch + 1, "cross_entropy": float(np.mean(losses))})
        model.policy.set_training_mode(False)
        model.save(output / "bc_policy.zip")
        csv_rows(output / "bc_curve.csv", curve)
        bc = evaluate_attacker(model, config, output / "bc_evaluation.csv", 10, 760000 + seed*100)
        print(f"seed {seed}: BC held-out capture rate {bc['capture_rate']:.3f}", flush=True)
        if bc["capture_rate"] < .6:
            save(output / "status.json", {"status": "BC capability screen failed", "bc": bc})
            return None
        training = output / "training"
        model.set_logger(configure(str(training), ["csv"]))
        model.learn(total_timesteps=steps, callback=EpisodeLog(training / "episodes.jsonl"))
        model.save(output / "policy.zip")
        final = evaluate_attacker(model, config, output / "capability_evaluation.csv", 10, 770000 + seed*100)
        save(output / "training_complete.json", {"actual_agent_transitions": model.num_timesteps,
            "demonstrated_decisions": len(y), "teacher_capture_rate": teacher_rate, "bc_screen": bc, "ppo_screen": final})
        save(output / "status.json", {"status": "promoted" if final["capture_rate"] >= .6 else "not promoted", "capability": final})
        print(f"seed {seed}: PPO held-out capture rate {final['capture_rate']:.3f}", flush=True)
        return output / "policy.zip" if final["capture_rate"] >= .6 else None
    finally: vec.close()
