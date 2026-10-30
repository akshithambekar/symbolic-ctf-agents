"""Reproducible PPO training, paired evaluation, and raw experiment records."""

import csv
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from .environment import MaritimeEnv
from .translation import load_spec
from .research_env import TeamArena, ArenaConfig, SharedTeamVecEnv


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


class EpisodeLog(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def _on_step(self):
        if self.num_timesteps % 2048 == 0:
            print(f"{self.path.parent.name}: {self.num_timesteps} training decisions", flush=True)
        for info in self.locals["infos"]:
            if "metrics" in info:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"timesteps": self.num_timesteps, **info["metrics"]}) + "\n")
        return True


def train(
    spec_path,
    output,
    guided=False,
    opponent="static",
    steps=16384,
    seed=0,
    horizon=240.0,
    resume=None,
    return_start_probability=0.0,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    spec = load_spec(spec_path)
    torch.set_num_threads(1)
    gamma = 0.99
    env = MaritimeEnv(
        spec,
        guided=guided,
        opponent=opponent,
        horizon=horizon,
        gamma=gamma,
        return_start_probability=return_start_probability,
    )
    config = dict(
        guided=guided,
        opponent=opponent,
        steps=steps,
        seed=seed,
        horizon=horizon,
        gamma=gamma,
        repeat=5,
        dynamics="heron",
        formula=spec.formula,
        resume=str(resume) if resume else None,
        return_start_probability=return_start_probability,
    )
    versions = {
        name: importlib.metadata.version(name)
        for name in ["pyquaticus", "stable-baselines3", "torch", "numpy", "gymnasium", "lark"]
    }
    save_json(
        output / "config.json",
        {
            **config,
            "versions": versions,
            "python": platform.python_version(),
            "pyquaticus_commit": "475d670650ec5d49691ba707cd1054c4e45d8053",
        },
    )
    save_json(
        output / "specification.json", json.loads(Path(spec_path).read_text(encoding="utf-8"))
    )
    try:
        if resume:
            old_config = json.loads((Path(resume).parent / "config.json").read_text())
            for key in ("guided", "formula", "gamma", "repeat", "dynamics", "horizon"):
                if old_config[key] != config[key]:
                    raise ValueError(f"Resume configuration mismatch: {key}")
            model = PPO.load(resume, env=env, device="cpu")
            model.set_random_seed(seed)
        else:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=64,
                n_epochs=5,
                gamma=gamma,
                ent_coef=0.01,
                policy_kwargs={"net_arch": [64, 64]},
                seed=seed,
                device="cpu",
                verbose=0,
            )
        model.set_logger(configure(str(output), ["csv"]))
        model.learn(total_timesteps=steps, callback=EpisodeLog(output / "episodes.jsonl"))
        model.save(output / "policy.zip")
        save_json(output / "training_complete.json", {"actual_timesteps": model.num_timesteps})
    finally:
        env.close()
    return output / "policy.zip"


def evaluate(
    policy,
    output,
    episodes=10,
    seed=10000,
    opponents=("static", "patrol", "chase"),
    guidance_override=None,
):
    policy, output = Path(policy), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
    config = json.loads((policy.parent / "config.json").read_text(encoding="utf-8"))
    spec = load_spec(policy.parent / "specification.json")
    guided = config["guided"] if guidance_override is None else guidance_override
    torch.set_num_threads(1)
    model = PPO.load(policy, device="cpu")
    rows, traces = [], {}
    for opponent in opponents:
        env = MaritimeEnv(
            spec,
            guided=guided,
            opponent=opponent,
            horizon=config["horizon"],
            gamma=config["gamma"],
            repeat=config["repeat"],
            dynamics=config["dynamics"],
        )
        try:
            for ep in range(episodes):
                obs, _ = env.reset(seed=seed + ep)
                trace = []
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = env.step(int(action))
                    if ep == 0:
                        trace.append(
                            {
                                "time": float(env.game.current_time),
                                "positions": env.game.state["agent_position"].tolist(),
                                "has_flag": bool(env.game.state["agent_has_flag"][0]),
                                "tagged": bool(env.game.state["agent_is_tagged"][0]),
                                "action": int(action),
                                "stage": env.guidance.monitor.stage,
                            }
                        )
                    done = term or trunc
                rows.append(
                    {
                        "training_seed": config["seed"],
                        "evaluation_seed": seed + ep,
                        "guided": guided,
                        **info["metrics"],
                    }
                )
                if ep == 0:
                    traces[opponent] = trace
        finally:
            env.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for opponent in opponents:
        group = [r for r in rows if r["opponent"] == opponent]
        times = [r["capture_time"] for r in group if r["capture_time"] is not None]
        n = len(group)
        successes = sum(r["flag_captures"] > 0 for r in group)
        p, z = successes / n, 1.96
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        summaries.append(
            {
                "opponent": opponent,
                "episodes": n,
                "capture_rate": p,
                "capture_rate_wilson95": [max(0.0, center - margin), min(1.0, center + margin)],
                "mean_capture_time_successes_only": float(np.mean(times)) if times else None,
                **{
                    f"mean_{k}": float(np.mean([r[k] for r in group]))
                    for k in [
                        "base_reward",
                        "episode_reward",
                        "tags_received",
                        "rule_violation_steps",
                        "ltlf_satisfied",
                    ]
                },
            }
        )
    save_json(
        output.with_suffix(".summary.json"),
        {
            "policy": str(policy),
            "evaluation_seed_start": seed,
            "summaries": summaries,
            "note": "Wilson intervals describe episodes for this fixed policy, not uncertainty across training seeds.",
        },
    )
    save_json(output.with_suffix(".traces.json"), {"evaluation_seed": seed, "traces": traces})
    plot_traces(traces, output.with_suffix(".trajectories.png"))
    return summaries


def plot_traces(traces, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(traces), figsize=(5 * len(traces), 3.5), squeeze=False)
    for ax, (opponent, trace) in zip(axes[0], traces.items()):
        positions = np.asarray([s["positions"] for s in trace])
        ax.plot(positions[:, 0, 0], positions[:, 0, 1], color="#1765ae", label="learner")
        ax.plot(positions[:, 1, 0], positions[:, 1, 1], color="#bd3232", label="opponent")
        ax.scatter([20, 140], [40, 40], c=["#1765ae", "#bd3232"], marker="P", s=90, label="flags")
        ax.scatter(
            positions[0, :, 0], positions[0, :, 1], c=["#1765ae", "#bd3232"], marker="o", s=25
        )
        ax.scatter(
            positions[-1, :, 0], positions[-1, :, 1], c=["#1765ae", "#bd3232"], marker="x", s=50
        )
        carrying = np.array([s["has_flag"] for s in trace])
        ax.scatter(
            positions[carrying, 0, 0],
            positions[carrying, 0, 1],
            color="#c19000",
            s=10,
            label="carrying flag",
        )
        ax.axvline(80, color="gray", linestyle="--", linewidth=0.7)
        ax.set(
            xlim=(0, 160),
            ylim=(0, 80),
            title=opponent,
            xlabel="meters",
            ylabel="meters",
            aspect="equal",
        )
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Learned policy: first held-out episode (circle = first sample, x = finish)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_runs(run_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(run_root)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for file in sorted(root.glob("*/episodes.jsonl")):
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        if not rows:
            continue
        window = min(20, len(rows))
        averages = np.convolve(
            [r["base_reward"] for r in rows], np.ones(window) / window, mode="valid"
        )
        axes[0].plot([r["timesteps"] for r in rows][window - 1 :], averages, label=file.parent.name)
    axes[0].set(
        xlabel="Training decisions",
        ylabel="Common task reward (rolling mean)",
        title="Learning progress",
    )
    axes[0].legend(fontsize=8)
    files = sorted(root.glob("*/evaluation.summary.json"))
    for i, file in enumerate(files):
        records = json.loads(file.read_text())["summaries"]
        width = 0.8 / max(len(files), 1)
        rates = np.array([r["capture_rate"] for r in records])
        bounds = np.array([r["capture_rate_wilson95"] for r in records])
        errors = np.maximum(0, np.stack([rates - bounds[:, 0], bounds[:, 1] - rates]))
        axes[1].bar(
            np.arange(3) + i * width, rates, width, label=file.parent.name, yerr=errors, capsize=3
        )
    offset = 0.4 - 0.4 / max(len(files), 1)
    axes[1].set(
        xticks=np.arange(3) + offset,
        xticklabels=["static", "patrol", "chase"],
        ylim=(0, 1.05),
        ylabel="Capture rate",
        title="Held-out episode seeds",
    )
    if files:
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "Evaluation pending", transform=axes[1].transAxes, ha="center")
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)


def train_team(
    spec_path,
    output,
    team_size=2,
    opponent="chase",
    scenario="standard",
    shield=False,
    steps=16384,
    seed=0,
    horizon=240.0,
    formula=None,
    reward_shaping=True,
    temporal_state=True,
    role_constraint=False,
    config_overrides=None,
    resume=None,
):
    """Train one shared PPO policy across a correlated team of blue agents."""
    from .specifications import read_artifact, compose, ROLE_CONSTRAINTS

    spec_data = read_artifact(spec_path)
    if role_constraint:
        spec_data = compose(spec_data, ROLE_CONSTRAINTS)
    ltl = spec_data["ltl"]
    if formula is not None and formula != ltl:
        raise ValueError(
            "Formula overrides break provenance; compose explicit human constraints instead"
        )
    values = dict(
        team_size=team_size,
        formula=ltl,
        opponent=opponent,
        scenario=scenario,
        shield=shield,
        horizon=horizon,
        guided=(reward_shaping or temporal_state),
        reward_shaping=reward_shaping,
        temporal_state=temporal_state,
    )
    overlap = set(values) & set(config_overrides or {})
    if overlap:
        raise ValueError(f"Config overrides duplicate fixed training fields: {sorted(overlap)}")
    if resume:
        resume = Path(resume)
        previous = json.loads((resume.parent / "config.json").read_text(encoding="utf-8"))
        previous_spec = read_artifact(resume.parent / "specification.json")
        if previous.get("mode") != "team" or previous.get("formula") != previous_spec["ltl"]:
            raise ValueError("Resume checkpoint has invalid team specification provenance")
        previous_values = json.loads(
            (resume.parent / "arena_config.json").read_text(encoding="utf-8")
        )
        previous_config = ArenaConfig(**previous_values)
        if (
            previous.get("team_size") != previous_config.team_size
            or previous_spec["ltl"] != previous_config.formula
        ):
            raise ValueError("Resume checkpoint configuration disagrees with its saved arena")
        for key in (
            "team_size", "formula", "scenario", "shield", "horizon",
            "reward_shaping", "temporal_state",
        ):
            if getattr(previous_config, key) != values[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        if config_overrides:
            raise ValueError("Resume uses the saved arena configuration; overrides are not supported")
        config = ArenaConfig(**{**asdict(previous_config), "opponent": opponent})
    else:
        config = ArenaConfig(**values, **(config_overrides or {}))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arena = TeamArena(config)
    vec = SharedTeamVecEnv(arena)
    torch.set_num_threads(1)
    try:
        if resume:
            model = PPO.load(resume, env=vec, device="cpu")
            model.set_random_seed(seed)
        else:
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
                seed=seed,
                device="cpu",
                verbose=0,
            )
    except Exception:
        vec.close()
        raise
    model.set_logger(configure(str(output), ["csv"]))
    save_json(
        output / "config.json",
        {
            "mode": "team",
            "team_size": team_size,
            "opponent": opponent,
            "scenario": scenario,
            "shield": shield,
            "steps": steps,
            "reward_shaping": reward_shaping,
            "temporal_state": temporal_state,
            "seed": seed,
            "horizon": horizon,
            "formula": ltl,
            "resume": str(resume) if resume else None,
            "resume_sha256": hashlib.sha256(resume.read_bytes()).hexdigest() if resume else None,
            "pyquaticus_commit": "475d670650ec5d49691ba707cd1054c4e45d8053",
        },
    )
    save_json(output / "specification.json", spec_data)
    save_json(output / "arena_config.json", asdict(config))
    ppo_config = (
        json.loads((resume.parent / "ppo_config.json").read_text(encoding="utf-8"))
        if resume
        else {
            "algorithm": "shared-policy PPO",
            "learning_rate": 3e-4,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 5,
            "gamma": config.gamma,
            "ent_coef": 0.01,
            "net_arch": [128, 128],
            "device": "cpu",
        }
    )
    save_json(
        output / "ppo_config.json",
        {**ppo_config, "seed": seed, "requested_agent_transitions": steps},
    )
    try:
        model.learn(
            total_timesteps=steps, callback=EpisodeLog(output / "training_episode_metrics.jsonl")
        )
        model.save(output / "policy.zip")
        save_json(
            output / "training_budget.json",
            {
                "actual_agent_transitions": model.num_timesteps,
                "actual_joint_decisions": model.num_timesteps // team_size,
            },
        )
    finally:
        vec.close()
    return output / "policy.zip"


def evaluate_team(
    policy,
    output,
    episodes=10,
    seed=10000,
    opponents=("static", "patrol", "chase"),
    scenarios=("standard", "wide", "narrow"),
    trace_output=None,
    trace_stride=0,
):
    policy, output = Path(policy), Path(output)
    if output.exists():
        raise FileExistsError(output)
    cfg = json.loads((policy.parent / "config.json").read_text())
    from .specifications import read_artifact

    spec = read_artifact(policy.parent / "specification.json")
    if spec["ltl"] != cfg["formula"]:
        raise ValueError("Saved training formula differs from specification; provenance mismatch")
    model = PPO.load(policy, device="cpu")
    rows = []
    traces = []
    arena_path = policy.parent / "arena_config.json"
    arena_values = (
        json.loads(arena_path.read_text())
        if arena_path.exists()
        else {
            "team_size": cfg["team_size"],
            "formula": cfg["formula"],
            "shield": cfg["shield"],
            "horizon": cfg["horizon"],
            "guided": cfg.get(
                "guided", cfg.get("reward_shaping", True) or cfg.get("temporal_state", True)
            ),
            "reward_shaping": cfg.get("reward_shaping", True),
            "temporal_state": cfg.get("temporal_state", True),
        }
    )
    for scenario in scenarios:
        for opponent in opponents:
            values = {
                **arena_values,
                "opponent": opponent,
                "scenario": scenario,
                "trace_stride": trace_stride,
            }
            arena = TeamArena(ArenaConfig(**values))
            try:
                for ep in range(episodes):
                    obs = arena.reset(seed=seed + ep)
                    done = False
                    while not done:
                        actions, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, info = arena.step(actions)
                    metrics = info["metrics"]
                    trajectory = metrics.pop("trajectory", None)
                    rows.append({"evaluation_seed": seed + ep, **metrics})
                    if trajectory is not None:
                        traces.append(
                            {
                                "evaluation_seed": seed + ep,
                                "scenario": scenario,
                                "opponent": opponent,
                                "trajectory": trajectory,
                            }
                        )
            finally:
                arena.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = []
    for scenario in scenarios:
        for opponent in opponents:
            group = [r for r in rows if r["scenario"] == scenario and r["opponent"] == opponent]
            grouped.append(
                {
                    "scenario": scenario,
                    "opponent": opponent,
                    "episodes": len(group),
                    "capture_rate": float(np.mean([r["captures"] > 0 for r in group])),
                    "mean_reward": float(np.mean([r["episode_reward"] for r in group])),
                    "mean_tags_received": float(np.mean([r["tags_received"] for r in group])),
                    "mean_shield_interventions": float(
                        np.mean([r["shield_interventions"] for r in group])
                    ),
                    "mean_shield_deadlocks": float(np.mean([r["shield_deadlocks"] for r in group])),
                }
            )
    save_json(output.with_suffix(".summary.json"), {"policy": str(policy), "summaries": grouped})
    if trace_output is not None:
        trace_output = Path(trace_output)
        trace_output.parent.mkdir(parents=True, exist_ok=True)
        with trace_output.open("w", encoding="utf-8") as f:
            for record in traces:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
    return grouped
