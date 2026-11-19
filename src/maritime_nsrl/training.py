"""Shared launcher and live metrics for standard PPO training."""

import random
import socket
import subprocess
import sys
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter


class MetricWriter(KVWriter):
    """Add stable public names for SB3's PPO diagnostics."""

    def __init__(self, writer):
        self.writer = writer

    def write(self, values, excluded, step=0):
        names = {
            "train/approx_kl": "policy/approx_kl",
            "train/clip_fraction": "policy/clip_fraction",
            "train/entropy_loss": "policy/entropy",
            "train/explained_variance": "critic/explained_variance",
        }
        for source, target in names.items():
            if source in values and np.isfinite(values[source]):
                value = float(values[source])
                value = -value if source.endswith("entropy_loss") else value
                self.writer.add_scalar(target, value, step)
        self.writer.flush()

    def close(self):
        self.writer.close()


class TrainingMetrics(BaseCallback):
    def __init__(self, writer, make_eval_env, team, eval_interval, eval_episodes, eval_seed):
        super().__init__()
        self.writer = writer
        self.make_eval_env = make_eval_env
        self.team = team
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.last_eval = 0
        self.recent = deque(maxlen=100)
        self.valid = self.actions = 0

    def _on_training_start(self):
        self.last_eval = self.num_timesteps

    def _evaluate(self):
        # The simulator uses global RNGs; evaluation must not perturb training.
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        env = None
        try:
            env = self.make_eval_env()
            successes = []
            for episode in range(self.eval_episodes):
                result = env.reset(seed=self.eval_seed + episode)
                obs = result if self.team else result[0]
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    if self.team:
                        obs, _, done, info = env.step(action)
                    else:
                        obs, _, terminated, truncated, info = env.step(int(action))
                        done = terminated or truncated
                successes.append(bool(info["metrics"]["ltlf_satisfied"]))
            self.writer.add_scalar("eval/solve_rate", np.mean(successes), self.num_timesteps)
        finally:
            if env is not None:
                env.close()
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    def _on_step(self):
        actions = np.asarray(self.locals["actions"]).reshape(-1)
        self.actions += len(actions)
        self.valid += sum(self.training_env.action_space.contains(int(action)) for action in actions)
        for info in self.locals["infos"]:
            metrics = info.get("metrics")
            if metrics is None:
                continue
            self.recent.append(metrics)
            success = [row for row in self.recent if row["ltlf_satisfied"]]
            self.writer.add_scalar("train/solve_rate", len(success) / len(self.recent), self.num_timesteps)
            rewards = [row["episode_reward"] for row in self.recent]
            self.writer.add_scalar("train/mean_reward", np.mean(rewards), self.num_timesteps)
            self.writer.add_scalar("rollout/valid_action_rate", self.valid / self.actions, self.num_timesteps)
            if success:
                step_key = "decisions" if self.team else "steps"
                steps = np.mean([row[step_key] for row in success])
                self.writer.add_scalar("rollout/steps_per_success", steps, self.num_timesteps)
            if "shield_decisions" in metrics:
                decisions = sum(r["shield_decisions"] for r in self.recent)
                if decisions:
                    accepted = sum(
                        row["shield_decisions"] - row["shield_interventions"] - row["shield_deadlocks"]
                        for row in self.recent
                    )
                    self.writer.add_scalar("shield/acceptance_rate", accepted / decisions, self.num_timesteps)
                    self.writer.add_scalar("shield/substitutions", metrics["shield_interventions"], self.num_timesteps)
                    self.writer.add_scalar("shield/deadlocks", metrics["shield_deadlocks"], self.num_timesteps)
            self.writer.flush()
        if self.num_timesteps - self.last_eval >= self.eval_interval:
            self._evaluate()
            self.last_eval = self.num_timesteps
        return True


@contextmanager
def tensorboard_server(logdir, enabled=True):
    if not enabled:
        yield
        return
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", 6006))
        except OSError:
            sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--host", "127.0.0.1",
         "--port", str(port), "--logdir", str(Path(logdir).resolve())],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    try:
        import time
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"TensorBoard failed to start: {process.stderr.read().decode(errors='replace')}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("TensorBoard did not start within 15 seconds")
        print(f"TensorBoard: http://127.0.0.1:{port}/", flush=True)
        yield
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stderr.close()


def launch_training(args):
    from .experiments import train, train_team

    options = dict(resume=args.resume, episodes=args.episodes, horizon=args.horizon, telemetry=True,
                   eval_interval=args.eval_interval, eval_episodes=args.eval_episodes,
                   eval_seed=args.eval_seed)
    with tensorboard_server(args.output, enabled=not args.no_tensorboard_server):
        if args.command == "train":
            return train(args.spec, args.output, args.guided, args.opponent, args.steps,
                         args.seed, device=args.device,
                         return_start_probability=args.return_start_probability, **options)
        return train_team(args.spec, args.output, team_size=args.team_size,
                          opponent=args.opponent, scenario=args.scenario, shield=args.shield,
                          steps=args.steps, seed=args.seed, role_constraint=args.role_constraint,
                          reward_shaping=not args.no_reward_shaping,
                          temporal_state=not args.no_temporal_state, **options)
