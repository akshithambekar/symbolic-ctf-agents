import numpy as np
from gymnasium.spaces import Discrete

from maritime_nsrl.training import MetricWriter, TrainingMetrics


class Writer:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, step):
        self.scalars[name] = (float(value), step)

    def flush(self):
        pass

    def close(self):
        pass


def test_ppo_metrics_are_renamed_and_entropy_is_positive():
    writer = Writer()
    MetricWriter(writer).write({"train/approx_kl": 0.02, "train/entropy_loss": -1.5,
                                "train/explained_variance": np.nan}, {}, 42)
    assert writer.scalars == {"policy/approx_kl": (0.02, 42), "policy/entropy": (1.5, 42)}


def test_team_metrics_count_games_and_shield_deadlocks():
    writer = Writer()
    callback = TrainingMetrics(writer, None, True, 100, 1, 10000)
    callback.model = type("Model", (), {"get_env": lambda self: type("Env", (), {
        "action_space": Discrete(8)})()})()
    callback.num_timesteps = 6
    callback.locals = {"actions": np.array([0, 1, 2]), "infos": [
        {"metrics": {"ltlf_satisfied": True, "episode_reward": 5, "decisions": 4,
                     "shield_decisions": 10, "shield_interventions": 2, "shield_deadlocks": 1}},
        {}, {},
    ]}
    assert callback._on_step()
    assert writer.scalars["train/solve_rate"] == (1, 6)
    assert writer.scalars["rollout/steps_per_success"] == (4, 6)
    assert writer.scalars["rollout/valid_action_rate"] == (1, 6)
    assert writer.scalars["shield/acceptance_rate"] == (0.7, 6)
