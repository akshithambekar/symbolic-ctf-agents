import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env
from maritime_nsrl.environment import MaritimeEnv
from maritime_nsrl.logic import validate

SPEC = validate("G(in_bounds) & F(has_flag & F(captured))")


def test_gym_contract_and_reproducible_reset():
    env = MaritimeEnv(SPEC, horizon=10)
    try:
        check_env(env, warn=True)
        a, _ = env.reset(seed=33)
        first = env.step(0)[0]
        b, _ = env.reset(seed=33)
        second = env.step(0)[0]
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(first, second)
    finally: env.close()


def test_real_pyquaticus_pickup_and_capture():
    # Controller integration check, deliberately NOT called an RL result.
    env = MaritimeEnv(SPEC, guided=True)
    try:
        env.reset(seed=9)
        done = False
        while not done:
            action = 1 if env.game.state["agent_has_flag"][0] else 0
            _, _, done, _, info = env.step(action)
        assert info["metrics"]["flag_captures"] == 1
        assert info["metrics"]["flag_pickups"] == 1
        assert info["metrics"]["capture_time"] > 0
        assert info["metrics"]["ltlf_satisfied"]
        with pytest.raises(RuntimeError): env.step(0)
    finally: env.close()


@pytest.mark.parametrize("opponent", ["static","patrol","chase"])
def test_opponents_finish_and_baseline_has_no_guidance(opponent):
    env = MaritimeEnv(SPEC, opponent=opponent, horizon=10)
    try:
        env.reset(seed=1)
        done = False
        while not done:
            _, reward, done, trunc, info = env.step(0)
            assert np.isfinite(reward) and not trunc
        assert info["environment_time_limit"]
        assert info["metrics"]["guidance_reward"] == 0
        assert not info["metrics"]["mission_completed"]
    finally: env.close()


def test_chasing_tags_are_grounded_as_persistent_safety_failure():
    spec = validate("G(!tagged) & F(has_flag & F(captured))")
    env = MaritimeEnv(spec, guided=True, opponent="chase")
    try:
        env.reset(seed=1)
        done = False
        while not done:
            _, _, done, _, info = env.step(0)
        m = info["metrics"]
        assert m["tags_received"] > 0 and m["rule_violation_steps"] > 0
        assert not m["ltlf_satisfied"]
    finally: env.close()


def test_return_curriculum_initializes_real_flag_and_can_score():
    env = MaritimeEnv(SPEC, guided=True, return_start_probability=1.)
    try:
        _, info = env.reset(seed=3)
        assert info["curriculum_return_start"]
        assert env.game.state["agent_has_flag"][0]
        assert env.guidance.monitor.stage == 1
        done = False
        while not done:
            _, _, done, _, info = env.step(1)
        assert info["metrics"]["flag_captures"] == 1
        assert info["metrics"]["flag_pickups"] == 0
        assert info["metrics"]["ltlf_satisfied"]
    finally: env.close()
