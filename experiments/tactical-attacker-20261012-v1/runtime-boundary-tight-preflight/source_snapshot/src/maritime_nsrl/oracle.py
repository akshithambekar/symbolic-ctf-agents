"""Transparent waypoint policies for environment and action-space validation."""
import numpy as np


class CaptureOracle:
    """A deterministic policy using the same eight discrete actions as PPO.

    In 3v3 both attackers approach together. This deliberately exploits the
    defender's tagging cooldown: one may be tagged while the other gets the flag.
    Agent 0 remains home. It is a feasibility probe, not an optimal opponent.
    """
    def __init__(self, team_size):
        self.team_size = team_size

    def __call__(self, arena):
        s, n = arena.game.state, self.team_size
        actions = [7] * n
        attackers = range(n) if n == 1 else range(1, n)
        for i in attackers:
            if s["agent_has_flag"][i]:
                # Use separated return lanes until safely across midfield.
                if s["agent_position"][i, 0] > .42 * arena.size[0]:
                    actions[i] = 4 if i % 2 else 5
                else:
                    actions[i] = 1
            elif n == 1:
                # The low flank avoids the single static defender's start.
                actions[i] = 2
            else:
                actions[i] = 0
        return np.asarray(actions, dtype=int)


def run_oracle(config, episodes=5, seed=30000, trace_stride=1):
    """Evaluate the oracle and retain episode metrics plus decision traces."""
    from dataclasses import replace
    from .research_env import TeamArena
    config = replace(config, guided=False, reward_shaping=False,
                     temporal_state=False, shield=False, trace_stride=trace_stride)
    policy, rows = CaptureOracle(config.team_size), []
    arena = TeamArena(config)
    try:
        for episode in range(episodes):
            arena.reset(seed + episode)
            while not arena.finished:
                _, _, _, info = arena.step(policy(arena))
            rows.append({"evaluation_seed": seed + episode, **info["metrics"]})
    finally:
        arena.close()
    return rows
