"""Grounded reward machine, separated from the learning algorithm."""
import numpy as np
from .logic import Monitor


def labels_from_state(env, agent_index=0, team_index=0, previous_captures=0):
    state = env.state
    labels = set()
    if not state["agent_oob"][agent_index]: labels.add("in_bounds")
    if state["agent_is_tagged"][agent_index]: labels.add("tagged")
    if state["agent_has_flag"][agent_index]: labels.add("has_flag")
    if state["captures"][team_index] > previous_captures: labels.add("captured")
    if np.linalg.norm(state["agent_position"][agent_index] - state["flag_home"][team_index]) <= env.catch_radius:
        labels.add("home")
    return labels


class Guidance:
    def __init__(self, spec, gamma=.99, potential_scale=5., milestone_reward=2., violation_penalty=1.):
        self.monitor = Monitor(spec)
        self.gamma, self.scale = gamma, potential_scale
        self.milestone_reward, self.violation_penalty = milestone_reward, violation_penalty

    def potential(self, env):
        goal = self.monitor.active_goal
        if goal is None: return float(self.monitor.stage)
        state = env.state
        if goal in {"has_flag", "captured"}:
            target = state["flag_home"][0 if state["agent_has_flag"][0] else 1]
        elif goal == "home": target = state["flag_home"][0]
        else: return float(self.monitor.stage)
        distance = np.linalg.norm(state["agent_position"][0] - target)
        return self.monitor.stage + 1. - min(distance / np.linalg.norm(env.env_size), 1.)

    def reward(self, previous_potential, next_potential, milestones, violated):
        return (self.scale * (self.gamma * next_potential - previous_potential)
                + self.milestone_reward * milestones - self.violation_penalty * int(violated))

