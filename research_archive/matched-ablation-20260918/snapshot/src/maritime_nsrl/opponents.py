"""Opponent policies and reusable low-level waypoint controller."""
import numpy as np


def steer(env, index, target, speed_fraction=1.0):
    if target is None: return np.array([0., 0.], dtype=np.float32)
    delta = np.asarray(target) - env.state["agent_position"][index]
    heading = np.degrees(np.arctan2(delta[0], delta[1]))
    error = (heading - env.state["agent_heading"][index] + 180) % 360 - 180
    speed = env.max_speeds[index] * speed_fraction if np.linalg.norm(delta) > 3 else 0.
    return np.array([speed, error], dtype=np.float32)


class Opponent:
    def __init__(self, kind):
        if kind not in {"static", "patrol", "chase"}: raise ValueError(kind)
        self.kind, self.waypoint = kind, 0

    def action(self, env, index=1, learner_index=0):
        if self.kind == "static": return steer(env, index, None)
        pos = env.state["agent_position"][index]
        target = env.state["agent_position"][learner_index]
        if self.kind == "chase" and not env.state["agent_on_sides"][learner_index]:
            return steer(env, index, target)
        points = [np.array([112., 20.]), np.array([112., 60.])]
        if np.linalg.norm(pos - points[self.waypoint]) < 6: self.waypoint = 1 - self.waypoint
        return steer(env, index, points[self.waypoint], .65)

