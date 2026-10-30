"""Observable milestones and shaping for a 1v1 upper-probe/lower-switch mission."""

import numpy as np
from .logic import atoms

MILESTONES = (
    "probe_upper", "pursuit_committed", "withdrawn", "attack_lower", "has_flag", "captured"
)
INSTRUCTION = (
    "Probe the upper flank, induce committed pursuit, withdraw into the upper home "
    "side, switch to the lower attack lane, pick up the flag, and capture it."
)
FORMULA = (
    "G(in_bounds) & F(probe_upper & F(pursuit_committed & F(withdrawn & "
    "F(attack_lower & F(has_flag & F(captured))))))"
)


def tactical_labels(game, size):
    blue, red = game.state["agent_position"]
    width, height = size
    labels = set()
    if game.state["agent_is_tagged"][0]:
        return labels
    if 0.50 * width <= blue[0] <= 0.64 * width and blue[1] >= 0.70 * height:
        labels.add("probe_upper")
    heading = np.radians(game.state["agent_heading"][1])
    forward = np.array([np.sin(heading), np.cos(heading)])
    approaching = np.dot(forward, blue - red) > 0
    if (
        blue[0] > 0.50 * width
        and blue[1] >= 0.65 * height
        and red[1] >= 0.65 * height
        and np.linalg.norm(red - game.state["flag_home"][1]) >= 0.15 * width
        and game.state["agent_speed"][1] > 0.1
        and not game.state["agent_is_tagged"][1]
        and approaching
    ):
        labels.add("pursuit_committed")
    if blue[0] <= 0.48 * width and blue[1] >= 0.65 * height:
        labels.add("withdrawn")
    if blue[0] >= 0.55 * width and blue[1] <= 0.30 * height:
        labels.add("attack_lower")
    return labels


def mission_phase(monitor):
    def phase(node):
        # Progression retains alternative start times; use the furthest live branch.
        if node[0] == "|":
            return max(phase(child) for child in node[1:])
        if node[0] == "&":
            return min(phase(child) for child in node[1:])
        remaining = atoms(node)
        return next(
            (i for i, name in enumerate(MILESTONES) if name in remaining), len(MILESTONES)
        )

    return phase(monitor.automaton.states[monitor.state])


def mission_potential(arena):
    if arena.monitor.impossible:
        return 0.0
    phase = mission_phase(arena.monitor)
    width, height = arena.size
    targets = [
        [0.56 * width, 0.80 * height],
        [0.56 * width, 0.80 * height],
        [0.40 * width, 0.85 * height],
        [0.58 * width, 0.15 * height],
        arena.game.state["flag_home"][1],
        [0.40 * width, 0.85 * height],
    ]
    if phase == len(MILESTONES):
        return float(phase + 1)
    position = arena.game.state["agent_position"][0]
    distance = np.linalg.norm(position - targets[phase])
    if phase == 3 and position[1] > 0.30 * height:
        transfer = np.array([0.40 * width, 0.15 * height])
        distance = np.linalg.norm(position - transfer) + np.linalg.norm(transfer - targets[phase])
    return phase + 1 - min(distance / np.linalg.norm(arena.size), 1.0)


class ProbeLureTeacher:
    """Use the nine PPO actions with observed state and temporal progress.

    Full completion is verified against a speed-0.6 defender. At full defender
    speed the probe is released only while the defender is climbing, so pursuit
    latches with a gap. Pickup is still blocked: the defender remains closer to
    the flag than the attacker. Tagging and recovery between milestones are allowed.
    """

    def __init__(self):
        self.seen_low = False
        self.press = False
        self.prev_red_y = None

    def __call__(self, arena):
        state = arena.game.state
        position = state["agent_position"][0]
        red = state["agent_position"][1]
        width, height = arena.size
        phase = mission_phase(arena.monitor)
        # A reused teacher must not keep the previous episode's release latch.
        if arena.game.current_time <= 0:
            self.seen_low = False
            self.press = False
            self.prev_red_y = None
        if self.prev_red_y is not None and red[1] < 0.40 * height:
            self.seen_low = True
        climbing = self.prev_red_y is not None and red[1] > self.prev_red_y + 2
        self.prev_red_y = float(red[1])
        full_speed = arena.config.opponent_speed >= 0.9
        if full_speed and self.seen_low and climbing and red[1] >= 0.45 * height:
            self.press = True
        if state["agent_is_tagged"][0]:
            action = 1
        elif phase <= 1 and full_speed and not self.press:
            action = 5
        elif phase <= 1:
            action = 5 if full_speed and phase == 1 and np.linalg.norm(position - red) < 24 else 8
        elif phase == 2:
            action = 5
        elif phase == 3:
            action = 4 if position[1] > 0.20 * height else 2
        elif phase == 4:
            action = 2 if position[0] < (135 / 160) * width else 0
        elif phase == 5:
            action = 5
        else:
            action = 7
        return np.asarray([action], dtype=int)
