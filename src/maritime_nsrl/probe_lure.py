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
        arena.game.state["flag_home"][0],
    ]
    if phase == len(MILESTONES):
        return float(phase + 1)
    distance = np.linalg.norm(arena.game.state["agent_position"][0] - targets[phase])
    return phase + 1 - min(distance / np.linalg.norm(arena.size), 1.0)
