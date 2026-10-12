"""Additive exploratory exposure distributions and recovery controls.

These classes are opt-in. Original arenas, policy actions and completed studies
are unchanged. Carrier starts test return control, not learned flag acquisition.
"""
import copy
from dataclasses import dataclass
import numpy as np
from .research_env import TeamArena
from .runtime_shield_study import AuditedShield
from .temporal import TemporalMonitor


@dataclass
class Exposure:
    stage: str = "ordinary"
    margin: float = 4.
    speed: float = 1.
    edge: str = "lower"

    def __post_init__(self):
        if self.stage not in {"ordinary", "approach", "return"}: raise ValueError("Unknown exposure stage")
        if self.edge not in {"lower", "upper"}: raise ValueError("Unknown edge")
        if not np.isfinite(self.margin) or self.margin < 2.: raise ValueError("Initial footprint must fit")
        if not np.isfinite(self.speed) or not 0 <= self.speed <= 2.: raise ValueError("Invalid initial speed")


def native_restart(arena, init):
    """Restart with simulator-supported initial conditions, then reset monitoring."""
    init = dict(init)
    if "agent_has_flag" in init:
        # Upstream indexes this field with NumPy team arrays, so use an array
        # even though its documented reset interface also accepts lists.
        init["agent_has_flag"] = np.asarray(init["agent_has_flag"], dtype=bool)
    for key, controllers in arena.controllers.items():
        arena.game.players[key]._pid_controllers = copy.deepcopy(controllers)
    arena.game.reset(seed=int(arena.rng.integers(0, 2**31-1)), options={"init_dict": init})
    arena.monitor = TemporalMonitor(arena.automaton)
    arena.monitor.update(arena.labels())
    arena.ever_flag = bool(any(arena.game.state["agent_has_flag"][:arena.n]))
    if arena.gate: arena.gate.initialize(arena.labels())
    arena.last_valid_observation = arena.observe()
    return arena.last_valid_observation


class ExposureArena(TeamArena):
    def __init__(self, config, exposure=None, recovery=False):
        if config.team_size != 1: raise ValueError("Offensive exposure study is 1v1")
        self.exposure = exposure or Exposure()
        self.recovery = recovery
        super().__init__(config)

    def reset(self, seed=None):
        obs = super().reset(seed)
        e = self.exposure
        if e.stage != "ordinary":
            w, h = self.size
            if e.margin >= h/2: raise ValueError("Margin exceeds arena interior")
            lower = e.edge == "lower"
            positions = self.game.state["agent_position"].tolist()
            positions[0] = [.78*w if e.stage == "approach" else .90*w, e.margin if lower else h-e.margin]
            obs = native_restart(self, {"agent_position": positions,
                "agent_heading": [180. if lower else 0., 180.], "agent_speed": [e.speed, 0.],
                "agent_has_flag": [e.stage == "return", False]})
        if self.config.shield: self.gate = RecoveryShield(self) if self.recovery == "proactive" else AuditedShield(self)
        return obs

    def candidates(self, actions, opponents):
        yield from super().candidates(actions, opponents)
        if self.recovery:
            # Emergency controls are shield substitutions, not learned actions.
            # Label them as hold (7) for legacy waypoint counts; raw controls are audited.
            for turn in (-90., 90., 180.):
                yield (7,), {**opponents, "agent_0": np.array([0., turn], dtype=np.float32)}


class RecoveryShield(AuditedShield):
    """One-step gate with proactive velocity reserve and extra brake-turn controls.

    The reserve is a heuristic, not a certified stopping-distance calculation.
    It preserves transactional successor commits and fails closed if infeasible.
    """
    def transition(self, game, candidates, ground):
        def reserve_labels(preview):
            labels = set(ground(preview))
            s = preview.state
            pos = s["agent_position"][:self.arena.n]
            theta = np.radians(s["agent_heading"][:self.arena.n])
            velocity = np.column_stack([np.sin(theta), np.cos(theta)])*s["agent_speed"][:self.arena.n, None]
            # Two-second outward-velocity projection, with center bounds matching
            # the original grounded proposition (no new physics or state edits).
            projected = pos + 2.*velocity
            if np.any(projected < 0.) or np.any(projected > self.arena.size): labels.discard("in_bounds")
            return labels
        return super().transition(game, candidates, reserve_labels)


class ChaseTrainingArena(TeamArena):
    """Directional approach/evasion reward screen without changing observations."""
    def __init__(self, config, variant="task"):
        if variant not in {"task", "progress", "directional"}: raise ValueError("Unknown chase reward")
        self.variant = variant
        super().__init__(config)

    def step(self, actions):
        target = self.game.state["flag_home"][0 if self.ever_flag else 1]
        pos = self.game.state["agent_position"][0].copy()
        distance = float(np.linalg.norm(pos-target))
        separation = float(np.min(np.linalg.norm(self.game.state["agent_position"][self.n:]-pos, axis=1)))
        obs, rewards, done, info = super().step(actions)
        after = self.game.state["agent_position"][0]
        progress = distance-float(np.linalg.norm(after-target))
        bonus = 0.
        if self.variant != "task" and not any(self.game.state["agent_is_tagged"][:self.n]):
            bonus = .1*progress
            if self.variant == "directional" and separation < 25. and progress > 0.:
                after_separation = float(np.min(np.linalg.norm(self.game.state["agent_position"][self.n:]-after, axis=1)))
                bonus += .05*max(0., after_separation-separation)
        rewards = rewards + bonus
        self.metrics["episode_reward"] += bonus
        self.metrics["directional_reward"] = self.metrics.get("directional_reward", 0.) + bonus
        if done:
            info["metrics"]["episode_reward"] += bonus
            info["metrics"]["directional_reward"] = self.metrics["directional_reward"]
        return obs, rewards, done, info
