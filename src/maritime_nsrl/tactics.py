"""Explicit per-player tactical primitives, independent of the language model."""
from dataclasses import dataclass
import numpy as np
from .opponents import steer
from .research_env import ArenaConfig, TeamArena
from .temporal import compile_ltlf


@dataclass
class TacticsConfig:
    defender_mode: str = "fixed"
    attack_style: str = "split_flank"
    red_behavior: str = "standard"
    carrier_reassignment: bool = False

    def __post_init__(self):
        if self.defender_mode not in {"none", "fixed", "conditional", "dynamic"}:
            raise ValueError("Unsupported defender assignment")
        if self.attack_style not in {"head_on", "split_flank", "escort"}:
            raise ValueError("Unsupported attacking tactic")
        if self.red_behavior not in {"standard", "raid"}:
            raise ValueError("Unsupported tactical opponent")


def role_contract(mode):
    TacticsConfig(defender_mode=mode)
    mission = "G(in_bounds) & F(has_flag & F(captured))"
    constraints = [] if mode == "none" else ["G(defender_behind_center)" if mode == "fixed"
        else "G(opponent_push -> defender_behind_center)"]
    formula = " & ".join([mission, *constraints])
    return {"kind": "human-authored-tactical-contract", "mission_instruction": "Obtain the enemy flag and capture it while staying in bounds.",
        "human_constraints": constraints, "ltl": formula, "logic_backend": "dfa",
        "validation": compile_ltlf(formula).report(),
        "role_grounding": "defender_behind_center refers to the currently assigned defender; no defender makes it false",
        "guarantee": "monitor only; role assignment and heuristic navigation do not guarantee satisfaction"}


class TacticalArena(TeamArena):
    """Same physics, role-specific waypoint semantics for scripted tactical tests."""
    def __init__(self, config=None, tactics=None):
        self.tactics = tactics or TacticsConfig()
        config = config or ArenaConfig(team_size=3)
        if config.team_size == 1 and self.tactics.defender_mode != "none":
            raise ValueError("A 1v1 tactical profile has no separate defender; use defender_mode=none")
        super().__init__(config)

    def roles(self, game=None):
        game = game or self.game
        s, n = game.state, self.n
        roles = ["attacker_low" if i % 2 else "attacker_high" for i in range(n)]
        if n == 1: return ["attacker_low"]
        threat = any(s["agent_position"][n:, 0] < self.size[0]/2)
        mode = self.tactics.defender_mode
        available = [i for i in range(n) if not s["agent_has_flag"][i]]
        if mode != "none" and available:
            defender = min(available, key=lambda i: np.linalg.norm(s["agent_position"][i]-s["flag_home"][0])) if mode == "dynamic" else 0
            if defender not in available and self.tactics.carrier_reassignment:
                defender = min(available, key=lambda i: np.linalg.norm(s["agent_position"][i]-s["flag_home"][0]))
            # A carrier always returns; a fixed role never overrides possession.
            if defender in available:
                roles[defender] = "defender" if mode in {"fixed", "dynamic"} or threat else "support"
        if self.tactics.attack_style == "escort" and any(s["agent_has_flag"][:n]):
            for i in available:
                if roles[i] != "defender": roles[i] = "support"
        for i in range(n):
            if s["agent_has_flag"][i]: roles[i] = "carrier"
        return roles

    def labels(self, game=None, previous_captures=None):
        game = game or self.game
        labels = super().labels(game, previous_captures)
        labels.discard("defender_behind_center"); labels.discard("attacker_forward")
        roles = self.roles(game)
        if any(role == "defender" and game.state["agent_position"][i, 0] < self.size[0]/2 for i, role in enumerate(roles)):
            labels.add("defender_behind_center")
        if any(role.startswith("attacker") and game.state["agent_position"][i, 0] > self.size[0]/2 for i, role in enumerate(roles)):
            labels.add("attacker_forward")
        return labels

    def target(self, index, action):
        if action != 0: return super().target(index, action)
        s, n = self.game.state, self.n
        role = self.roles()[index]
        pos = s["agent_position"][index]
        if role == "defender":
            intruders = [i for i in range(n, 2*n) if s["agent_position"][i, 0] < self.size[0]/2 and not s["agent_is_tagged"][i]]
            if intruders:
                chosen = min(intruders, key=lambda i: (not s["agent_has_flag"][i], np.linalg.norm(s["agent_position"][i]-pos)))
                theta = np.radians(s["agent_heading"][chosen])
                velocity = np.array([np.sin(theta), np.cos(theta)]) * s["agent_speed"][chosen]
                return np.clip(s["agent_position"][chosen] + 2*velocity, [5., 5.], [self.size[0]/2-5., self.size[1]-5.])
            home = s["flag_home"][0]
            angle = np.arctan2(*(pos-home)[::-1]) + np.pi/3
            return np.clip(home + .14*min(self.size)*np.array([np.cos(angle), np.sin(angle)]), [5., 5.], [self.size[0]/2-5., self.size[1]-5.])
        if role == "support":
            carriers = np.flatnonzero(s["agent_has_flag"][:n])
            if len(carriers):
                carrier = s["agent_position"][carriers[0]]
                offset = np.array([-8., 8. if index % 2 else -8.])
                return np.clip(carrier + offset, [5., 5.], self.size-5.)
            return np.array([.42*self.size[0], (.32 if index % 2 else .68)*self.size[1]])
        return super().target(index, action)

    def opponent_actions(self):
        if self.tactics.red_behavior != "raid": return super().opponent_actions()
        s = self.game.state
        return {f"agent_{i}": steer(self.game, i, s["flag_home"][1 if s["agent_has_flag"][i] or s["agent_is_tagged"][i] else 0], .8)
            for i in range(self.n, 2*self.n)}


class TacticalPolicy:
    def __call__(self, arena):
        s = arena.game.state
        actions = []
        for i, role in enumerate(arena.roles()):
            if s["agent_is_tagged"][i]: action = 1
            elif role == "carrier":
                action = (4 if s["agent_position"][i, 1] < arena.size[1]/2 else 5) if s["agent_position"][i, 0] > .42*arena.size[0] else 1
            elif role in {"defender", "support"}: action = 0
            elif arena.tactics.attack_style == "head_on": action = 0
            else: action = 2 if role == "attacker_low" else 3
            actions.append(action)
        return np.asarray(actions, dtype=int)
