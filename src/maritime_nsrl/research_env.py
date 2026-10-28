"""Joint PyQuaticus arena for parameter-sharing multi-agent PPO experiments."""

import copy
import warnings
from dataclasses import dataclass
import numpy as np
from gymnasium import spaces
from pyquaticus import pyquaticus_v0
from stable_baselines3.common.vec_env import VecEnv
from .opponents import steer
from .temporal import compile_ltlf, TemporalMonitor
from .shield import PredictiveShield

SCENARIOS = {"standard": (160.0, 80.0), "wide": (200.0, 100.0), "narrow": (160.0, 60.0)}
DEFAULT_FORMULA = "G(in_bounds) & G(has_flag -> F(captured)) & F(captured)"
ROLE_FORMULA = "G(in_bounds) & G(!tagged) & G(defender_behind_center) & F(has_flag & F(captured))"


@dataclass
class ArenaConfig:
    team_size: int = 1
    guided: bool = True
    reward_shaping: bool = True
    temporal_state: bool = True
    shield: bool = False
    shield_formula: str = "G(in_bounds)"
    formula: str = DEFAULT_FORMULA
    opponent: str = "chase"
    opponent_speed: float = 1.0
    scenario: str = "standard"
    random_starts: bool = True
    horizon: float = 240.0
    repeat: int = 5
    gamma: float = 0.99
    capture_reward: float = 20.0
    pickup_reward: float = 2.0
    tag_penalty: float = 2.0
    time_penalty: float = 0.01
    potential_scale: float = 5.0
    acceptance_bonus: float = 4.0
    failure_penalty: float = 1.0
    # Chase-recovery diagnostics.  These are disabled by default so existing
    # experiments retain their original observation and reward definitions.
    approach_progress_scale: float = 0.0
    evasion_scale: float = 0.0
    defender_distance_feature: bool = False
    trace_stride: int = 0
    # Separate evaluation stress distribution; default reproduces prior resets.
    start_mode: str = "standard"
    boundary_start_margin: float = 2.5
    boundary_start_speed: float = 0.0


class TeamArena:
    """Every blue agent takes a learned action; red agents are scripted opponents.

    Team LTL grounding: in_bounds=all blue in bounds, tagged=any blue tagged,
    has_flag=any blue carrier, captured=blue score event, home=any blue at home.
    """

    def __init__(self, config=None):
        self.config = config or ArenaConfig()
        config = self.config
        if config.team_size not in (1, 2, 3):
            raise ValueError("Supported teams: 1, 2, or 3")
        if config.opponent not in {"static", "patrol", "chase"}:
            raise ValueError(config.opponent)
        if config.scenario not in SCENARIOS or config.repeat < 1 or config.horizon <= 0:
            raise ValueError("Invalid arena config")
        if not 0 < config.opponent_speed <= 1:
            raise ValueError("Opponent speed must be in (0,1]")
        if config.start_mode not in {"standard", "boundary_lower", "boundary_upper"}:
            raise ValueError("Unknown start distribution")
        if not 0 < config.boundary_start_margin < SCENARIOS[config.scenario][1] / 2:
            raise ValueError("Boundary start margin must be inside the arena")
        self.n = config.team_size
        self.size = np.array(SCENARIOS[config.scenario])
        self.automaton = compile_ltlf(config.formula)
        self.rng = np.random.default_rng()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="divide by zero encountered in divide", category=RuntimeWarning
            )
            self.game = pyquaticus_v0.PyQuaticusEnv(
                team_size=self.n,
                action_space="continuous",
                config_dict={
                    "env_bounds": self.size.tolist(),
                    "max_time": config.horizon,
                    "max_score": 1,
                    "tau": 0.1,
                    "sim_speedup_factor": 10,
                    "tagging_cooldown": 30.0,
                    "suppress_numpy_warnings": False,
                },
            )
        self.controllers = {
            k: copy.deepcopy(p._pid_controllers) for k, p in self.game.players.items()
        }
        if config.start_mode != "standard" and config.boundary_start_margin < max(
            self.game.agent_radius[: self.n]
        ):
            self.game.close()
            raise ValueError(
                "Boundary start must keep the complete initial vessel footprint inside the field"
            )
        if not 0 <= config.boundary_start_speed <= min(self.game.max_speeds[: self.n]):
            self.game.close()
            raise ValueError("Boundary initial speed exceeds simulator limits")
        # 0 flag, 1 home, 2 low flank, 3 high flank, 4 low return lane,
        # 5 high return lane, 6 evade nearest opponent, 7 hold position.
        self.action_space = spaces.Discrete(8)
        extra_features = int(config.defender_distance_feature)
        self.observation_space = spaces.Box(
            -2.0,
            2.0,
            shape=(16 * self.n + 3 + len(self.automaton.states) + extra_features,),
            dtype=np.float32,
        )
        self.finished = True

    def labels(self, game=None, previous_captures=None):
        game = game or self.game
        state, team_size = game.state, self.n
        result = set()
        if not any(state["agent_oob"][:team_size]):
            result.add("in_bounds")
        if any(state["agent_is_tagged"][:team_size]):
            result.add("tagged")
        if any(state["agent_has_flag"][:team_size]):
            result.add("has_flag")
        if previous_captures is not None and state["captures"][0] > previous_captures:
            result.add("captured")
        if (
            min(np.linalg.norm(state["agent_position"][:team_size] - state["flag_home"][0], axis=1))
            <= game.catch_radius
        ):
            result.add("home")
        center = self.size[0] / 2
        if any(state["agent_position"][team_size:, 0] < center):
            result.add("opponent_push")
        # Agent 0 is the defender; the remaining blue agents are attackers.
        if team_size >= 2 and state["agent_position"][0, 0] < center:
            result.add("defender_behind_center")
        if team_size >= 2 and any(state["agent_position"][1:team_size, 0] > center):
            result.add("attacker_forward")
        if not any(state["agent_is_tagged"][:team_size]):
            result.add("teammate_safe")
        return result

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        width, height = self.size
        config, team_size = self.config, self.n
        ys = (
            np.linspace(0.3 * height, 0.7 * height, team_size)
            if team_size > 1
            else np.array([height / 2])
        )
        if config.random_starts:
            ys += self.rng.uniform(-0.12 * height, 0.12 * height, team_size)
        red_ys = (
            np.linspace(0.2 * height, 0.8 * height, team_size)
            if team_size > 1
            else np.array([0.84 * height])
        )
        if config.random_starts:
            red_ys += self.rng.uniform(-0.08 * height, 0.08 * height, team_size)
        positions = [[0.22 * width, y] for y in ys] + [[0.75 * width, y] for y in red_ys]
        headings = [90.0] * team_size + [180.0] * team_size
        speeds = [0.0] * (2 * team_size)
        if config.start_mode != "standard":
            lower = config.start_mode == "boundary_lower"
            positions[:team_size] = [
                [
                    0.22 * width + 5 * i,
                    config.boundary_start_margin
                    if lower
                    else height - config.boundary_start_margin,
                ]
                for i in range(team_size)
            ]
            headings[:team_size] = [180.0 if lower else 0.0] * team_size
            speeds[:team_size] = [config.boundary_start_speed] * team_size
        for key, controllers in self.controllers.items():
            self.game.players[key]._pid_controllers = copy.deepcopy(controllers)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="divide by zero encountered in divide", category=RuntimeWarning
            )
            self.game.reset(
                seed=int(self.rng.integers(0, 2**31 - 1)),
                options={
                    "init_dict": {
                        "agent_position": positions,
                        "agent_heading": headings,
                        "agent_speed": speeds,
                    }
                },
            )
        self.monitor = TemporalMonitor(self.automaton)
        self.monitor.update(self.labels())
        self.gate = PredictiveShield(config.shield_formula) if config.shield else None
        if self.gate:
            self.gate.initialize(self.labels())
        self.patrol_phase = [i % 2 for i in range(team_size)]
        self.ever_flag = False
        self.finished = False
        self.metrics = dict(
            base_reward=0.0,
            episode_reward=0.0,
            guidance_reward=0.0,
            captures=0,
            opponent_captures=0,
            pickups=0,
            tags_received=0,
            out_of_bounds_steps=0,
            temporal_failure_steps=0,
            shield_interventions=0,
            shield_deadlocks=0,
            shield_predictions=0,
            capture_time=None,
            decisions=0,
            per_agent_pickups=[0] * team_size,
            per_agent_tags=[0] * team_size,
            enemy_territory_steps=0,
            reached_enemy_territory=False,
            first_enemy_territory_time=None,
            reached_enemy_flag=False,
            first_enemy_flag_time=None,
            enemy_flag_distance_min=None,
            defender_distance_min=None,
            defender_distance_sum=0.0,
            defender_distance_samples=0,
            pickup_time=None,
            return_crossed_midfield=False,
            first_return_midfield_time=None,
            tagged_steps=0,
            tags_before_pickup=0,
            tags_after_pickup=0,
            tag_events_before_pickup=0,
            tag_events_after_pickup=0,
            numerical_terminations=0,
            numerical_termination_time=None,
            common_task_reward=0.0,
            temporal_state_visits=[0] * len(self.automaton.states),
            flag_carrier_counts=[0] * team_size,
            defender_violation_steps=0,
            fixed_safety_violation_steps=0,
            requested_action_counts=[0] * 8,
            executed_action_counts=[0] * 8,
            shield_offensive_interventions=0,
            shield_substitutions={},
        )
        self.trace = []
        self.last_valid_observation = self.observe()
        return self.last_valid_observation

    def finite_game_state(self):
        """Detect simulator numerical failure before invalid state reaches PPO."""
        state = self.game.state
        keys = ("agent_position", "agent_heading", "agent_speed", "flag_home")
        return all(
            key in state and np.all(np.isfinite(np.asarray(state[key], dtype=float)))
            for key in keys
        )

    def observe(self):
        state, team_size = self.game.state, self.n
        hdg = np.radians(state["agent_heading"])
        entities = np.column_stack(
            [
                state["agent_position"] / self.size,
                np.sin(hdg),
                np.cos(hdg),
                state["agent_speed"] / np.asarray(self.game.max_speeds),
                state["agent_has_flag"],
                state["agent_is_tagged"],
                state["agent_tagging_cooldown"] / self.game.tagging_cooldown,
            ]
        )
        temporal_state = np.zeros(len(self.automaton.states))
        if self.config.guided and self.config.temporal_state:
            temporal_state[self.monitor.state] = 1.0
        observations = []
        for i in range(team_size):
            order = (
                [i]
                + [j for j in range(team_size) if j != i]
                + list(range(team_size, 2 * team_size))
            )
            observation = np.concatenate(
                [
                    entities[order].reshape(-1),
                    [
                        self.game.current_time / self.config.horizon,
                        i / max(team_size - 1, 1),
                        float(self.ever_flag),
                    ],
                    temporal_state,
                ]
            )
            if self.config.defender_distance_feature:
                enemies = state["agent_position"][team_size:]
                distance = min(
                    np.linalg.norm(state["agent_position"][i] - enemy) for enemy in enemies
                )
                observation = np.concatenate([observation, [distance / np.linalg.norm(self.size)]])
            observations.append(observation)
        return np.clip(observations, -2, 2).astype(np.float32)

    def approach_evasion_state(self):
        """Nearest blue-to-flag and blue-to-defender distances for shaping."""
        positions = self.game.state["agent_position"][: self.n]
        enemies = self.game.state["agent_position"][self.n :]
        flag_distance = float(
            min(
                np.linalg.norm(position - self.game.state["flag_home"][1]) for position in positions
            )
        )
        defender_distance = float(
            min(np.linalg.norm(position - enemy) for position in positions for enemy in enemies)
        )
        in_enemy_territory = bool(any(positions[:, 0] > self.size[0] / 2))
        return flag_distance, defender_distance, in_enemy_territory

    def target(self, index, action):
        width, height = self.size
        pos = self.game.state["agent_position"][index]
        if action == 0:
            return self.game.state["flag_home"][1]
        if action == 1:
            return self.game.state["flag_home"][0]
        if action in (2, 3):
            y = (0.12 if action == 2 else 0.88) * height
            if pos[0] < 0.55 * width:
                return np.array([0.55 * width, y])
            if pos[0] < 0.86 * width:
                return np.array([0.91 * width, y])
            return self.game.state["flag_home"][1]
        if action in (4, 5):
            return np.array([0.40 * width, (0.15 if action == 4 else 0.85) * height])
        if action == 6:
            enemies = self.game.state["agent_position"][self.n :]
            closest = enemies[np.argmin(np.linalg.norm(enemies - pos, axis=1))]
            direction = pos - closest
            direction /= max(np.linalg.norm(direction), 1e-6)
            return np.clip(pos + direction * 30, [5, 5], self.size - 5)
        if action == 7:
            return None
        raise ValueError(action)

    def opponent_actions(self):
        result = {}
        state, config, team_size = self.game.state, self.config, self.n
        width, height = self.size
        for j in range(team_size):
            i = team_size + j
            if config.opponent == "static":
                target = None
            else:
                targets = [
                    k
                    for k in range(team_size)
                    if not state["agent_on_sides"][k] and not state["agent_is_tagged"][k]
                ]
                if config.opponent == "chase" and targets:
                    chosen = min(
                        targets,
                        key=lambda k: np.linalg.norm(
                            state["agent_position"][k] - state["agent_position"][i]
                        ),
                    )
                    target = state["agent_position"][chosen]
                else:
                    points = [
                        np.array([0.70 * width, 0.20 * height]),
                        np.array([0.70 * width, 0.80 * height]),
                    ]
                    if (
                        np.linalg.norm(state["agent_position"][i] - points[self.patrol_phase[j]])
                        < 6
                    ):
                        self.patrol_phase[j] = 1 - self.patrol_phase[j]
                    target = points[self.patrol_phase[j]]
            fraction = (
                config.opponent_speed
                if config.opponent == "chase"
                else 0.65 * config.opponent_speed
            )
            result[f"agent_{i}"] = steer(self.game, i, target, fraction)
        return result

    def potential(self):
        state, team_size = self.game.state, self.n
        carriers = [i for i in range(team_size) if state["agent_has_flag"][i]]
        target = state["flag_home"][0 if carriers else 1]
        ids = carriers or list(range(team_size))
        distance = min(np.linalg.norm(state["agent_position"][i] - target) for i in ids)
        return 2 * float(self.ever_flag) + 1 - min(distance / np.linalg.norm(self.size), 1.0)

    def candidates(self, actions, opponents):
        requested = {
            f"agent_{i}": steer(self.game, i, self.target(i, int(a))) for i, a in enumerate(actions)
        }
        yield tuple(map(int, actions)), {**opponents, **requested}
        # Bounded candidate set, not exhaustive joint-action enumeration. Correct
        # single agents first, then a common return/stop/evade action for the team.
        for i in range(self.n):
            for alternative in (1, 4, 5, 6, 7, 2, 3, 0):
                if alternative == actions[i]:
                    continue
                changed = list(map(int, actions))
                changed[i] = alternative
                yield (
                    tuple(changed),
                    {
                        **opponents,
                        **requested,
                        f"agent_{i}": steer(self.game, i, self.target(i, alternative)),
                    },
                )
        for alternative in (1, 7, 6):
            yield (
                (alternative,) * self.n,
                {
                    **opponents,
                    **{
                        f"agent_{i}": steer(self.game, i, self.target(i, alternative))
                        for i in range(self.n)
                    },
                },
            )

    def step(self, actions):
        if self.finished:
            raise RuntimeError("Reset required")
        actions = np.asarray(actions).reshape(-1)
        if len(actions) != self.n or any(not self.action_space.contains(int(a)) for a in actions):
            raise ValueError("One valid action per learning agent required")
        phi = self.potential()
        flag_distance_before, defender_distance_before, enemy_before = self.approach_evasion_state()
        base, extra, team_size = 0.0, 0.0, self.n
        requested_actions = tuple(map(int, actions))
        executed_actions = requested_actions
        for action in requested_actions:
            self.metrics["requested_action_counts"][action] += 1
        accepted_before = self.monitor.satisfied
        for _ in range(self.config.repeat):
            state = self.game.state
            caps, grabs, tags = (
                state["captures"].copy(),
                state["grabs"].copy(),
                state["tags"].copy(),
            )
            had_flag = np.array(state["agent_has_flag"][:team_size]).copy()
            was_tagged = np.array(state["agent_is_tagged"][:team_size]).copy()
            opponents = self.opponent_actions()
            choices = list(self.candidates(actions, opponents))
            if self.gate:
                game, result, index = self.gate.transition(
                    self.game, (low for _, low in choices), lambda g: self.labels(g, caps[0])
                )
                if game is None:
                    self.finished = True
                    base -= 20.0
                    break
                self.game = game
                executed_actions = choices[index][0]
                if index > 0:
                    key = f"{','.join(map(str, requested_actions))}->{','.join(map(str, executed_actions))}"
                    self.metrics["shield_substitutions"][key] = (
                        self.metrics["shield_substitutions"].get(key, 0) + 1
                    )
                    if self.ever_flag or any(
                        self.game.state["agent_position"][:team_size, 0] > self.size[0] / 2
                    ):
                        self.metrics["shield_offensive_interventions"] += 1
            else:
                result = self.game.step(choices[0][1])
            _, _, terms, truncs, _ = result
            state = self.game.state
            if not self.finite_game_state():
                # PyQuaticus can occasionally produce invalid interception geometry.
                # End this episode, retain the previous valid observation, and let
                # VecEnv reset instead of contaminating the PPO rollout with NaNs.
                self.finished = True
                self.metrics["numerical_terminations"] += 1
                self.metrics["numerical_termination_time"] = float(self.game.current_time)
                base -= 20.0
                break
            scored = int(state["captures"][0] - caps[0])
            picked = int(state["grabs"][0] - grabs[0])
            entered_tag = np.asarray(state["agent_is_tagged"][:team_size]) & ~was_tagged
            base += (
                self.config.capture_reward * scored
                + self.config.pickup_reward * picked
                - self.config.tag_penalty * sum(entered_tag)
                - self.config.time_penalty
            )
            # A fixed evaluation measure.  It remains comparable when an arm changes
            # its training shaping or tag coefficient.
            metrics = self.metrics
            metrics["common_task_reward"] += (
                20.0 * scored + 2.0 * picked - 2.0 * sum(entered_tag) - 0.01
            )
            self.ever_flag |= any(state["agent_has_flag"][:team_size])
            self.monitor.update(self.labels(previous_captures=caps[0]))
            metrics["captures"] += scored
            metrics["opponent_captures"] += int(state["captures"][1] - caps[1])
            metrics["pickups"] += picked
            metrics["tags_received"] += int(state["tags"][1] - tags[1])
            metrics["out_of_bounds_steps"] += int(any(state["agent_oob"][:team_size]))
            metrics["temporal_failure_steps"] += int(self.monitor.impossible)
            labels = self.labels(previous_captures=caps[0])
            positions = state["agent_position"][:team_size]
            enemy_steps = int(any(positions[:, 0] > self.size[0] / 2))
            tagged_step = int("tagged" in labels)
            defender_step = int(team_size > 1 and "defender_behind_center" not in labels)
            fixed_violation = int("in_bounds" not in labels or tagged_step or defender_step)
            metrics["enemy_territory_steps"] += enemy_steps
            metrics["tagged_steps"] += tagged_step
            if any(entered_tag):
                suffix = "after_pickup" if metrics["pickup_time"] is not None else "before_pickup"
                metrics[f"tags_{suffix}"] += int(sum(entered_tag))
                metrics[f"tag_events_{suffix}"] += 1
            metrics["defender_violation_steps"] += defender_step
            metrics["fixed_safety_violation_steps"] += fixed_violation
            metrics["rule_violation_steps"] = metrics["fixed_safety_violation_steps"]
            now = float(self.game.current_time)
            if enemy_steps and not metrics["reached_enemy_territory"]:
                metrics["reached_enemy_territory"], metrics["first_enemy_territory_time"] = (
                    True,
                    now,
                )
            flag_distance = float(min(np.linalg.norm(positions - state["flag_home"][1], axis=1)))
            metrics["enemy_flag_distance_min"] = (
                flag_distance
                if metrics["enemy_flag_distance_min"] is None
                else min(metrics["enemy_flag_distance_min"], flag_distance)
            )
            enemies = state["agent_position"][team_size:]
            defender_distance = float(
                min(np.linalg.norm(position - enemy) for position in positions for enemy in enemies)
            )
            metrics["defender_distance_min"] = (
                defender_distance
                if metrics["defender_distance_min"] is None
                else min(metrics["defender_distance_min"], defender_distance)
            )
            metrics["defender_distance_sum"] += defender_distance
            metrics["defender_distance_samples"] += 1
            if flag_distance <= self.game.catch_radius and not metrics["reached_enemy_flag"]:
                metrics["reached_enemy_flag"], metrics["first_enemy_flag_time"] = True, now
            if picked and metrics["pickup_time"] is None:
                metrics["pickup_time"] = now
            carriers = np.flatnonzero(state["agent_has_flag"][:team_size])
            if (scored or any(positions[carriers, 0] < self.size[0] / 2)) and not metrics[
                "return_crossed_midfield"
            ]:
                metrics["return_crossed_midfield"], metrics["first_return_midfield_time"] = (
                    True,
                    now,
                )
            for i in range(team_size):
                metrics["per_agent_pickups"][i] += int(
                    state["agent_has_flag"][i] and not had_flag[i]
                )
                metrics["per_agent_tags"][i] += int(entered_tag[i])
                metrics["flag_carrier_counts"][i] += int(state["agent_has_flag"][i])
            if scored and metrics["capture_time"] is None:
                metrics["capture_time"] = float(self.game.current_time)
            if self.config.guided and self.config.reward_shaping and self.monitor.impossible:
                extra -= self.config.failure_penalty
            self.finished = bool(terms["agent_0"] or truncs["agent_0"])
            if self.finished:
                break
        if self.config.guided and self.config.reward_shaping:
            extra += self.config.potential_scale * (
                self.config.gamma * (0 if self.finished else self.potential()) - phi
            )
            extra += self.config.acceptance_bonus * int(
                self.monitor.satisfied and not accepted_before
            )
            flag_distance_after, defender_distance_after, enemy_after = (
                self.approach_evasion_state()
            )
            scale = max(float(np.linalg.norm(self.size)), 1.0)
            # Reward closing the gap only before the flag is obtained; afterward
            # the existing potential rewards the return leg.
            if not self.ever_flag:
                extra += (
                    self.config.approach_progress_scale
                    * (flag_distance_before - flag_distance_after)
                    / scale
                )
                if enemy_before or enemy_after:
                    extra += (
                        self.config.evasion_scale
                        * (defender_distance_after - defender_distance_before)
                        / scale
                    )
        metrics = self.metrics
        metrics["base_reward"] += float(base)
        metrics["guidance_reward"] += float(extra)
        metrics["episode_reward"] += float(base + extra)
        metrics["decisions"] += 1
        metrics["temporal_state_visits"][self.monitor.state] += 1
        for action in executed_actions:
            metrics["executed_action_counts"][action] += 1
        if self.config.trace_stride and (metrics["decisions"] - 1) % self.config.trace_stride == 0:
            self.trace.append(
                {
                    "decision": metrics["decisions"],
                    "time": float(self.game.current_time),
                    "positions": self.game.state["agent_position"][:team_size].tolist(),
                    "has_flag": list(map(bool, self.game.state["agent_has_flag"][:team_size])),
                    "tagged": list(map(bool, self.game.state["agent_is_tagged"][:team_size])),
                    "requested_actions": list(requested_actions),
                    "executed_actions": list(executed_actions),
                    "monitor_state": self.monitor.state,
                }
            )
        if self.gate:
            metrics.update(
                shield_interventions=self.gate.interventions,
                shield_deadlocks=self.gate.deadlocks,
                shield_predictions=self.gate.predictions,
            )
        info = {}
        if self.finished:
            info["metrics"] = {
                **copy.deepcopy(metrics),
                "duration": float(self.game.current_time),
                "common_task_reward": metrics["common_task_reward"],
                "enemy_territory_time": 0.1 * metrics["enemy_territory_steps"],
                "defender_distance_mean": metrics["defender_distance_sum"]
                / max(metrics["defender_distance_samples"], 1),
                "rule_violation_steps": metrics.get("rule_violation_steps", 0),
                "ltlf_satisfied": self.monitor.satisfied,
                "team_size": team_size,
                "scenario": self.config.scenario,
                "opponent": self.config.opponent,
                "opponent_speed": self.config.opponent_speed,
                "trajectory": copy.deepcopy(self.trace) if self.config.trace_stride else None,
            }
        if self.finished and metrics["numerical_terminations"]:
            observation = self.last_valid_observation.copy()
        else:
            observation = self.observe()
            self.last_valid_observation = observation.copy()
        return (
            observation,
            np.full(team_size, float(base + extra), dtype=np.float32),
            self.finished,
            info,
        )

    def close(self):
        self.game.close()


class SharedTeamVecEnv(VecEnv):
    """Agents are PPO rollout lanes, with ONE joint game step per step_wait.

    This is parameter-sharing independent PPO with common team rewards and full
    state observations. The lanes are correlated agents, not independent games.
    """

    def __init__(self, arena):
        self.arena = arena
        self.render_mode = None
        super().__init__(arena.n, arena.observation_space, arena.action_space)

    def reset(self):
        obs = self.arena.reset(seed=self._seeds[0])
        self._reset_seeds()
        self._reset_options()
        return obs

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs, rewards, done, info = self.arena.step(self.actions)
        infos = [{} for _ in range(self.num_envs)]
        infos[0].update(info)  # Count each team episode once.
        if done:
            for i in range(self.num_envs):
                infos[i]["terminal_observation"] = obs[i].copy()
            obs = self.arena.reset()
        return obs, rewards, np.full(self.num_envs, done), infos

    def close(self):
        self.arena.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self.arena, method_name)(*args, **kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))
