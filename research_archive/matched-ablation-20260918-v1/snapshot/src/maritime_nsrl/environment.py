"""Gymnasium 1v1 facade; PyQuaticus remains the game/physics authority."""
import copy
import warnings
import gymnasium as gym
import numpy as np
from pyquaticus import pyquaticus_v0
from .guidance import Guidance, labels_from_state
from .opponents import Opponent, steer


class MaritimeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, spec, guided=False, opponent="static", horizon=240., gamma=.99,
                 repeat=5, dynamics="heron", return_start_probability=0.):
        super().__init__()
        self.spec, self.guided, self.opponent_kind = spec, guided, opponent
        self.horizon, self.gamma, self.repeat = horizon, gamma, repeat
        if not 0 <= return_start_probability <= 1: raise ValueError("Curriculum probability must be in [0,1]")
        self.return_start_probability = return_start_probability
        if repeat < 1 or horizon <= 0: raise ValueError("Positive horizon and repeat required")
        self.game = pyquaticus_v0.PyQuaticusEnv(team_size=1, action_space="continuous",
            config_dict={"max_time": horizon, "max_score": 1, "tau": .1,
                         "sim_speedup_factor": 10, "dynamics": dynamics,
                         "tagging_cooldown": 30., "suppress_numpy_warnings": False})
        # Upstream BaseUSV.reset() resets thrust/rudder but retains PID history.
        # Restore pristine controllers so evaluation does not depend on prior episodes.
        self.initial_controllers = {k: copy.deepcopy(v._pid_controllers)
                                    for k,v in self.game.players.items() if hasattr(v, "_pid_controllers")}
        # Identical motor primitives for baseline and guided PPO; no goal-based selection.
        self.waypoints = [np.array(p) for p in [(140,40), (20,40), (85,18), (85,62), (120,18), (120,62)]] + [None]
        self.action_space = gym.spaces.Discrete(len(self.waypoints))
        self.observation_space = gym.spaces.Box(-2., 2., shape=(19,), dtype=np.float32)
        self.finished = True

    def observation(self):
        s = self.game.state
        positions = (s["agent_position"] / self.game.env_size).reshape(-1)
        headings = np.radians(s["agent_heading"])
        monitor = self.guidance.monitor
        values = np.concatenate([positions, np.sin(headings), np.cos(headings),
            s["agent_speed"] / np.asarray(self.game.max_speeds), s["agent_has_flag"],
            s["agent_is_tagged"], s["agent_tagging_cooldown"] / self.game.tagging_cooldown,
            [self.game.current_time / self.horizon,
             monitor.stage / max(len(self.spec.goals), 1) if self.guided else 0.,
             float(monitor.violated) if self.guided else 0.]])
        return np.clip(values, -2, 2).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(0, 2**31-1))
        jitter = self.np_random.uniform(-6, 6)
        return_start = self.return_start_probability > 0 and self.np_random.random() < self.return_start_probability
        init = {"agent_position": [[110. if return_start else 35., 40. + jitter], [120., 67.]],
                "agent_heading": [-90. if return_start else 90., 180.], "agent_speed": [0., 0.]}
        if return_start: init["agent_has_flag"] = np.array([True, False])
        for key, controllers in self.initial_controllers.items():
            self.game.players[key]._pid_controllers = copy.deepcopy(controllers)
        # Parallel-line geometry in upstream reset divides by zero before masking.
        # Keep all other numerical warnings visible and avoid changing np.seterr globally.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in divide",
                                    category=RuntimeWarning, module=r"pyquaticus\.utils\.utils")
            self.game.reset(seed=game_seed, options={"init_dict": init})
        self.opponent = Opponent(self.opponent_kind)
        self.guidance = Guidance(self.spec, gamma=self.gamma)
        self.guidance.monitor.update(labels_from_state(self.game))
        self.metrics = dict(episode_reward=0., base_reward=0., guidance_reward=0.,
            flag_captures=0, flag_pickups=0, capture_time=None, tags_made=0, tags_received=0,
            tagged_events=0, rule_violation_steps=0, out_of_bounds_steps=0, steps=0,
            curriculum_return_start=bool(return_start))
        self.finished = False
        return self.observation(), {"curriculum_return_start": bool(return_start)}

    def step(self, action):
        if self.finished: raise RuntimeError("Reset before stepping a completed episode")
        if not self.action_space.contains(action): raise ValueError(f"Invalid action: {action}")
        phi_before = self.guidance.potential(self.game)
        base, milestones, violations = 0., 0, 0
        terminated = truncated = False
        for _ in range(self.repeat):
            s = self.game.state
            captures, grabs, tags = s["captures"].copy(), s["grabs"].copy(), s["tags"].copy()
            was_tagged = bool(s["agent_is_tagged"][0])
            _, _, terms, truncs, _ = self.game.step({
                "agent_0": steer(self.game, 0, self.waypoints[int(action)]),
                "agent_1": self.opponent.action(self.game)})
            s = self.game.state
            scored = int(s["captures"][0] - captures[0])
            grabbed = int(s["grabs"][0] - grabs[0])
            tagged_event = int(bool(s["agent_is_tagged"][0]) and not was_tagged)
            # Common task reward: captures, pickups, time cost, and entering tagged state.
            base += 20.*scored + 2.*grabbed - 2.*tagged_event - .01
            new_goals, violated = self.guidance.monitor.update(labels_from_state(self.game, previous_captures=captures[0]))
            milestones += new_goals
            violations += int(violated)
            m = self.metrics
            m["flag_captures"] += scored
            m["flag_pickups"] += grabbed
            m["tags_made"] += int(s["tags"][0] - tags[0])
            m["tags_received"] += int(s["tags"][1] - tags[1])
            m["tagged_events"] += tagged_event
            m["out_of_bounds_steps"] += int(s["agent_oob"][0])
            if scored and m["capture_time"] is None: m["capture_time"] = float(self.game.current_time)
            terminated = bool(terms["agent_0"])
            truncated = bool(truncs["agent_0"])
            if terminated or truncated: break
        self.finished = terminated or truncated
        # A time limit is a finite task deadline, so both kinds of ending have Phi=0.
        phi_after = 0. if self.finished else self.guidance.potential(self.game)
        shaped = self.guidance.reward(phi_before, phi_after, milestones, False) - self.guidance.violation_penalty * violations
        extra = shaped if self.guided else 0.
        total = base + extra
        self.metrics["steps"] += 1
        self.metrics["episode_reward"] += total
        self.metrics["base_reward"] += base
        self.metrics["guidance_reward"] += extra
        self.metrics["rule_violation_steps"] = self.guidance.monitor.violation_steps
        info = {"base_reward": base, "guidance_reward": extra}
        if self.finished:
            info["metrics"] = {**self.metrics, "ltlf_satisfied": self.guidance.monitor.satisfied,
                "mission_completed": self.guidance.monitor.stage == len(self.spec.goals),
                "duration_seconds": float(self.game.current_time), "opponent": self.opponent_kind}
        # Deadline is part of this finite-horizon MDP (time is observed). Returning
        # termination prevents SB3 from bootstrapping beyond the task's deadline.
        info["environment_time_limit"] = truncated
        return self.observation(), total, self.finished, False, info

    def close(self): self.game.close()
