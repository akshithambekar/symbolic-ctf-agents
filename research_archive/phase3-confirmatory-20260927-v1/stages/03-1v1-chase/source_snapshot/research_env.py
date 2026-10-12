"""Joint PyQuaticus arena for parameter-sharing multi-agent PPO experiments."""
import copy
import warnings
from dataclasses import dataclass, asdict
import numpy as np
from gymnasium import spaces
from pyquaticus import pyquaticus_v0
from stable_baselines3.common.vec_env import VecEnv
from .opponents import steer
from .temporal import compile_ltlf, TemporalMonitor
from .shield import PredictiveShield

SCENARIOS = {"standard": (160., 80.), "wide": (200., 100.), "narrow": (160., 60.)}
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
    opponent_speed: float = 1.
    scenario: str = "standard"
    random_starts: bool = True
    horizon: float = 240.
    repeat: int = 5
    gamma: float = .99
    capture_reward: float = 20.
    pickup_reward: float = 2.
    tag_penalty: float = 2.
    time_penalty: float = .01
    potential_scale: float = 5.
    acceptance_bonus: float = 4.
    failure_penalty: float = 1.
    trace_stride: int = 0


class TeamArena:
    """Every blue agent takes a learned action; red agents are scripted opponents.

    Team LTL grounding: in_bounds=all blue in bounds, tagged=any blue tagged,
    has_flag=any blue carrier, captured=blue score event, home=any blue at home.
    """
    def __init__(self, config=None):
        self.config = config or ArenaConfig()
        c = self.config
        if c.team_size not in (1, 2, 3): raise ValueError("Supported teams: 1, 2, or 3")
        if c.opponent not in {"static", "patrol", "chase"}: raise ValueError(c.opponent)
        if c.scenario not in SCENARIOS or c.repeat < 1 or c.horizon <= 0: raise ValueError("Invalid arena config")
        if not 0 < c.opponent_speed <= 1: raise ValueError("Opponent speed must be in (0,1]")
        self.n = c.team_size
        self.size = np.array(SCENARIOS[c.scenario])
        self.automaton = compile_ltlf(c.formula)
        self.rng = np.random.default_rng()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in divide", category=RuntimeWarning)
            self.game = pyquaticus_v0.PyQuaticusEnv(team_size=self.n, action_space="continuous",
                config_dict={"env_bounds": self.size.tolist(), "max_time": c.horizon,
                    "max_score": 1, "tau": .1, "sim_speedup_factor": 10,
                    "tagging_cooldown": 30., "suppress_numpy_warnings": False})
        self.controllers = {k: copy.deepcopy(p._pid_controllers) for k,p in self.game.players.items()}
        # 0 flag, 1 home, 2 low flank, 3 high flank, 4 low return lane,
        # 5 high return lane, 6 evade nearest opponent, 7 hold position.
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(-2., 2., shape=(16*self.n+3+len(self.automaton.states),), dtype=np.float32)
        self.finished = True

    def labels(self, game=None, previous_captures=None):
        game = game or self.game
        s, n = game.state, self.n
        result = set()
        if not any(s["agent_oob"][:n]): result.add("in_bounds")
        if any(s["agent_is_tagged"][:n]): result.add("tagged")
        if any(s["agent_has_flag"][:n]): result.add("has_flag")
        if previous_captures is not None and s["captures"][0] > previous_captures: result.add("captured")
        if min(np.linalg.norm(s["agent_position"][:n] - s["flag_home"][0], axis=1)) <= game.catch_radius:
            result.add("home")
        center = self.size[0] / 2
        # Agent 0 is the defender; the remaining blue agents are attackers.
        if n >= 2 and s["agent_position"][0, 0] < center: result.add("defender_behind_center")
        if n >= 2 and any(s["agent_position"][1:n, 0] > center): result.add("attacker_forward")
        if not any(s["agent_is_tagged"][:n]): result.add("teammate_safe")
        return result

    def reset(self, seed=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        w,h = self.size
        c, n = self.config, self.n
        ys = np.linspace(.3*h, .7*h, n) if n > 1 else np.array([h/2])
        if c.random_starts: ys += self.rng.uniform(-.12*h, .12*h, n)
        red_ys = np.linspace(.2*h, .8*h, n) if n > 1 else np.array([.84*h])
        if c.random_starts: red_ys += self.rng.uniform(-.08*h, .08*h, n)
        positions = [[.22*w, y] for y in ys] + [[.75*w, y] for y in red_ys]
        for key, controllers in self.controllers.items():
            self.game.players[key]._pid_controllers = copy.deepcopy(controllers)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in divide", category=RuntimeWarning)
            self.game.reset(seed=int(self.rng.integers(0, 2**31-1)), options={"init_dict": {
                "agent_position": positions, "agent_heading": [90.]*n+[180.]*n, "agent_speed": [0.]*(2*n)}})
        self.monitor = TemporalMonitor(self.automaton)
        self.monitor.update(self.labels())
        self.gate = PredictiveShield(c.shield_formula) if c.shield else None
        if self.gate: self.gate.initialize(self.labels())
        self.patrol_phase = [i % 2 for i in range(n)]
        self.ever_flag = False
        self.finished = False
        self.metrics = dict(base_reward=0., episode_reward=0., guidance_reward=0., captures=0,
            opponent_captures=0, pickups=0, tags_received=0, out_of_bounds_steps=0,
            temporal_failure_steps=0, shield_interventions=0, shield_deadlocks=0,
            shield_predictions=0, capture_time=None, decisions=0,
            per_agent_pickups=[0]*n, per_agent_tags=[0]*n,
            enemy_territory_steps=0, reached_enemy_territory=False,
            first_enemy_territory_time=None, reached_enemy_flag=False,
            first_enemy_flag_time=None, enemy_flag_distance_min=None,
            pickup_time=None, return_crossed_midfield=False,
            first_return_midfield_time=None, tagged_steps=0,
            tags_before_pickup=0, tags_after_pickup=0,
            tag_events_before_pickup=0, tag_events_after_pickup=0,
            numerical_terminations=0, numerical_termination_time=None,
            common_task_reward=0., temporal_state_visits=[0]*len(self.automaton.states),
            flag_carrier_counts=[0]*n,
            defender_violation_steps=0, fixed_safety_violation_steps=0,
            requested_action_counts=[0]*8, executed_action_counts=[0]*8,
            shield_offensive_interventions=0, shield_substitutions={})
        self.trace = []
        self.last_valid_observation = self.observe()
        return self.last_valid_observation

    def finite_game_state(self):
        """Detect simulator numerical failure before invalid state reaches PPO."""
        s = self.game.state
        keys = ("agent_position", "agent_heading", "agent_speed", "flag_home")
        return all(key in s and np.all(np.isfinite(np.asarray(s[key], dtype=float))) for key in keys)

    def observe(self):
        s,n = self.game.state,self.n
        hdg = np.radians(s["agent_heading"])
        entities = np.column_stack([s["agent_position"]/self.size, np.sin(hdg), np.cos(hdg),
            s["agent_speed"]/np.asarray(self.game.max_speeds), s["agent_has_flag"],
            s["agent_is_tagged"], s["agent_tagging_cooldown"]/self.game.tagging_cooldown])
        state = np.zeros(len(self.automaton.states))
        if self.config.guided and self.config.temporal_state: state[self.monitor.state] = 1.
        observations = []
        for i in range(n):
            order = [i] + [j for j in range(n) if j != i] + list(range(n,2*n))
            observations.append(np.concatenate([entities[order].reshape(-1),
                [self.game.current_time/self.config.horizon, i/max(n-1,1), float(self.ever_flag)], state]))
        return np.clip(observations, -2, 2).astype(np.float32)

    def target(self, index, action):
        w,h = self.size
        pos = self.game.state["agent_position"][index]
        if action == 0: return self.game.state["flag_home"][1]
        if action == 1: return self.game.state["flag_home"][0]
        if action in (2,3):
            y = (.12 if action == 2 else .88)*h
            if pos[0] < .55*w: return np.array([.55*w,y])
            if pos[0] < .86*w: return np.array([.91*w,y])
            return self.game.state["flag_home"][1]
        if action in (4,5): return np.array([.40*w, (.15 if action == 4 else .85)*h])
        if action == 6:
            enemies = self.game.state["agent_position"][self.n:]
            closest = enemies[np.argmin(np.linalg.norm(enemies-pos, axis=1))]
            direction = pos-closest
            direction /= max(np.linalg.norm(direction), 1e-6)
            return np.clip(pos+direction*30, [5,5], self.size-5)
        if action == 7: return None
        raise ValueError(action)

    def opponent_actions(self):
        result = {}
        s,c,n = self.game.state,self.config,self.n
        w,h = self.size
        for j in range(n):
            i = n+j
            if c.opponent == "static": target = None
            else:
                targets = [k for k in range(n) if not s["agent_on_sides"][k] and not s["agent_is_tagged"][k]]
                if c.opponent == "chase" and targets:
                    chosen = min(targets, key=lambda k: np.linalg.norm(s["agent_position"][k]-s["agent_position"][i]))
                    target = s["agent_position"][chosen]
                else:
                    points = [np.array([.70*w,.20*h]), np.array([.70*w,.80*h])]
                    if np.linalg.norm(s["agent_position"][i]-points[self.patrol_phase[j]]) < 6:
                        self.patrol_phase[j] = 1-self.patrol_phase[j]
                    target = points[self.patrol_phase[j]]
            fraction = c.opponent_speed if c.opponent == "chase" else .65*c.opponent_speed
            result[f"agent_{i}"] = steer(self.game,i,target,fraction)
        return result

    def potential(self):
        s,n = self.game.state,self.n
        carriers = [i for i in range(n) if s["agent_has_flag"][i]]
        target = s["flag_home"][0 if carriers else 1]
        ids = carriers or list(range(n))
        distance = min(np.linalg.norm(s["agent_position"][i]-target) for i in ids)
        return 2*float(self.ever_flag)+1-min(distance/np.linalg.norm(self.size),1.)

    def candidates(self, actions, opponents):
        requested = {f"agent_{i}": steer(self.game,i,self.target(i,int(a))) for i,a in enumerate(actions)}
        yield tuple(map(int, actions)), {**opponents, **requested}
        # Bounded candidate set, not exhaustive joint-action enumeration. Correct
        # single agents first, then a common return/stop/evade action for the team.
        for i in range(self.n):
            for alternative in (1,4,5,6,7,2,3,0):
                if alternative == actions[i]: continue
                changed = list(map(int, actions)); changed[i] = alternative
                yield tuple(changed), {**opponents, **requested, f"agent_{i}": steer(self.game,i,self.target(i,alternative))}
        for alternative in (1,7,6):
            yield (alternative,)*self.n, {**opponents, **{f"agent_{i}": steer(self.game,i,self.target(i,alternative)) for i in range(self.n)}}

    def step(self, actions):
        if self.finished: raise RuntimeError("Reset required")
        actions = np.asarray(actions).reshape(-1)
        if len(actions) != self.n or any(not self.action_space.contains(int(a)) for a in actions):
            raise ValueError("One valid action per learning agent required")
        phi = self.potential()
        base, extra, n = 0.,0.,self.n
        requested_actions = tuple(map(int, actions))
        executed_actions = requested_actions
        for action in requested_actions: self.metrics["requested_action_counts"][action] += 1
        accepted_before = self.monitor.satisfied
        for _ in range(self.config.repeat):
            s = self.game.state
            caps,grabs,tags = s["captures"].copy(),s["grabs"].copy(),s["tags"].copy()
            had_flag = np.array(s["agent_has_flag"][:n]).copy()
            was_tagged = np.array(s["agent_is_tagged"][:n]).copy()
            opponents = self.opponent_actions()
            choices = list(self.candidates(actions,opponents))
            if self.gate:
                game,result,index = self.gate.transition(self.game,(low for _,low in choices),lambda g: self.labels(g,caps[0]))
                if game is None:
                    self.finished = True
                    base -= 20.
                    break
                self.game = game
                executed_actions = choices[index][0]
                if index > 0:
                    key = f"{','.join(map(str,requested_actions))}->{','.join(map(str,executed_actions))}"
                    self.metrics["shield_substitutions"][key] = self.metrics["shield_substitutions"].get(key,0)+1
                    if self.ever_flag or any(self.game.state["agent_position"][:n,0] > self.size[0]/2):
                        self.metrics["shield_offensive_interventions"] += 1
            else:
                result = self.game.step(choices[0][1])
            _,_,terms,truncs,_ = result
            s = self.game.state
            if not self.finite_game_state():
                # PyQuaticus can occasionally produce invalid interception geometry.
                # End this episode, retain the previous valid observation, and let
                # VecEnv reset instead of contaminating the PPO rollout with NaNs.
                self.finished = True
                self.metrics["numerical_terminations"] += 1
                self.metrics["numerical_termination_time"] = float(self.game.current_time)
                base -= 20.
                break
            scored = int(s["captures"][0]-caps[0])
            picked = int(s["grabs"][0]-grabs[0])
            entered_tag = np.asarray(s["agent_is_tagged"][:n]) & ~was_tagged
            base += (self.config.capture_reward*scored + self.config.pickup_reward*picked
                     - self.config.tag_penalty*sum(entered_tag) - self.config.time_penalty)
            # A fixed evaluation measure.  It remains comparable when an arm changes
            # its training shaping or tag coefficient.
            m = self.metrics
            m["common_task_reward"] += 20.*scored + 2.*picked - 2.*sum(entered_tag) - .01
            self.ever_flag |= any(s["agent_has_flag"][:n])
            self.monitor.update(self.labels(previous_captures=caps[0]))
            m["captures"] += scored
            m["opponent_captures"] += int(s["captures"][1]-caps[1])
            m["pickups"] += picked
            m["tags_received"] += int(s["tags"][1]-tags[1])
            m["out_of_bounds_steps"] += int(any(s["agent_oob"][:n]))
            m["temporal_failure_steps"] += int(self.monitor.impossible)
            labels = self.labels(previous_captures=caps[0])
            positions = s["agent_position"][:n]
            enemy_steps = int(any(positions[:,0] > self.size[0]/2))
            tagged_step = int("tagged" in labels)
            defender_step = int(n > 1 and "defender_behind_center" not in labels)
            fixed_violation = int("in_bounds" not in labels or tagged_step or defender_step)
            m["enemy_territory_steps"] += enemy_steps
            m["tagged_steps"] += tagged_step
            if any(entered_tag):
                suffix = "after_pickup" if m["pickup_time"] is not None else "before_pickup"
                m[f"tags_{suffix}"] += int(sum(entered_tag))
                m[f"tag_events_{suffix}"] += 1
            m["defender_violation_steps"] += defender_step
            m["fixed_safety_violation_steps"] += fixed_violation
            m["rule_violation_steps"] = m["fixed_safety_violation_steps"]
            now = float(self.game.current_time)
            if enemy_steps and not m["reached_enemy_territory"]:
                m["reached_enemy_territory"],m["first_enemy_territory_time"] = True,now
            flag_distance = float(min(np.linalg.norm(positions-s["flag_home"][1],axis=1)))
            m["enemy_flag_distance_min"] = flag_distance if m["enemy_flag_distance_min"] is None else min(m["enemy_flag_distance_min"],flag_distance)
            if flag_distance <= self.game.catch_radius and not m["reached_enemy_flag"]:
                m["reached_enemy_flag"],m["first_enemy_flag_time"] = True,now
            if picked and m["pickup_time"] is None: m["pickup_time"] = now
            carriers = np.flatnonzero(s["agent_has_flag"][:n])
            if ((scored or any(positions[carriers,0] < self.size[0]/2)) and
                    not m["return_crossed_midfield"]):
                m["return_crossed_midfield"],m["first_return_midfield_time"] = True,now
            for i in range(n):
                m["per_agent_pickups"][i] += int(s["agent_has_flag"][i] and not had_flag[i])
                m["per_agent_tags"][i] += int(entered_tag[i])
                m["flag_carrier_counts"][i] += int(s["agent_has_flag"][i])
            if scored and m["capture_time"] is None: m["capture_time"] = float(self.game.current_time)
            if self.config.guided and self.config.reward_shaping and self.monitor.impossible:
                extra -= self.config.failure_penalty
            self.finished = bool(terms["agent_0"] or truncs["agent_0"])
            if self.finished: break
        if self.config.guided and self.config.reward_shaping:
            extra += self.config.potential_scale*(self.config.gamma*(0 if self.finished else self.potential())-phi)
            extra += self.config.acceptance_bonus*int(self.monitor.satisfied and not accepted_before)
        m = self.metrics
        m["base_reward"] += float(base)
        m["guidance_reward"] += float(extra)
        m["episode_reward"] += float(base+extra)
        m["decisions"] += 1
        m["temporal_state_visits"][self.monitor.state] += 1
        for action in executed_actions: m["executed_action_counts"][action] += 1
        if self.config.trace_stride and (m["decisions"]-1) % self.config.trace_stride == 0:
            self.trace.append({"decision":m["decisions"],"time":float(self.game.current_time),
                "positions":self.game.state["agent_position"][:n].tolist(),
                "has_flag":list(map(bool,self.game.state["agent_has_flag"][:n])),
                "tagged":list(map(bool,self.game.state["agent_is_tagged"][:n])),
                "requested_actions":list(requested_actions),"executed_actions":list(executed_actions),
                "monitor_state":self.monitor.state})
        if self.gate:
            m.update(shield_interventions=self.gate.interventions, shield_deadlocks=self.gate.deadlocks,
                     shield_predictions=self.gate.predictions)
        info = {}
        if self.finished:
            info["metrics"] = {**copy.deepcopy(m), "duration": float(self.game.current_time),
                "common_task_reward": m["common_task_reward"],
                "rule_violation_steps": m.get("rule_violation_steps", 0),
                "ltlf_satisfied": self.monitor.satisfied, "team_size": n,
                "scenario": self.config.scenario, "opponent": self.config.opponent,
                "opponent_speed": self.config.opponent_speed,
                "trajectory": copy.deepcopy(self.trace) if self.config.trace_stride else None}
        if self.finished and m["numerical_terminations"]:
            observation = self.last_valid_observation.copy()
        else:
            observation = self.observe()
            self.last_valid_observation = observation.copy()
        return observation, np.full(n,float(base+extra),dtype=np.float32), self.finished, info

    def close(self): self.game.close()


class SharedTeamVecEnv(VecEnv):
    """Agents are PPO rollout lanes, with ONE joint game step per step_wait.

    This is parameter-sharing independent PPO with common team rewards and full
    state observations. The lanes are correlated agents, not independent games.
    """
    def __init__(self, arena):
        self.arena = arena
        self.render_mode = None
        super().__init__(arena.n,arena.observation_space,arena.action_space)

    def reset(self):
        obs = self.arena.reset(seed=self._seeds[0])
        self._reset_seeds()
        self._reset_options()
        return obs

    def step_async(self, actions): self.actions = actions

    def step_wait(self):
        obs,rewards,done,info = self.arena.step(self.actions)
        infos = [{} for _ in range(self.num_envs)]
        infos[0].update(info)  # Count each team episode once.
        if done:
            for i in range(self.num_envs): infos[i]["terminal_observation"] = obs[i].copy()
            obs = self.arena.reset()
        return obs,rewards,np.full(self.num_envs,done),infos

    def close(self): self.arena.close()
    def get_attr(self, attr_name, indices=None):
        return [getattr(self,attr_name,None)]*len(self._get_indices(indices))
    def set_attr(self, attr_name, value, indices=None): setattr(self,attr_name,value)
    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self.arena,method_name)(*args,**kwargs)]
    def env_is_wrapped(self, wrapper_class, indices=None): return [False]*len(self._get_indices(indices))
