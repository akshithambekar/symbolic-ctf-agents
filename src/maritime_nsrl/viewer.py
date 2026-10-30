"""PyQuaticus rendering of live states and recorded snapshots; no new physics."""

from dataclasses import fields
import json
import hashlib
from pathlib import Path
import random
import time
import numpy as np

ACTION_NAMES = [
    "flag",
    "home",
    "low flank",
    "high flank",
    "low return",
    "high return",
    "evade",
    "hold",
    "upper probe",
]
SINGLE_ACTION_NAMES = [
    "flag",
    "home",
    "low midfield",
    "high midfield",
    "low forward",
    "high forward",
    "hold",
]


def load_policy_environment(policy, scenario=None, opponent=None, team_size=None, config_path=None):
    from .research_env import ArenaConfig, TeamArena
    from .environment import MaritimeEnv
    from .translation import load_spec
    from .specifications import read_artifact

    if policy == "oracle":
        from .oracle import CaptureOracle

        values = json.loads(Path(config_path).read_text()) if config_path else {}
        values.update(
            team_size=team_size or values.get("team_size", 1),
            scenario=scenario or values.get("scenario", "standard"),
            opponent=opponent or values.get("opponent", "static"),
        )
        values.setdefault("formula", "G(in_bounds) & F(has_flag & F(captured))")
        return TeamArena(ArenaConfig(**values)), CaptureOracle(values["team_size"]), True
    from stable_baselines3 import PPO

    policy = Path(policy).resolve()
    config = json.loads((policy.parent / "config.json").read_text())
    if config.get("mode") == "team":
        stored = policy.parent / "arena_config.json"
        values = (
            json.loads(stored.read_text())
            if stored.exists()
            else {
                key: value
                for key, value in config.items()
                if key in {f.name for f in fields(ArenaConfig)}
            }
        )
        if config_path:
            values = json.loads(Path(config_path).read_text())
        if team_size is not None and team_size != values["team_size"]:
            raise ValueError("Team-size override incompatible with saved policy observation shape")
        if scenario:
            values["scenario"] = scenario
        if opponent:
            values["opponent"] = opponent
        artifact = read_artifact(policy.parent / "specification.json")
        if artifact["ltl"] != values["formula"]:
            raise ValueError("Saved formula differs from validated specification")
        env = TeamArena(ArenaConfig(**values))
        is_team = True
    else:
        if team_size not in (None, 1) or scenario not in (None, "standard"):
            raise ValueError("Legacy 1v1 policy supports standard arena and team size 1 only")
        if config_path:
            raise ValueError("Config override supported for TeamArena policies only")
        env = MaritimeEnv(
            load_spec(policy.parent / "specification.json"),
            guided=config["guided"],
            opponent=opponent or config["opponent"],
            horizon=config["horizon"],
            dynamics=config.get("dynamics", "heron"),
            return_start_probability=config.get("return_start_probability", 0.0),
        )
        is_team = False
    model = PPO.load(policy, device="cpu")
    if (
        model.observation_space.shape != env.observation_space.shape
        or model.action_space.n != env.action_space.n
    ):
        env.close()
        raise ValueError("Policy spaces differ from environment; use matching saved config")
    return env, model, is_team


def snapshot(env, actions, episode, is_team, previous_captures=None):
    game = env.game
    state = game.state
    team_size = env.n if is_team else 1
    if is_team:
        labels = env.labels(previous_captures=previous_captures)
        monitor = env.monitor
        dfa = monitor.state
        violated = monitor.impossible
        satisfied = monitor.satisfied
    else:
        from .guidance import labels_from_state

        labels = labels_from_state(game)
        monitor = env.guidance.monitor
        dfa = monitor.stage
        violated = monitor.violated
        satisfied = monitor.satisfied
    if state["captures"][0] > 0:
        stage = "capture"
    elif any(state["agent_has_flag"][:team_size]):
        stage = "return"
    else:
        stage = "approach"
    return dict(
        version=1,
        episode=episode,
        time=float(game.current_time),
        team_size=team_size,
        arena_size=np.asarray(game.env_size).tolist(),
        positions=np.asarray(state["agent_position"]).tolist(),
        headings=np.asarray(state["agent_heading"]).tolist(),
        speeds=np.asarray(state["agent_speed"]).tolist(),
        has_flag=list(map(bool, state["agent_has_flag"])),
        tagged=list(map(bool, state["agent_is_tagged"])),
        cooldown=np.asarray(state["agent_tagging_cooldown"]).tolist(),
        on_sides=list(map(bool, state["agent_on_sides"])),
        oob=list(map(bool, state["agent_oob"])),
        flag_home=np.asarray(state["flag_home"]).tolist(),
        captures=np.asarray(state["captures"]).tolist(),
        tags=np.asarray(state["tags"]).tolist(),
        actions=list(map(int, np.asarray(actions).reshape(-1))),
        dfa_state=int(dfa),
        propositions=sorted(labels),
        contract_violated=bool(violated),
        satisfied=bool(satisfied),
        stage=stage,
        executed_actions=(env.trace[-1]["executed_actions"] if is_team and env.trace else None),
        shield_interventions=int(getattr(getattr(env, "gate", None), "interventions", 0)),
        shield_deadlocks=int(getattr(getattr(env, "gate", None), "deadlocks", 0)),
        action_names=ACTION_NAMES[:env.action_space.n] if is_team else SINGLE_ACTION_NAMES,
    )


class Display:
    """Mirror recorded positions/flags into PyQuaticus's existing pygame renderer."""

    def __init__(self, first, headless=False, speed=1.0, record=None):
        import pygame
        from pyquaticus import pyquaticus_v0

        self.pygame = pygame
        self.headless = headless
        self.speed = speed
        self.closed = False
        self.paused = False
        self.last_time = None
        self.frame = 0
        self.record = record
        np_state, py_state = np.random.get_state(), random.getstate()
        try:
            self.game = pyquaticus_v0.PyQuaticusEnv(
                team_size=first["team_size"],
                action_space="continuous",
                render_mode="rgb_array",
                config_dict={
                    "env_bounds": first["arena_size"],
                    "screen_frac": 0.65,
                    "render_agent_ids": True,
                    "render_saving": False,
                    "render_traj_mode": None,
                },
            )
            self.game.reset(seed=0)
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 17)
        self.window = None
        if not headless:
            self.window = pygame.display.set_mode(
                (self.game.screen_width, self.game.screen_height + 155)
            )
            pygame.display.set_caption("Maritime CTF | Space pause | Esc quit")

    def draw(self, frame, partial=False):
        pygame = self.pygame
        game = self.game
        team_size = frame["team_size"]
        # No simulator step: the simulator renderer only displays the recorded state.
        keys = {
            "positions": "agent_position",
            "headings": "agent_heading",
            "speeds": "agent_speed",
            "has_flag": "agent_has_flag",
            "tagged": "agent_is_tagged",
            "cooldown": "agent_tagging_cooldown",
            "on_sides": "agent_on_sides",
            "oob": "agent_oob",
            "captures": "captures",
            "tags": "tags",
        }
        for key, statekey in keys.items():
            if key in frame:
                values = np.asarray(frame[key])
                game.state[statekey][: len(values)] = values
        game.current_time = frame["time"]
        for i, player in enumerate(game.players.values()):
            if i >= len(frame["positions"]):
                player.pos = np.array([-1000.0, -1000.0])
                continue
            player.pos = np.asarray(frame["positions"][i], dtype=float)
            for key, attribute in [
                ("headings", "heading"),
                ("speeds", "speed"),
                ("has_flag", "has_flag"),
                ("tagged", "is_tagged"),
                ("cooldown", "tagging_cooldown"),
                ("on_sides", "on_own_side"),
                ("oob", "oob"),
            ]:
                if key in frame and i < len(frame[key]):
                    setattr(player, attribute, frame[key][i])
        for team, flag in enumerate(game.flags):
            if "flag_home" in frame:
                flag.home = np.asarray(frame["flag_home"][team])
                flag.pos = flag.home.copy()
            carriers = [
                i
                for i, v in enumerate(frame.get("has_flag", []))
                if v and (i < team_size) != (team == 0)
            ]
            flag.taken = bool(carriers)
        game.render()
        surface = pygame.Surface((game.screen_width, game.screen_height + 155))
        surface.fill((20, 26, 34))
        surface.blit(game.screen, (0, 0))
        roles = frame.get("roles")
        for i, pos in enumerate(frame["positions"]):
            if i >= team_size:
                role = f"red {i - team_size}"
            elif roles:
                role = f"blue {i} / {frame['roles'][i]}"
            elif team_size > 1 and i == 0:
                role = "agent 0 / designated defender"
            else:
                role = f"blue {i} / attacker"
            xy = game.env_to_screen(pos)
            surface.blit(self.font.render(role, True, (10, 10, 10)), (xy[0] + 8, xy[1] + 10))
        actions = frame.get("actions", frame.get("requested_actions", []))
        names = frame.get("action_names", ACTION_NAMES)
        action_text = ", ".join(
            f"{i}:{names[a] if 0 <= a < len(names) else a}" for i, a in enumerate(actions)
        )
        if partial:
            footer = (
                "PARTIAL HISTORICAL TRACE: blue positions only; "
                "opponents/headings/cooldowns and flag state incomplete"
            )
        elif roles:
            footer = "Roles: " + str(roles) + " | Space: pause; Esc: quit."
        else:
            footer = (
                "Blue IDs 0..n-1; red IDs n..2n-1. Role label does not enforce a role. "
                "Space: pause; Esc: quit."
            )
        lines = [
            f"Episode {frame.get('episode', '?')}  time {frame['time']:.1f}s "
            f"| captures {frame.get('captures', 'unknown')} | tags {frame.get('tags', 'unknown')}",
            f"Stage {frame.get('stage', 'unknown')} "
            f"| DFA {frame.get('dfa_state', frame.get('monitor_state', 'unknown'))} "
            f"| violated {frame.get('contract_violated', 'unknown')}",
            f"Actions {action_text} | executed {frame.get('executed_actions', 'unknown')} "
            f"| shield {frame.get('shield_interventions', 'unknown')} "
            f"| deadlocks {frame.get('shield_deadlocks', 'unknown')}",
            "Propositions: " + ", ".join(frame.get("propositions", [])),
            footer,
        ]
        for j, line in enumerate(lines):
            surface.blit(
                self.font.render(line, True, (240, 243, 248)), (8, game.screen_height + 8 + j * 27)
            )
        if self.record:
            pygame.image.save(surface, str(self.record / f"frame-{self.frame:06d}.png"))
        self.frame += 1
        if self.window:
            self.window.blit(surface, (0, 0))
            pygame.display.flip()
            delay = (
                0 if self.last_time is None else max(0, frame["time"] - self.last_time) / self.speed
            )
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline or self.paused:
                for event in pygame.event.get():
                    if (
                        event.type == pygame.QUIT
                        or event.type == pygame.KEYDOWN
                        and event.key == pygame.K_ESCAPE
                    ):
                        self.closed = True
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                time.sleep(0.01)
        self.last_time = frame["time"]

    def close(self):
        self.game.close()
        self.pygame.quit()


def watch(args):
    if args.speed <= 0 or args.episodes < 1:
        raise ValueError("Positive speed and episode count required")
    record = Path(args.record) if args.record else None
    if record:
        record.mkdir(parents=True, exist_ok=False)
    env, policy, is_team = load_policy_environment(
        args.policy, args.scenario, args.opponent, args.team_size, args.config
    )
    if is_team:
        env.config.trace_stride = 1  # diagnostic logging only; no policy/physics change
    display = None
    trace = None
    try:
        if record:
            trace = (record / "replay.jsonl").open("x", encoding="utf-8")
            metadata = dict(
                arguments=vars(args),
                viewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                policy_sha256=None
                if args.policy == "oracle"
                else hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(),
            )
            (record / "viewer_config.json").write_text(json.dumps(metadata, default=str, indent=2))
        frames = 0
        for episode in range(args.episodes):
            if is_team:
                obs = env.reset(args.seed + episode)
            else:
                obs, _ = env.reset(seed=args.seed + episode)
            if args.policy != "oracle":
                policy.set_random_seed(args.seed + episode)
            while not env.finished:
                if args.policy == "oracle":
                    actions = policy(env)
                else:
                    actions, _ = policy.predict(obs, deterministic=not args.stochastic)
                previous_captures = int(env.game.state["captures"][0])
                if is_team:
                    obs, _, _, info = env.step(actions)
                else:
                    obs, _, _, _, info = env.step(int(actions))
                frame = snapshot(env, actions, episode, is_team, previous_captures)
                if trace:
                    trace.write(json.dumps(frame) + "\n")
                    trace.flush()
                if display is None:
                    display = Display(frame, args.headless, args.speed, record)
                display.draw(frame)
                frames += 1
                if display.closed or (args.max_frames and frames >= args.max_frames):
                    return
    finally:
        if trace:
            trace.close()
        if display:
            display.close()
        env.close()


def replay(args):
    if args.speed <= 0:
        raise ValueError("Playback speed must be positive")
    from .research_env import SCENARIOS

    records = []
    with Path(args.trajectory).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError("Empty trace")
    partial = False
    if "trajectory" in records[0]:
        if args.episode < 0 or args.episode >= len(records):
            raise ValueError("Episode index out of range")
        episode = records[args.episode]
        frames = episode["trajectory"]
        partial = True
        if not frames:
            raise ValueError("Empty episode")
        for frame in frames:
            frame.update(
                team_size=len(frame["positions"]),
                arena_size=list(SCENARIOS[episode.get("scenario", "standard")]),
                episode=args.episode,
            )
    else:
        frames = [r for r in records if r.get("episode", 0) == args.episode]
    if not frames:
        raise ValueError("No frames for selected episode")
    record = Path(args.record) if args.record else None
    if record:
        record.mkdir(parents=True, exist_ok=False)
    display = Display(frames[0], args.headless, args.speed, record)
    try:
        for index, frame in enumerate(frames):
            display.draw(frame, partial)
            if display.closed or (args.max_frames and index + 1 >= args.max_frames):
                break
    finally:
        display.close()
