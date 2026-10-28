"""Runs demos with PyQuaticus's existing sample policies."""

import argparse
import copy
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/demo.json")
    parser.add_argument("--example", choices=("random", "attack-defend", "combined"))
    parser.add_argument("--render", choices=("human", "none"))
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--seed", type=int)
    preliminary, _ = parser.parse_known_args()
    settings = json.loads(preliminary.config.read_text())
    if not isinstance(settings, dict) or settings.keys() - {"example", "render", "seconds", "seed"}:
        parser.error("Config supports only example, render, seconds, and seed")
    parser.set_defaults(**settings)
    args = parser.parse_args()
    if args.example not in ("random", "attack-defend", "combined") or args.render not in ("human", "none"):
        parser.error("Invalid example or render setting")
    if type(args.seed) is not int or type(args.seconds) not in (int, float) or not 0 < args.seconds <= 600:
        parser.error("Seed must be an integer; seconds must be greater than zero and at most 600")
    if args.render == "none":
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import numpy as np
    import pygame
    from pyquaticus.config import config_dict_std
    from pyquaticus.envs.pyquaticus import PyQuaticusEnv
    from pyquaticus.base_policies.deprecated.base_attack import BaseAttacker
    from pyquaticus.base_policies.deprecated.base_defend import BaseDefender
    from pyquaticus.base_policies.deprecated.base_combined import Heuristic_CTF_Agent

    config = copy.deepcopy(config_dict_std)
    config.update(gps_env=False, max_time=args.seconds, sim_speedup_factor=10, render_agent_ids=True)
    team_size = 2 if args.example == "combined" else 1
    if args.example == "combined":
        config["dynamics"] = "si"
    env = PyQuaticusEnv(team_size=team_size, config_dict=config,
                       render_mode="human" if args.render == "human" else None)
    try:
        observations, _ = env.reset(seed=args.seed)
        policies = {}
        for index, agent in enumerate(observations):
            env.action_space(agent).seed(args.seed + index)
            if args.example != "random":
                policy = Heuristic_CTF_Agent if args.example == "combined" else BaseAttacker if index == 0 else BaseDefender
                options = {"continuous": False, "mode": "hard" if args.example == "combined" else "medium"}
                if policy != BaseAttacker:
                    options.update(flag_keepout=config["flag_keepout"], catch_radius=env.catch_radius)
                policies[agent] = policy(agent, env.players[agent].team,
                                         env.max_speeds[env.players[agent].idx], env.aquaticus_field_points, **options)
        print(f"{args.example}: {team_size}v{team_size}, seed={args.seed}, limit={args.seconds}s", flush=True)
        for step in range(1, env.max_cycles + 2):
            if args.render == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        print("Demo closed")
                        return
            # These upstream policies consume each agent's unnormalized observation dictionary.
            raw_obs = {agent: env.state_to_obs(agent, normalize=False)[0] for agent in observations}
            actions = {agent: policies[agent].compute_action(raw_obs) if agent in policies
                       else env.action_space(agent).sample() for agent in observations}
            if not all(env.action_space(agent).contains(action) for agent, action in actions.items()):
                raise RuntimeError("Policy returned an invalid discrete action")
            observations, _, terminated, truncated, _ = env.step(actions)
            if not all(np.isfinite(obs).all() for obs in observations.values()):
                raise RuntimeError("Simulator returned non-finite observations")
            if all(terminated[agent] or truncated[agent] for agent in observations):
                break
        else:
            raise RuntimeError("Episode exceeded its configured step limit")
        reason = "terminated" if all(terminated[a] for a in observations) else "time limit"
        print(f"Completed {step} steps, {env.current_time:g} simulated seconds ({reason})")
        for key in ("captures", "grabs", "tags"):
            print(f"{key} (blue/red): {[int(value) for value in env.state[key]]}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
