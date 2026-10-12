"""Fresh, staged exploratory screens; no previous result directories are written."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import random
import shutil
import subprocess
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maritime_nsrl.followup import Exposure, ExposureArena, ChaseTrainingArena, native_restart
from maritime_nsrl.research_env import ArenaConfig, SharedTeamVecEnv
from maritime_nsrl.runtime_shield_study import csv_rows, save
from maritime_nsrl.tactics import TacticalArena, TacticalPolicy, TacticsConfig, role_contract
from maritime_nsrl.attacker_training import evaluate_attacker
from maritime_nsrl.experiments import EpisodeLog
from maritime_nsrl.viewer import snapshot

SOURCE = ROOT / "experiments/tactical-attacker-20261012-v1/attackers"
EXPOSURES = {
    "ordinary": Exposure(),
    "approach_tight": Exposure("approach", 2.01, 1.),
    "approach_reserve": Exposure("approach", 6., 2.),
    "return_tight": Exposure("return", 2.01, 1.),
    "return_reserve": Exposure("return", 6., 2.),
    "return_severe": Exposure("return", 2.01, 2.),
}
ARMS = ["off", "original", "brake", "proactive"]


def load(seed):
    path = SOURCE / f"seed{seed}"
    config = ArenaConfig(**json.loads((path / "arena_config.json").read_text(encoding="utf-8")))
    return PPO.load(path / "policy.zip", device="cpu"), config


def initialize(output):
    output.mkdir(parents=True, exist_ok=False)
    snapshot_dir = output / "source"
    shutil.copytree(ROOT / "src", snapshot_dir / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(__file__, snapshot_dir / "runner.py")
    save(output / "source_hashes.json", {p.relative_to(snapshot_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot_dir.rglob("*") if p.is_file()})
    save(output / "protocol.json", {
        "study": "exploratory tactical/offensive follow-up", "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_policies": {str(seed): hashlib.sha256((SOURCE / f"seed{seed}/policy.zip").read_bytes()).hexdigest() for seed in (0, 1)},
        "exposures": {name: asdict(value) for name, value in EXPOSURES.items()}, "shield_arms": ARMS,
        "exposure_episodes_per_seed_cell": 1, "training_seeds": [0, 1], "opponents": ["static", "patrol", "chase"],
        "roles": ["none", "fixed", "conditional", "dynamic"], "role_attack_styles": ["split_flank", "escort"], "role_episodes_per_cell": 3,
        "chase_variants": ["task", "progress", "directional"], "chase_training_speeds": [.65, .85, 1.],
        "chase_decisions_per_stage": 2048, "promotion_rule_defined_before_runs": True,
        "promotion_rule": "Advance to next training speed only if five held-out current-speed episodes give >=20% pickups AND five static episodes give >=60% captures. Final candidate requires full-speed held-out pickups >=20% and static captures >=60%; larger comparisons are not automatic.",
        "paired_evaluation": "same checkpoint/RNG/init per off/on exposure; role attack style held fixed within comparison; chase evaluation seeds matched across variants",
        "limits": ["two training seeds", "scripted roles are not learned MARL", "return starts initialize possession and cannot prove flag acquisition", "proactive reserve is heuristic, not a safety certificate", "additional controls are shield actions, not learned policy actions"],
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "torch", "stable-baselines3", "gymnasium", "pygame"]},
    })
    simulator = Path(__import__("pyquaticus").__file__).parent
    save(output / "simulator_hashes.json", {p.relative_to(simulator).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in simulator.rglob("*.py")})


def exposures(output):
    directory = output / "exposure"; directory.mkdir()
    rows = []
    with (directory / "traces.jsonl").open("x", encoding="utf-8") as stream:
        for seed in (0, 1):
            model, original = load(seed)
            for cell_index, (name, opponent) in enumerate(product(EXPOSURES, ("static", "patrol", "chase"))):
                evaluation_seed = 910000 + seed*100 + cell_index
                for arm in ARMS:
                    config = copy.deepcopy(original); config.opponent = opponent; config.shield = arm != "off"
                    arena = ExposureArena(config, EXPOSURES[name], recovery=arm if arm in {"brake", "proactive"} else False)
                    random.seed(evaluation_seed); np.random.seed(evaluation_seed); torch.manual_seed(evaluation_seed)
                    frames = []
                    try:
                        obs = arena.reset(evaluation_seed)
                        initial = snapshot(arena, [], 0, True, 0)
                        while not arena.finished:
                            actions, _ = model.predict(obs, deterministic=True)
                            if arena.gate: arena.gate.waypoints = list(map(int, actions))
                            caps = int(arena.game.state["captures"][0])
                            obs, _, _, info = arena.step(actions)
                            frames.append(snapshot(arena, actions, 0, True, caps))
                        metrics = info["metrics"]; metrics.pop("trajectory", None)
                        events = arena.gate.events if arena.gate else []
                        row = {"training_seed": seed, "condition": name, "arm": arm, "evaluation_seed": evaluation_seed,
                            "initial_carrier": EXPOSURES[name].stage == "return", **metrics,
                            "offensive_events": sum(e["enemy_territory"] for e in events if not e["deadlock"]),
                            "carrier_events": sum(e["carrying_flag"] for e in events if not e["deadlock"]),
                            "near_flag_events": sum(e["near_enemy_flag"] for e in events if not e["deadlock"])}
                        rows.append(row)
                        stream.write(json.dumps({"identity": {k: row[k] for k in ["training_seed", "condition", "opponent", "arm", "evaluation_seed"]},
                            "config": asdict(config), "exposure": asdict(EXPOSURES[name]), "initial": initial, "frames": frames, "events": events})+"\n")
                        stream.flush()
                    finally: arena.close()
                print(f"exposure seed{seed} {name}/{opponent}", flush=True)
    csv_rows(directory / "episodes.csv", rows); save(directory / "episodes.json", rows)


def roles(output):
    directory = output / "roles"; directory.mkdir()
    rows = []
    for style, mode, opponent in product(("split_flank", "escort"), ("none", "fixed", "conditional", "dynamic"), ("static", "patrol", "chase", "raid")):
        contract = role_contract(mode)
        config = ArenaConfig(team_size=3, guided=False, reward_shaping=False, temporal_state=False, shield=False,
            formula=contract["ltl"], opponent="static" if opponent == "raid" else opponent)
        tactics = TacticsConfig(mode, style, "raid" if opponent == "raid" else "standard")
        cell = directory / f"{style}-{mode}-{opponent}"; cell.mkdir()
        save(cell / "config.json", {"arena": asdict(config), "tactics": asdict(tactics), "specification": contract})
        arena = TacticalArena(config, tactics); policy = TacticalPolicy()
        try:
            for ep in range(3):
                arena.reset(920000+ep)
                with (cell / f"episode-{ep}.jsonl").open("x", encoding="utf-8") as trace:
                    while not arena.finished:
                        actions = policy(arena); caps = int(arena.game.state["captures"][0])
                        _, _, _, info = arena.step(actions)
                        trace.write(json.dumps(snapshot(arena, actions, ep, True, caps))+"\n")
                metrics = info["metrics"]; metrics.pop("trajectory", None)
                rows.append({"attack_style": style, "role_mode": mode, "tactical_opponent": opponent, "evaluation_seed": 920000+ep, **metrics})
        finally: arena.close()
        print(f"matched roles {style}/{mode}/{opponent}", flush=True)
    csv_rows(directory / "episodes.csv", rows); save(directory / "episodes.json", rows)


def conflicts(output):
    directory = output / "conflicts"; directory.mkdir()
    rows=[]
    for mode, reassign in product(("fixed", "conditional", "dynamic"), (False, True)):
        contract = role_contract(mode)
        config = ArenaConfig(team_size=3, guided=False, reward_shaping=False, temporal_state=False, shield=False,
            formula=contract["ltl"], opponent="static", horizon=60.)
        tactics = TacticsConfig(mode, "escort", "raid", reassign)
        arena = TacticalArena(config, tactics); policy = TacticalPolicy()
        try:
            arena.reset(930000)
            positions = arena.game.state["agent_position"].tolist()
            positions[0] = [.8*arena.size[0], .25*arena.size[1]]
            positions[3] = [.45*arena.size[0], .75*arena.size[1]]
            init = {"agent_position": positions, "agent_heading": [270.]*3+[270.]*3,
                "agent_speed": [0.]*6, "agent_has_flag": [True,False,False,False,False,False]}
            native_restart(arena, init)
            initial_roles = arena.roles()
            with (directory / f"{mode}-{reassign}.jsonl").open("x", encoding="utf-8") as trace:
                trace.write(json.dumps({"init_dict": init, "initial_roles": initial_roles, "initial_labels": sorted(arena.labels()),
                    "config": asdict(config), "tactics": asdict(tactics), "specification": contract})+"\n")
                while not arena.finished:
                    actions=policy(arena); caps=int(arena.game.state["captures"][0])
                    _,_,_,info=arena.step(actions)
                    trace.write(json.dumps(snapshot(arena,actions,0,True,caps))+"\n")
            metrics=info["metrics"];metrics.pop("trajectory",None)
            rows.append({"role_mode":mode,"carrier_reassignment":reassign,"initial_roles":initial_roles, **metrics})
        finally: arena.close()
    csv_rows(directory / "episodes.csv", rows);save(directory / "episodes.json",rows)


def chase(output):
    directory=output/"chase";directory.mkdir()
    rows=[]
    for seed in (0, 1):
        model, config = load(seed)
        cell = directory / f"initial{seed}"; cell.mkdir()
        config.opponent = "chase"
        result = evaluate_attacker(model, config, cell/"full_chase.csv", 10, 950000+seed*100)
        config.opponent = "static"
        retained = evaluate_attacker(model, config, cell/"final_static.csv", 10, 951000+seed*100)
        rows.append({"training_seed":seed,"variant":"unmodified","actual_decisions":0,"completed_curriculum":False,
            "full_speed_pickup_rate":result["pickup_rate"],"full_speed_capture_rate":result["capture_rate"],
            "static_capture_rate":retained["capture_rate"],"candidate":False})
    for seed,variant in product((0,1),("task","progress","directional")):
        model,config=load(seed);config.opponent="chase"
        cell=directory/f"{variant}{seed}";cell.mkdir()
        save(cell/"config.json", {"arena":asdict(config),"variant":variant,"reward_progress_per_metre":0. if variant=="task" else .1,
            "evasion_per_metre":.05 if variant=="directional" else 0.,"gate":"positive progress and opponent within25m; disabled when tagged",
            "initial_policy_sha256":hashlib.sha256((SOURCE/f"seed{seed}/policy.zip").read_bytes()).hexdigest(),"training_speeds":[.65,.85,1.],"decisions_per_stage":2048})
        actual=0;promoted=True
        for stage,speed in enumerate((.65,.85,1.)):
            cfg=copy.deepcopy(config);cfg.opponent_speed=speed
            arena=ChaseTrainingArena(cfg,variant);vec=SharedTeamVecEnv(arena)
            model.set_env(vec);model.set_random_seed(seed)
            log=cell/f"stage{stage}";model.set_logger(configure(str(log),["csv"]))
            try: model.learn(total_timesteps=2048,callback=EpisodeLog(log/"episodes.jsonl"),reset_num_timesteps=True)
            finally:vec.close()
            actual+=model.num_timesteps;model.save(cell/f"stage{stage}.zip")
            screen=evaluate_attacker(model,cfg,cell/f"screen{stage}.csv",5,940000+stage*100+seed*10)
            static=copy.deepcopy(config);static.opponent="static"
            retention=evaluate_attacker(model,static,cell/f"static{stage}.csv",5,941000+stage*100+seed*10)
            passed=screen["pickup_rate"]>=.2 and retention["capture_rate"]>=.6
            save(log/"promotion.json",{"screen":screen,"static_retention":retention,"passed":passed,"speed":speed})
            print(f"chase {variant}/seed{seed} speed{speed}: pickup={screen['pickup_rate']}, static={retention['capture_rate']}, advance={passed}",flush=True)
            if not passed:promoted=False;break
        model.save(cell/"policy.zip")
        full=copy.deepcopy(config);full.opponent_speed=1.
        result=evaluate_attacker(model,full,cell/"full_chase.csv",10,950000+seed*100)
        final_static=copy.deepcopy(config);final_static.opponent="static"
        retained=evaluate_attacker(model,final_static,cell/"final_static.csv",10,951000+seed*100)
        row={"training_seed":seed,"variant":variant,"actual_decisions":actual,"completed_curriculum":promoted,
            "full_speed_pickup_rate":result["pickup_rate"],"full_speed_capture_rate":result["capture_rate"],"static_capture_rate":retained["capture_rate"],
            "candidate":promoted and result["pickup_rate"]>=.2 and retained["capture_rate"]>=.6}
        rows.append(row);save(cell/"status.json",row)
    csv_rows(directory/"summary.csv",rows);save(directory/"summary.json",rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    initialize(args.output)
    try:
        exposures(args.output);roles(args.output);conflicts(args.output);chase(args.output)
        save(args.output/"status.json",{"status":"completed exploratory; no automatic confirmatory launch"})
    except Exception as exc:
        save(args.output/"status.json",{"status":"failed; raw partial outputs preserved","error":repr(exc)})
        raise
