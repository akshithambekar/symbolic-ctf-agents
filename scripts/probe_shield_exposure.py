"""Exploratory calibration of valid moving boundary starts; preserve every probe."""
import hashlib
import json
import random
import shutil
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maritime_nsrl.research_env import ArenaConfig, TeamArena
from maritime_nsrl.runtime_shield_study import AuditedShield, csv_rows, save


def run(output, policy):
    output = Path(output); policy = Path(policy)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = PPO.load(policy, device="cpu")
    values = json.loads((policy.parent / "arena_config.json").read_text(encoding="utf-8"))
    assert not any(values[k] for k in ("guided", "reward_shaping", "temporal_state", "shield"))
    shutil.copy2(__file__, output / "probe_source.py")
    save(output / "protocol.json", {"type": "post-hoc exploratory stress calibration", "policy": str(policy),
        "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(), "seed": 861100,
        "initial_speeds": [1., 1.5, 2.], "boundary_margins": [2.01, 2.5, 3.],
        "selection_rule": "first grid case with unshielded bounds violation, >=1 substitution, and no shield deadlock; rerun on fresh evaluation seeds and both checkpoints",
        "note": "Calibration outcomes are not pooled with the final evaluation. Severe and unsuccessful probes remain recorded."})
    save(output / "source_hashes.json", {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "src").rglob("*.py")})
    rows = []
    with (output / "traces.jsonl").open("w", encoding="utf-8") as trace:
        for speed, margin in product((1., 1.5, 2.), (2.01, 2.5, 3.)):
            for enabled in (False, True):
                config = ArenaConfig(**values)
                config.start_mode = "boundary_lower"; config.boundary_start_margin = margin
                config.boundary_start_speed = speed; config.shield = enabled
                arena = TeamArena(config)
                random.seed(861100); np.random.seed(861100); torch.manual_seed(861100)
                trajectory = []
                try:
                    obs = arena.reset(861100)
                    if enabled: arena.gate = AuditedShield(arena)
                    while not arena.finished:
                        actions, _ = model.predict(obs, deterministic=True)
                        if enabled: arena.gate.waypoints = list(map(int, actions))
                        obs, _, _, info = arena.step(actions)
                        trajectory.append({"time": float(arena.game.current_time), "positions": arena.game.state["agent_position"].tolist(), "actions": list(map(int, actions))})
                    metrics = info["metrics"]; metrics.pop("trajectory", None)
                    rows.append({"speed": speed, "margin": margin, "shield": enabled, **metrics})
                    trace.write(json.dumps({"config": asdict(config), "trajectory": trajectory, "events": arena.gate.events if enabled else []}) + "\n")
                    print(f"probe v={speed} margin={margin} gate={enabled}: OOB={metrics['out_of_bounds_steps']} interventions={metrics['shield_interventions']} deadlocks={metrics['shield_deadlocks']} captures={metrics['captures']}", flush=True)
                finally: arena.close()
    csv_rows(output / "episodes.csv", rows); save(output / "episodes.json", rows)
    eligible = []
    for off, on in zip(rows[::2], rows[1::2]):
        if off["out_of_bounds_steps"] > 0 and on["shield_interventions"] > 0 and on["shield_deadlocks"] == 0:
            eligible.append({"speed": off["speed"], "margin": off["margin"]})
    save(output / "selection.json", {"eligible": eligible, "selected": eligible[0] if eligible else None})


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    a = p.parse_args(); run(a.output, a.policy)
