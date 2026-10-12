"""Validate pairing and produce a post-run exploratory analysis without editing raw data."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from itertools import product
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maritime_nsrl.runtime_shield_study import csv_rows, save, summarize


def analyze(directory):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
    has_starts = "start_mode" in rows[0]
    starts = protocol.get("start_modes", ["standard"])
    keys = ("training_seed", "scenario", "opponent", "evaluation_seed") + (("start_mode",) if has_starts else ())
    pairs = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if row["arm"] in pairs.setdefault(key, {}): raise ValueError("Duplicate episode")
        pairs[key][row["arm"]] = row
    expected = len(protocol["training_seeds"]) * len(protocol["evaluation_seeds"]) * len(protocol["scenarios"]) * len(protocol["opponents"]) * len(starts)
    assert len(pairs) == expected and len(rows) == expected * 2
    assert all(set(pair) == set(protocol["arms"]) for pair in pairs.values())
    for seed, scenario, opponent, start_mode in product(protocol["training_seeds"], protocol["scenarios"], protocol["opponents"], starts):
        prefix = directory / "configs" / f"seed{seed}" / f"{scenario}-{opponent}"
        if protocol.get("config_layout") == "with-start-mode": prefix = Path(str(prefix) + f"-{start_mode}")
        off = json.loads(Path(str(prefix) + "-unshielded.json").read_text(encoding="utf-8"))
        on = json.loads(Path(str(prefix) + "-runtime_bounds_shield.json").read_text(encoding="utf-8"))
        assert {**off, "shield": True} == on
    traces = [json.loads(line) for line in (directory / "traces.jsonl").read_text().splitlines()]
    by_trace = {(tuple(r[k] for k in keys), r["arm"]): r for r in traces}
    identical = 0
    for key in pairs:
        off = by_trace[(key, "unshielded")]
        on = by_trace[(key, "runtime_bounds_shield")]
        if not on["shield_events"]:
            assert off["trajectory"] == on["trajectory"], "Dormant gate changed trajectory"
            identical += 1
    totals = {arm: summarize([r for r in rows if r["arm"] == arm]) for arm in protocol["arms"]}
    cells = []
    seed_rows = []
    for scenario, opponent, start_mode, arm in product(protocol["scenarios"], protocol["opponents"], starts, protocol["arms"]):
        subset = [r for r in rows if (r["scenario"], r["opponent"], r.get("start_mode", "standard"), r["arm"]) == (scenario, opponent, start_mode, arm)]
        cells.append({"scenario": scenario, "opponent": opponent, "start_mode": start_mode, "arm": arm, **summarize(subset)})
    for seed in protocol["training_seeds"]:
        subsets = {arm: summarize([r for r in rows if r["training_seed"] == seed and r["arm"] == arm]) for arm in protocol["arms"]}
        seed_rows.append({"training_seed": seed, **{f"delta_{k}": subsets[protocol["arms"][1]][k] - subsets[protocol["arms"][0]][k]
            for k in ("capture_rate", "pickup_rate", "tag_rate", "out_of_bounds_episode_rate", "mean_common_reward", "mean_duration")}})
    csv_rows(directory / "cell_summary.csv", cells)
    csv_rows(directory / "seed_paired_summary.csv", seed_rows)
    lines = ["# Post-run analysis: paired runtime-only bounds shield", "",
        f"Completed {len(rows)} episodes, {expected} matched pairs, {len(protocol['training_seeds'])} independently trained checkpoints. Both arms use the same saved unshielded neural checkpoints (see input config.json for the training method), deterministic action selection, seeds, maps, opponents, and evaluation horizon; only the runtime gate differs. No retraining or guidance features were introduced.", "",
        "| Map | Opponent | Starts | Arm | Captures / episodes | Pickup % | Territory % | Tag % | OOB % | Deadlock % | Common reward |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in cells:
        lines.append(f"| {s['scenario']} | {s['opponent']} | {s['start_mode']} | {s['arm']} | {s['captures']}/{s['episodes']} | {100*s['pickup_rate']:.1f} | {100*s['enemy_territory_rate']:.1f} | {100*s['tag_rate']:.1f} | {100*s['out_of_bounds_episode_rate']:.1f} | {100*s['deadlock_rate']:.1f} | {s['mean_common_reward']:.3f} |")
    on = totals["runtime_bounds_shield"]
    lines += ["", "## Intervention evidence", "", f"The shield performed {on['checks']} successor checks, substituted {on['interventions']} actions, and recorded {on['offensive_interventions']} interventions in enemy-territory/carrier states, {on['near_flag_interventions']} near the flag, and {on['carrying_interventions']} while carrying. {identical}/{expected} pairs had no shield events and identical recorded action/state trajectories."]
    if on["interventions"] == 0 and on["deadlock_rate"] == 0:
        lines += ["", "The shield was dormant throughout this sample. The checkpoints already avoided the checked bounds failures. Consequently this study does not estimate how shielding changes behavior when an unsafe proposal occurs. It establishes no safety advantage, no offensive benefit, and no shield-caused suppression of offensive actions."]
    lines += ["", "## Limits and next useful test", "", "This is exploratory with two training seeds. Held-out episodes are paired observations within training replicates, not extra independent training seeds. The training method and opponent are preserved in inputs/. Any new map, opponent, or boundary start is a transfer/stress condition; boundary starts are engineered exposure probes and are not pooled with ordinary starts for task-performance claims. Capture time is computed only among successful episodes. Deadlocks and numerical termination are explicitly recorded because shorter exposure can improve common reward without improving control.", "", "Before spending more evaluation budget, verify offensive capability in the saved checkpoint and inspect whether it ever proposes unsafe boundary actions. Reuse that same checkpoint and held-out initial conditions with the gate off/on. A separate, clearly labeled boundary-stress evaluation could assess intervention effectiveness; it would not replace task-level performance testing.", "", "## Integrity", "", "All expected pairs and configurations were checked. Frozen source and simulator hashes, saved model weights and training configurations, raw metrics, trajectories, and intervention events accompany this analysis. Earlier Phase 2/3 results are unchanged. Timing is diagnostic wall-clock measurement, not a controlled performance benchmark."]
    (directory / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(__file__, directory / "analysis_source.py")
    save(directory / "analysis_provenance.json", {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pair_count": expected, "identical_dormant_pairs": identical, "configuration_pairing_verified": True})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, metric, label in zip(axes, ("capture_rate", "enemy_territory_rate", "out_of_bounds_episode_rate"), ("Capture rate", "Enemy-territory rate", "Out-of-bounds rate")):
        for arm_index, arm in enumerate(protocol["arms"]):
            for index, seed in enumerate(protocol["training_seeds"]):
                value = summarize([r for r in rows if r["arm"] == arm and r["training_seed"] == seed])[metric]
                ax.scatter(arm_index + (index - .5) * .09, value, color=f"C{index}", marker=("o", "s")[index % 2], label=f"training seed {seed}" if arm_index == 0 else None)
        ax.set_xticks([0, 1], ["Unshielded", "Runtime shield"])
        ax.set_ylim(-.04, 1.04); ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    fig.suptitle("Exploratory paired study: separate points for two training seeds")
    fig.tight_layout(); fig.savefig(directory / "seed_comparison.png", dpi=200); plt.close(fig)
    print(json.dumps({"totals": totals, "identical_pairs": identical}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    analyze(parser.parse_args().directory)
