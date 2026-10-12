"""Run feasibility probes across team sizes, maps, and opponent policies."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from maritime_nsrl.oracle import run_oracle
from maritime_nsrl.research_env import ArenaConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, traces = [], []
    for team_size in (1, 3):
        for scenario in ("standard", "wide", "narrow"):
            for opponent in ("static", "patrol", "chase"):
                config = ArenaConfig(team_size=team_size, scenario=scenario,
                    opponent=opponent, horizon=240, formula="F(captured)")
                for row in run_oracle(config, args.episodes, trace_stride=1):
                    trace = row.pop("trajectory")
                    row.update(team_size=team_size, scenario=scenario, opponent=opponent)
                    rows.append(row)
                    traces.append({"team_size":team_size,"scenario":scenario,
                        "opponent":opponent,"evaluation_seed":row["evaluation_seed"],
                        "trajectory":trace})
                print(f"oracle {team_size}v{team_size} {scenario} {opponent}",flush=True)
    with (args.output/"episodes.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (args.output/"trajectories.jsonl").open("w",encoding="utf-8") as f:
        for record in traces: f.write(json.dumps(record,separators=(",",":"))+"\n")
    summary=[]
    for team_size in (1,3):
        for scenario in ("standard","wide","narrow"):
            for opponent in ("static","patrol","chase"):
                group=[r for r in rows if r["team_size"]==team_size and
                       r["scenario"]==scenario and r["opponent"]==opponent]
                rate=lambda key: float(np.mean([bool(r[key]) for r in group]))
                summary.append({"team_size":team_size,"scenario":scenario,"opponent":opponent,
                    "episodes":len(group),"enemy_territory_rate":rate("reached_enemy_territory"),
                    "enemy_flag_rate":rate("reached_enemy_flag"),
                    "pickup_rate":float(np.mean([r["pickups"]>0 for r in group])),
                    "return_midfield_rate":rate("return_crossed_midfield"),
                    "capture_rate":float(np.mean([r["captures"]>0 for r in group])),
                    "mean_tags":float(np.mean([r["tags_received"] for r in group]))})
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")


if __name__ == "__main__": main()
