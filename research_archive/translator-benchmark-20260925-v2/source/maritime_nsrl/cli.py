import argparse
import json
from pathlib import Path
from .translation import translate, load_spec
from .experiments import train, evaluate, train_team, evaluate_team, save_json, plot_runs


def main():
    parser = argparse.ArgumentParser(description="Maritime neuro-symbolic RL proof of concept")
    sub = parser.add_subparsers(dest="command", required=True)
    bench=sub.add_parser('benchmark-translators')
    bench.add_argument('--output',type=Path,required=True)
    bench.add_argument('--model',default='gpt-4o-mini-2024-07-18')
    bench.add_argument('--base-url',default='https://api.openai.com/v1')
    bench.add_argument('--trials',type=int,default=3)
    bench.add_argument('--timeout',type=float,default=20)
    bench.add_argument('--logic-backend',choices=['fragment','dfa'],default='dfa')
    for command in ('watch','replay'):
        viewer=sub.add_parser(command)
        if command=='watch':
            viewer.add_argument('policy',help='policy.zip or oracle')
            viewer.add_argument('--scenario',choices=['standard','wide','narrow'])
            viewer.add_argument('--opponent',choices=['static','patrol','chase'])
            viewer.add_argument('--team-size',type=int,choices=[1,2,3])
            viewer.add_argument('--seed',type=int,default=40000)
            viewer.add_argument('--episodes',type=int,default=1)
            viewer.add_argument('--stochastic',action='store_true')
            viewer.add_argument('--config',type=Path,help='Explicit ArenaConfig JSON (must match policy spaces/formula)')
        else:
            viewer.add_argument('trajectory',type=Path)
            viewer.add_argument('--episode',type=int,default=0)
        viewer.add_argument('--speed',type=float,default=10.)
        viewer.add_argument('--headless',action='store_true',help='Render without opening a window')
        viewer.add_argument('--record',type=Path,help='New directory for PNG frames and live replay data')
        viewer.add_argument('--max-frames',type=int,default=0,help='0 plays the entire episode')
    t = sub.add_parser("translate")
    t.add_argument("instruction", type=Path)
    t.add_argument("--backend", choices=["offline","llm"], default="offline")
    t.add_argument("--model")
    t.add_argument("--base-url")
    t.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("validate")
    v.add_argument("specification", type=Path)
    co = sub.add_parser("compose", help="Add explicit human constraints to a validated translation")
    co.add_argument("--mission", type=Path, required=True)
    co.add_argument("--constraint", action="append", required=True)
    co.add_argument("--output", type=Path, required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--spec", type=Path, required=True)
    tr.add_argument("--output", type=Path, required=True)
    tr.add_argument("--guided", action="store_true")
    tr.add_argument("--opponent", choices=["static","patrol","chase"], default="static")
    tr.add_argument("--steps", type=int, default=16384)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--horizon", type=float, default=240.)
    tr.add_argument("--resume", type=Path)
    tr.add_argument("--return-start-probability", type=float, default=0.)
    ev = sub.add_parser("evaluate")
    ev.add_argument("policy", type=Path)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--episodes", type=int, default=10)
    ev.add_argument("--seed", type=int, default=10000)
    ex = sub.add_parser("experiment")
    ex.add_argument("--strategy", type=Path, default=Path("strategies/capture.txt"))
    ex.add_argument("--output", type=Path, required=True)
    ex.add_argument("--steps", type=int, default=16384)
    ex.add_argument("--seeds", type=int, nargs="+", default=[0])
    ex.add_argument("--episodes", type=int, default=10)
    ex.add_argument("--opponent", choices=["static","patrol","chase"], default="static")
    ex.add_argument("--horizon", type=float, default=240.)
    ex.add_argument("--return-start-probability", type=float, default=0.)
    pl = sub.add_parser("plot")
    pl.add_argument("directory", type=Path)
    tt = sub.add_parser("train-team")
    tt.add_argument("--spec", type=Path, required=True)
    tt.add_argument("--output", type=Path, required=True)
    tt.add_argument("--team-size", type=int, choices=[2,3], default=2)
    tt.add_argument("--opponent", choices=["static","patrol","chase"], default="chase")
    tt.add_argument("--scenario", choices=["standard","wide","narrow"], default="standard")
    tt.add_argument("--shield", action="store_true")
    tt.add_argument("--role-constraint", action="store_true", help="Use defender/teammate temporal contract")
    tt.add_argument("--no-reward-shaping", action="store_true")
    tt.add_argument("--no-temporal-state", action="store_true")
    tt.add_argument("--steps", type=int, default=16384)
    tt.add_argument("--seed", type=int, default=0)
    te = sub.add_parser("evaluate-team")
    te.add_argument("policy", type=Path)
    te.add_argument("--output", type=Path, required=True)
    te.add_argument("--episodes", type=int, default=10)
    te.add_argument("--seed", type=int, default=10000)
    mx = sub.add_parser("matrix")
    mx.add_argument("--spec", type=Path, required=True)
    mx.add_argument("--output", type=Path, required=True)
    mx.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    mx.add_argument("--steps", type=int, default=8192)
    mx.add_argument("--guided", action="store_true")
    mx.add_argument("--shield", action="store_true")
    mx.add_argument("--team-size", type=int, choices=[1,2,3], default=1)
    args = parser.parse_args()
    if getattr(args, "episodes", 1) < 1 or getattr(args, "steps", 1) < 1:
        parser.error("episodes and steps must be positive")
    if args.command == 'benchmark-translators':
        from .translator_benchmark.runner import run
        print(run(args.output,args.model,args.base_url,args.trials,args.timeout,args.logic_backend))
    elif args.command == 'watch':
        from .viewer import watch
        watch(args)
    elif args.command == 'replay':
        from .viewer import replay
        replay(args)
    elif args.command == "translate":
        result = translate(args.instruction.read_text(encoding="utf-8"), args.backend, args.model, args.base_url)
        save_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command == "compose":
        from .specifications import read_artifact, compose
        save_json(args.output, compose(read_artifact(args.mission), args.constraint))
    elif args.command == "validate":
        from .specifications import read_artifact, check_artifact
        print(json.dumps(check_artifact(read_artifact(args.specification)).report(), indent=2))
    elif args.command == "train":
        print(train(args.spec, args.output, args.guided, args.opponent, args.steps, args.seed,
                    args.horizon, args.resume, args.return_start_probability))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.policy, args.output, args.episodes, args.seed), indent=2))
    elif args.command == "plot": plot_runs(args.directory)
    elif args.command == "train-team":
        print(train_team(args.spec,args.output,args.team_size,args.opponent,args.scenario,args.shield,args.steps,args.seed,
                         role_constraint=args.role_constraint, reward_shaping=not args.no_reward_shaping,
                         temporal_state=not args.no_temporal_state))
    elif args.command == "evaluate-team":
        print(json.dumps(evaluate_team(args.policy,args.output,args.episodes,args.seed),indent=2))
    elif args.command == "matrix":
        args.output.mkdir(parents=True, exist_ok=False)
        records=[]
        for seed in args.seeds:
            if args.team_size == 1:
                run=args.output/f"seed{seed}"
                policy=train(args.spec,run,args.guided,"chase",args.steps,seed)
                summary=evaluate(policy,run/"evaluation.csv",episodes=5,seed=20000+seed,opponents=("static","patrol","chase"))
            else:
                run=args.output/f"seed{seed}"
                policy=train_team(args.spec,run,args.team_size,"chase","standard",args.shield,args.steps,seed)
                summary=evaluate_team(policy,run/"evaluation.csv",episodes=2,seed=20000+seed,opponents=("static","patrol","chase"),scenarios=("standard","wide","narrow"))
            records.append({"seed":seed,"summary":summary})
        save_json(args.output/"matrix.json",records)
        print(json.dumps(records,indent=2))
    elif args.command == "experiment":
        args.output.mkdir(parents=True, exist_ok=False)
        spec_path = args.output / "specification.json"
        save_json(spec_path, translate(args.strategy.read_text(encoding="utf-8")))
        for seed in args.seeds:
            for guided in [False, True]:
                name = f"{'guided' if guided else 'baseline'}-seed{seed}"
                out = args.output / name
                print(f"Training {name}: {args.steps} decisions against {args.opponent}", flush=True)
                policy = train(spec_path, out, guided, args.opponent, args.steps, seed, args.horizon,
                               return_start_probability=args.return_start_probability)
                result = evaluate(policy, out / "evaluation.csv", args.episodes)
                print(json.dumps({"run": name, "evaluation": result}), flush=True)
        plot_runs(args.output)


if __name__ == "__main__": main()
