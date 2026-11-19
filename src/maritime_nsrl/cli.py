"""Commands for specification validation, PPO training, evaluation, and viewing."""

import argparse
import json
import sys
from pathlib import Path

from .experiments import evaluate, evaluate_team, plot_runs, save_json
from .training import launch_training
from .specifications import check_artifact, compose, read_artifact
from .translation import translate


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be at least 1")
    return number


def positive_float(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("Must be a positive finite number")
    return number


def probability(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Must be between 0 and 1")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    translation = commands.add_parser("translate")
    translation.add_argument("instruction", type=Path)
    translation.add_argument("--backend", choices=("offline", "llm"), default="offline")
    translation.add_argument("--logic-backend", choices=("fragment", "dfa"), default="fragment")
    translation.add_argument("--model")
    translation.add_argument("--base-url")
    translation.add_argument("--output", type=Path, required=True)
    commands.add_parser("validate").add_argument("specification", type=Path)
    composition = commands.add_parser("compose")
    composition.add_argument("--mission", type=Path, required=True)
    composition.add_argument("--constraint", action="append", required=True)
    composition.add_argument("--output", type=Path, required=True)

    for name in ("train", "train-team"):
        training = commands.add_parser(name)
        training.add_argument("--spec", type=Path, required=True)
        training.add_argument("--output", type=Path, required=True)
        training.add_argument("--steps", type=positive_int, default=16384)
        training.add_argument("--horizon", type=positive_float, default=240.0)
        training.add_argument("--seed", type=int, default=0)
        training.add_argument("--opponent", choices=("static", "patrol", "chase"), default="static")
        training.add_argument("--config", type=Path)
        training.add_argument("--resume", type=Path)
        training.add_argument("--eval-interval", type=positive_int, default=10000)
        training.add_argument("--eval-episodes", type=positive_int, default=5)
        training.add_argument("--eval-seed", type=int, default=10000)
        training.add_argument("--no-tensorboard-server", action="store_true")
        if name == "train":
            training.add_argument("--guided", action="store_true")
            training.add_argument("--return-start-probability", type=probability, default=0.0)
            training.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
            training.add_argument("--episodes", type=positive_int)
        else:
            training.add_argument("--team-size", type=int, choices=(1, 2, 3), default=3)
            training.add_argument("--episodes", type=positive_int)
            training.add_argument(
                "--scenario", choices=("standard", "wide", "narrow"), default="standard"
            )
            training.add_argument("--shield", action="store_true")
            training.add_argument("--role-constraint", action="store_true")
            training.add_argument("--no-reward-shaping", action="store_true")
            training.add_argument("--no-temporal-state", action="store_true")

    for name in ("evaluate", "evaluate-team"):
        evaluation = commands.add_parser(name)
        evaluation.add_argument("policy", type=Path)
        evaluation.add_argument("--output", type=Path, required=True)
        evaluation.add_argument("--episodes", type=positive_int, default=10)
        evaluation.add_argument("--seed", type=int, default=10000)

    for name in ("watch", "replay"):
        viewer = commands.add_parser(name)
        if name == "watch":
            viewer.add_argument("policy", help="Saved policy.zip, oracle, or teacher")
            viewer.add_argument("--scenario", choices=("standard", "wide", "narrow"))
            viewer.add_argument("--opponent", choices=("static", "patrol", "chase"))
            viewer.add_argument("--team-size", type=int, choices=(1, 2, 3))
            viewer.add_argument("--opponent-speed", type=float)
            viewer.add_argument("--seed", type=int, default=40000)
            viewer.add_argument("--episodes", type=positive_int, default=1)
            viewer.add_argument("--stochastic", action="store_true")
            viewer.add_argument("--config", type=Path)
        else:
            viewer.add_argument("trajectory", type=Path)
            viewer.add_argument("--episode", type=int, default=0)
        viewer.add_argument("--speed", type=float, default=10.0)
        viewer.add_argument("--headless", action="store_true")
        viewer.add_argument("--record", type=Path)
        viewer.add_argument("--max-frames", type=int, default=0)
    commands.add_parser("plot").add_argument("directory", type=Path)
    argv = sys.argv[1:]
    if argv and argv[0] in ("train", "train-team") and "--config" in argv:
        index = argv.index("--config")
        if index + 1 >= len(argv):
            parser.error("--config requires a JSON file")
        settings = json.loads(Path(argv[index + 1]).read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            parser.error("Training configuration must be a JSON object")
        training_parser = commands.choices[argv[0]]
        by_name = {action.dest: action for action in training_parser._actions}
        configured = []
        for key, value in settings.items():
            action = by_name.get(key.replace("-", "_"))
            if not action or not action.option_strings or key == "config":
                parser.error(f"Unknown training configuration field: {key}")
            option = action.option_strings[-1]
            if isinstance(action, argparse._StoreTrueAction):
                if not isinstance(value, bool):
                    parser.error(f"{key} must be true or false")
                if value:
                    configured.append(option)
            else:
                configured.extend((option, str(value)))
        argv = [argv[0], *configured, *argv[1:]]
    args = parser.parse_args(argv)

    if args.command in ("translate", "compose"):
        if args.output.exists():
            parser.error(f"Output already exists: {args.output}")
        if args.command == "translate":
            result = translate(
                args.instruction.read_text(encoding="utf-8"),
                args.backend,
                args.model,
                args.base_url,
                args.logic_backend,
            )
        else:
            result = compose(read_artifact(args.mission), args.constraint)
        save_json(args.output, result)
    elif args.command == "validate":
        result = check_artifact(read_artifact(args.specification)).report()
    elif args.command in ("train", "train-team"):
        result = str(launch_training(args))
    elif args.command == "evaluate":
        result = evaluate(args.policy, args.output, args.episodes, args.seed)
    elif args.command == "evaluate-team":
        result = evaluate_team(args.policy, args.output, args.episodes, args.seed)
    elif args.command == "plot":
        plot_runs(args.directory)
        return
    else:
        from .viewer import replay, watch

        (watch if args.command == "watch" else replay)(args)
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
