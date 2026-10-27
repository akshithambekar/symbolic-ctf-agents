"""Inspect benchmark cases. Live translation and scoring are not implemented yet."""

import argparse
import json
from pathlib import Path


DEFAULT_CASES = Path(__file__).with_name("cases.json")


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list):
        raise ValueError("The cases file must contain a JSON list.")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="Path to a JSON cases file")
    parser.add_argument("--list", action="store_true", help="List cases without calling a model")
    args = parser.parse_args()

    if not args.list:
        parser.error("Live benchmark execution is not implemented yet; use --list to inspect cases.")

    for case in load_cases(args.cases):
        print(f"{case['id']}: {case['instruction']} -> {case['reference_formula']}")


if __name__ == "__main__":
    main()
