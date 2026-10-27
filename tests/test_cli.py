import json
import sys

import pytest

from maritime_nsrl.cli import main


def test_translate_validate_and_protect_existing_spec(tmp_path, monkeypatch, capsys):
    instruction = tmp_path / "strategy.txt"
    instruction.write_text(
        "Always stay within the arena. Pick up the enemy flag and then capture it."
    )
    output = tmp_path / "spec.json"
    monkeypatch.setattr(
        sys, "argv", ["mctf", "translate", str(instruction), "--output", str(output)]
    )
    main()
    saved = output.read_bytes()
    assert json.loads(saved)["ltl"] == "G(in_bounds) & F(has_flag & F(captured))"
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["mctf", "validate", str(output)])
    main()
    assert json.loads(capsys.readouterr().out)["supported_fragment"]
    monkeypatch.setattr(
        sys, "argv", ["mctf", "translate", str(instruction), "--output", str(output)]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert output.read_bytes() == saved


def test_invalid_training_budget_creates_no_run(tmp_path, monkeypatch):
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        ["mctf", "train", "--spec", "unused.json", "--output", str(output), "--steps", "0"],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not output.exists()
