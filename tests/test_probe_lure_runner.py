import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/train_probe_lure_full.py"
SPEC = importlib.util.spec_from_file_location("probe_lure_full_runner", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_selection_requires_full_mission_and_reports_capture_separately():
    def row(captured, satisfied, phase):
        return dict(captures=captured, opponent_captures=0, pickups=captured,
                    ltlf_satisfied=satisfied, numerical_terminations=0, furthest_phase=phase)

    direct_capture = [row(1, False, 0)]
    full_capture = [row(1, True, 6)]
    assert runner.selection_score(full_capture) > runner.selection_score(direct_capture)
    assert runner.summarize(direct_capture)["wins"] == 1
    assert runner.summarize(direct_capture)["full_mission_completions"] == 0
    assert runner.stage_success_rate([row(0, False, 3)], 3) == 1.0
    assert runner.stage_success_rate([row(0, False, 3)], 6) == 0.0


def test_completed_checkpoint_resume_and_manifest_rejection(tmp_path, monkeypatch):
    output = tmp_path / "study"
    argv = ["--output", str(output), "--chunk-steps", "256", "--stage-steps", "512",
            "--validation-episodes", "1", "--eval-episodes", "1", "--horizon", "10"]
    runner.run(runner.arguments([*argv, "--max-chunks", "1"]))
    state = json.loads((output / "state.json").read_text())
    assert state["status"] == "running"
    assert state["total_steps"] == 256
    assert (output / state["checkpoint"] / "policy.zip").is_file()
    assert (output / state["best_checkpoint"] / "arena_config.json").is_file()
    assert not (output / "summary.json").exists()
    log = output / "training_episodes.jsonl"
    with log.open("a") as stream:
        stream.write('{"uncommitted":true}\n')
    runner.run(runner.arguments([*argv, "--resume"]))
    state = json.loads((output / "state.json").read_text())
    assert state["status"] == "stage_budget_exhausted"
    assert state["total_steps"] == 512
    assert log.stat().st_size == state["log_bytes"]
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["episode"] for row in rows] == list(range(1, len(rows) + 1))
    assert rows[-1]["timesteps"] == 512
    report = json.loads((output / "summary.json").read_text())
    assert report["primary_standard_full_speed"]["episodes"] == 1
    with pytest.raises(ValueError, match="manifest differs"):
        runner.run(runner.arguments([*argv, "--resume", "--tag-penalty", "0.0"]))
    initialized = tmp_path / "initialized"
    checkpoint = output / state["checkpoint"] / "policy.zip"
    new_args = ["--output", str(initialized), "--chunk-steps", "256",
                "--stage-steps", "512", "--validation-episodes", "1",
                "--eval-episodes", "1", "--horizon", "10", "--max-chunks", "1",
                "--initialize-from", str(checkpoint), "--opponent-speed", "0.8",
                "--start-phase", "6"]
    learn = runner.PPO.learn

    def interrupt_before_first_chunk(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner.PPO, "learn", interrupt_before_first_chunk)
    runner.run(runner.arguments(new_args))
    initial_state = json.loads((initialized / "state.json").read_text())
    assert initial_state["checkpoint"] is None
    assert initial_state["best_checkpoint"] == "selected/step-000000000"
    monkeypatch.setattr(runner.PPO, "learn", learn)
    runner.run(runner.arguments([*new_args, "--resume"]))
    manifest = json.loads((initialized / "manifest.json").read_text())
    assert manifest["initial_checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert manifest["arena_config"]["opponent_speed"] == 0.8
    assert manifest["curriculum_phases"] == [6, 0]
    initial = initialized / "selected/step-000000000"
    assert json.loads((initial / "config.json").read_text())["initial_checkpoint_sha256"] == (
        manifest["initial_checkpoint_sha256"]
    )
    assert json.loads((initial / "validation.jsonl").read_text())["seed"] == 20000
    rows = [json.loads(line) for line in (initialized / "training_episodes.jsonl").read_text().splitlines()]
    assert rows[-1]["timesteps"] == 768
    initialized_state = json.loads((initialized / "state.json").read_text())
    selected = initialized_state["best_checkpoint"]
    # Neither checkpoint can reach midfield within ten seconds, so an equal
    # development score must retain the initial candidate.
    assert selected == "selected/step-000000000"
    assert initialized_state["best_score"] == [0, 0, 0]
    assert json.loads((initialized / selected / "arena_config.json").read_text())["opponent_speed"] == 1.0


def test_invalid_chunk_budget_rejected(tmp_path):
    with pytest.raises(SystemExit):
        runner.arguments(["--output", str(tmp_path / "study"), "--chunk-steps", "257"])
