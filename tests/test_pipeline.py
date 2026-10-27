import json
import pytest
from maritime_nsrl.translation import translate
from maritime_nsrl.experiments import train, evaluate, save_json


def test_train_save_reload_evaluate(tmp_path):
    spec = tmp_path / "spec.json"
    save_json(spec, translate("Always stay within the arena. Eventually capture the enemy flag."))
    policy = train(spec, tmp_path / "train", guided=True, steps=16, horizon=10.0, seed=3)
    assert policy.exists()
    assert (
        json.loads((policy.parent / "training_complete.json").read_text())["actual_timesteps"]
        == 512
    )
    output = tmp_path / "evaluation.csv"
    result = evaluate(policy, output, episodes=1)
    assert len(result) == 3 and all(r["episodes"] == 1 for r in result)
    assert output.with_suffix(".traces.json").exists()
    assert output.with_suffix(".trajectories.png").exists()
    with pytest.raises(FileExistsError):
        evaluate(policy, output, episodes=1)


def test_team_train_save_reload_evaluate(tmp_path):
    from maritime_nsrl.experiments import train_team, evaluate_team

    spec = tmp_path / "spec.json"
    save_json(spec, translate("Always stay within the arena. Eventually capture the enemy flag."))
    policy = train_team(
        spec,
        tmp_path / "team",
        team_size=3,
        opponent="static",
        steps=1,
        horizon=5.0,
        seed=3,
        reward_shaping=False,
        temporal_state=False,
    )
    output = tmp_path / "team-evaluation.csv"
    result = evaluate_team(policy, output, episodes=1)
    assert len(result) == 9
    assert all(row["episodes"] == 1 for row in result)
    assert all(row["capture_rate"] == 0 for row in result)
    assert output.is_file()
    with pytest.raises(FileExistsError):
        evaluate_team(policy, output, episodes=1)
