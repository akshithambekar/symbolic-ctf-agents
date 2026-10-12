import json
import pytest
from maritime_nsrl.translation import translate
from maritime_nsrl.experiments import train, evaluate, save_json


def test_train_save_reload_evaluate(tmp_path):
    spec = tmp_path / "spec.json"
    save_json(spec, translate("Always stay within the arena. Eventually capture the enemy flag."))
    policy = train(spec, tmp_path / "train", guided=True, steps=16, horizon=10., seed=3)
    assert policy.exists()
    assert json.loads((policy.parent / "training_complete.json").read_text())["actual_timesteps"] == 512
    output = tmp_path / "evaluation.csv"
    result = evaluate(policy, output, episodes=1)
    assert len(result) == 3 and all(r["episodes"] == 1 for r in result)
    assert output.with_suffix(".traces.json").exists()
    assert output.with_suffix(".trajectories.png").exists()
    with pytest.raises(FileExistsError): evaluate(policy, output, episodes=1)
