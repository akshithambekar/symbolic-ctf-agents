import copy
import json
import pytest
from maritime_nsrl.translation import translate
from maritime_nsrl.specifications import compose, check_artifact, ROLE_CONSTRAINTS


def mission():
    data = translate("Always stay within the arena. Pick up the enemy flag and then capture it.")
    data.update(
        backend="llm",
        model="mock",
        attempts=[{"response": json.dumps({"ltl": data["ltl"]}), "valid": True}],
    )
    return data


def test_composition_preserves_response_and_ordered_mission():
    source = mission()
    original = copy.deepcopy(source)
    result = compose(source, ROLE_CONSTRAINTS)
    assert source == original == result["mission_translation"]
    assert "F(has_flag & F(captured))" in result["combined_ltl"]
    assert result["ltl"] == result["combined_ltl"] == result["validation"]["formula"]
    assert check_artifact(result).formula == result["ltl"]
    assert compose(result, ROLE_CONSTRAINTS) == result


@pytest.mark.parametrize(
    "location", ["report", "response", "combined", "instruction", "constraints"]
)
def test_tampering_is_rejected(location):
    data = compose(mission(), ROLE_CONSTRAINTS)
    if location == "report":
        data["validation"]["formula"] = "F(captured)"
    if location == "response":
        data["mission_translation"]["attempts"][0]["response"] = '{"ltl":"F(captured)"}'
    if location == "combined":
        data["combined_ltl"] = "F(captured)"
    if location == "instruction":
        data["mission_instruction"] = "Never capture."
    if location == "constraints":
        data["human_constraints"] = []
    with pytest.raises(ValueError, match="provenance mismatch"):
        check_artifact(data)


def test_training_rejects_unrecorded_override_before_creating_run(tmp_path):
    from maritime_nsrl.experiments import train_team

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(mission()))
    with pytest.raises(ValueError, match="overrides break provenance"):
        train_team(spec, tmp_path / "run", formula="F(captured)")
    assert not (tmp_path / "run").exists()


def test_shield_only_has_no_guidance_reward_or_temporal_observation():
    import numpy as np
    from maritime_nsrl.research_env import TeamArena, ArenaConfig

    arena = TeamArena(
        ArenaConfig(
            team_size=3,
            shield=True,
            guided=False,
            reward_shaping=False,
            temporal_state=False,
            horizon=5,
            formula=compose(mission(), ROLE_CONSTRAINTS)["ltl"],
        )
    )
    try:
        obs = arena.reset(5)
        assert np.all(obs[:, -len(arena.automaton.states) :] == 0)
        arena.step([7, 7, 7])
        assert arena.metrics["guidance_reward"] == 0
        assert arena.gate.predictions > 0
    finally:
        arena.close()
