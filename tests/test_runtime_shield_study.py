from types import SimpleNamespace
import numpy as np
import pytest

from maritime_nsrl.research_env import ArenaConfig, TeamArena
from maritime_nsrl.runtime_shield_study import AuditedShield, baseline_config, summarize
from maritime_nsrl.shield import PredictiveShield


def test_runtime_arm_changes_only_gate_and_evaluation_context():
    values = dict(guided=False, reward_shaping=False, temporal_state=False, shield=False)
    off = baseline_config(values, "wide", "patrol", False)
    on = baseline_config(values, "wide", "patrol", True)
    assert off.__dict__ | {"shield": True} == on.__dict__
    with pytest.raises(ValueError, match="unguided"):
        baseline_config({**values, "temporal_state": True}, "wide", "patrol", True)


def test_audit_records_preintervention_offensive_context(monkeypatch):
    game = SimpleNamespace(state={"agent_position": np.array([[130., 40.], [100., 60.]]),
        "agent_has_flag": [True, False], "flag_home": np.array([[20., 40.], [140., 40.]])},
        catch_radius=10., current_time=2.)
    arena = SimpleNamespace(n=1, size=np.array([160., 80.]), labels=lambda: {"in_bounds"})
    gate = AuditedShield(arena)
    monkeypatch.setattr(PredictiveShield, "transition", lambda *args: (game, {}, 1))
    gate.waypoints = [0]
    gate.transition(game, [{"agent_0": [1., 90.]}, {"agent_0": [0., 0.]}], lambda g: set())
    event = gate.events[0]
    assert gate.checks == 1 and event["carrying_flag"] and event["enemy_territory"]
    assert event["near_enemy_flag"] and event["candidate_index"] == 1
    assert event["proposed_controls"] != event["substituted_controls"]


def test_gate_and_audit_preserve_safe_unshielded_successors():
    values = dict(guided=False, reward_shaping=False, temporal_state=False, horizon=30.)
    off, on = TeamArena(ArenaConfig(**values)), TeamArena(ArenaConfig(**values, shield=True))
    try:
        np.testing.assert_array_equal(off.reset(seed=991), on.reset(seed=991))
        on.gate = AuditedShield(on)
        for _ in range(3):
            obs_off, _, done_off, _ = off.step([7])
            obs_on, _, done_on, _ = on.step([7])
            np.testing.assert_array_equal(obs_off, obs_on)
            np.testing.assert_array_equal(off.game.state["agent_position"], on.game.state["agent_position"])
            assert done_off == done_on
        assert not on.gate.events
    finally:
        off.close(); on.close()
