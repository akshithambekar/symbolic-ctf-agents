import numpy as np
import pytest
from maritime_nsrl.research_env import ArenaConfig, TeamArena
from maritime_nsrl.tactics import TacticalArena, TacticalPolicy, TacticsConfig, role_contract
from maritime_nsrl.temporal import TemporalMonitor, compile_ltlf


def test_conditional_defense_uses_grounded_intrusion():
    cfg = ArenaConfig(team_size=3, guided=False, reward_shaping=False, temporal_state=False)
    arena = TacticalArena(cfg, TacticsConfig(defender_mode="conditional"))
    try:
        arena.reset(11)
        assert arena.roles()[0] == "support" and "opponent_push" not in arena.labels()
        arena.game.state["agent_position"][3, 0] = 40.
        assert arena.roles()[0] == "defender"
        assert {"opponent_push", "defender_behind_center"} <= arena.labels()
        assert arena.target(0, 0)[0] < arena.size[0]/2
        assert all(arena.action_space.contains(int(a)) for a in TacticalPolicy()(arena))
    finally: arena.close()


def test_dynamic_defender_never_overrides_flag_carrier():
    arena = TacticalArena(ArenaConfig(team_size=3), TacticsConfig(defender_mode="dynamic", attack_style="escort"))
    try:
        arena.reset(12)
        arena.game.state["agent_has_flag"][0] = True
        roles = arena.roles()
        assert roles[0] == "carrier" and roles.count("defender") == 1
        assert "support" in roles
        assert TacticalPolicy()(arena)[0] in {1, 4, 5}
    finally: arena.close()


def test_role_contract_is_conditional_not_global():
    artifact = role_contract("conditional")
    monitor = TemporalMonitor(compile_ltlf(artifact["human_constraints"][0]))
    monitor.update(set())
    assert monitor.satisfied
    monitor.update({"opponent_push"})
    assert monitor.impossible
    with pytest.raises(ValueError): role_contract("unknown")


@pytest.mark.parametrize("mode,y,heading", [("boundary_lower", 2.5, 180.), ("boundary_upper", 77.5, 0.)])
def test_boundary_probe_is_valid_stationary_initial_state(mode, y, heading):
    arena = TeamArena(ArenaConfig(start_mode=mode))
    try:
        arena.reset(12)
        assert arena.game.state["agent_position"][0, 1] == y
        assert arena.game.state["agent_heading"][0] == heading
        assert arena.game.state["agent_speed"][0] == 0
        assert "in_bounds" in arena.labels()
    finally: arena.close()


def test_simulator_footprint_is_respected_for_boundary_starts():
    with pytest.raises(ValueError, match="footprint"):
        TeamArena(ArenaConfig(start_mode="boundary_lower", boundary_start_margin=.05))


def test_feasible_moving_start_substitution_prevents_bounds_failure_without_abort():
    values = dict(guided=False, reward_shaping=False, temporal_state=False, horizon=20.,
        start_mode="boundary_lower", boundary_start_margin=2.01, boundary_start_speed=1.)
    off = TeamArena(ArenaConfig(**values))
    on = TeamArena(ArenaConfig(**values, shield=True))
    try:
        off.reset(861100); on.reset(861100)
        for arena in (off, on):
            while not arena.finished: arena.step([2])
        assert off.metrics["out_of_bounds_steps"] > 0
        assert on.metrics["out_of_bounds_steps"] == 0
        assert on.metrics["shield_interventions"] > 0
        assert on.metrics["shield_deadlocks"] == 0
        assert on.game.current_time == 20.
    finally: off.close(); on.close()
