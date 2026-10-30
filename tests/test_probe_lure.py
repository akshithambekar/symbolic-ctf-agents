from types import SimpleNamespace
import numpy as np
from maritime_nsrl.probe_lure import INSTRUCTION, MILESTONES, tactical_labels, mission_phase
from maritime_nsrl.translation import translate
from maritime_nsrl.temporal import compile_ltlf, TemporalMonitor
from maritime_nsrl.research_env import TeamArena, ArenaConfig
from maritime_nsrl.probe_lure import FORMULA


def test_translation_requires_order_and_capture():
    spec = translate(INSTRUCTION, logic_backend="dfa")
    automaton = compile_ltlf(spec["ltl"])
    monitor = TemporalMonitor(automaton)
    for phase, name in enumerate(MILESTONES):
        assert mission_phase(monitor) == phase
        monitor.update({"in_bounds", name})
    assert monitor.satisfied
    reversed_monitor = TemporalMonitor(automaton)
    for name in reversed(MILESTONES):
        reversed_monitor.update({"in_bounds", name})
    assert not reversed_monitor.satisfied
    monitor.update(set())
    assert monitor.impossible


def test_commitment_requires_motion_toward_probe_and_untagged_agents():
    game = SimpleNamespace(state={
        "agent_position": np.array([[90., 65.], [110., 65.]]),
        "flag_home": np.array([[20., 40.], [140., 40.]]),
        "agent_heading": np.array([90., -90.]),
        "agent_speed": np.array([1., 1.]),
        "agent_is_tagged": np.array([False, False]),
    })
    assert {"probe_upper", "pursuit_committed"} <= tactical_labels(game, [160., 80.])
    game.state["agent_heading"][1] = 90.
    assert "pursuit_committed" not in tactical_labels(game, [160., 80.])
    game.state["agent_is_tagged"][0] = True
    assert not tactical_labels(game, [160., 80.])


def test_scripted_simulator_run_reaches_switch_before_capture():
    arena = TeamArena(ArenaConfig(formula=FORMULA))
    actions = [3, 7, 5, 2, 0, 1, 1]
    phases = set()
    try:
        arena.reset(7)
        while not arena.finished:
            phase = mission_phase(arena.monitor)
            phases.add(phase)
            arena.step([actions[phase]])
        assert {0, 1, 2, 3, 4} <= phases
    finally:
        arena.close()


def test_probe_action_stops_beyond_midfield_and_warmup_completes():
    arena = TeamArena(ArenaConfig(formula=FORMULA, probe_action=True, curriculum_phase=2))
    try:
        arena.reset(7)
        assert arena.action_space.n == 9
        from maritime_nsrl.viewer import snapshot

        assert snapshot(arena, [8], 0, True)["action_names"][8] == "upper probe"
        np.testing.assert_allclose(arena.target(0, 8), [89.6, 64.0])
        while not arena.finished:
            _, _, _, info = arena.step([8])
        assert info["metrics"]["curriculum_success"]
        assert not info["metrics"]["ltlf_satisfied"]
        assert info["metrics"]["captures"] == 0
        assert info["metrics"]["duration"] < 240
    finally:
        arena.close()
