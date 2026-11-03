from types import SimpleNamespace
import numpy as np
import pytest
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
        assert arena.metrics["enemy_territory_steps"] > 0
        assert arena.metrics["enemy_territory_time"] == pytest.approx(
            arena.metrics["enemy_territory_steps"]
        )
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


def test_teacher_reaches_lower_attack_against_full_speed_defender():
    from maritime_nsrl.probe_lure import ProbeLureTeacher

    arena = TeamArena(ArenaConfig(formula=FORMULA, probe_action=True, opponent_speed=1.0))
    teacher = ProbeLureTeacher()
    try:
        arena.reset(0)
        while not arena.finished:
            _, _, _, info = arena.step(teacher(arena))
        metrics = info["metrics"]
        assert metrics["milestone_times"][3] is not None
        assert metrics["furthest_phase"] >= 4
        assert metrics["out_of_bounds_steps"] == 0
        if metrics["captures"]:
            assert metrics["ltlf_satisfied"]
    finally:
        arena.close()


def test_teacher_completes_full_mission_against_slower_defender():
    from maritime_nsrl.probe_lure import ProbeLureTeacher

    arena = TeamArena(ArenaConfig(formula=FORMULA, probe_action=True, opponent_speed=0.6))
    teacher = ProbeLureTeacher()
    try:
        arena.reset(0)
        while not arena.finished:
            _, _, _, info = arena.step(teacher(arena))
        metrics = info["metrics"]
        assert metrics["captures"] == 1
        assert metrics["ltlf_satisfied"]
        assert metrics["out_of_bounds_steps"] == 0
        assert metrics["furthest_phase"] == 6
        assert all(value is not None for value in metrics["milestone_times"])
    finally:
        arena.close()


def test_milestone_rewards_are_paid_once_across_steps(monkeypatch):
    config = ArenaConfig(
        formula=FORMULA, repeat=1, horizon=20, probe_milestone_reward=2.5,
        potential_scale=0, time_penalty=0, tag_penalty=0, acceptance_bonus=0,
    )
    arena = TeamArena(config)
    try:
        arena.reset(7)
        labels = {"in_bounds", "probe_upper"}
        monkeypatch.setattr(arena, "labels", lambda **kwargs: labels)
        _, first_reward, _, _ = arena.step([7])
        first_time = arena.metrics["milestone_times"][0]
        _, repeated_reward, _, _ = arena.step([7])
        labels = {"in_bounds", "pursuit_committed"}
        _, next_reward, _, _ = arena.step([7])
        _, repeated_next_reward, _, _ = arena.step([7])
        assert first_reward[0] == next_reward[0] == 2.5
        assert repeated_reward[0] == repeated_next_reward[0] == 0
        assert arena.metrics["milestone_reward"] == 5.0
        assert arena.metrics["furthest_phase"] == 2
        assert arena.metrics["milestone_times"][0] == first_time
        assert arena.metrics["milestone_times"][1] > first_time
    finally:
        arena.close()


def test_bounds_failure_cannot_award_completion_milestones(monkeypatch):
    arena = TeamArena(ArenaConfig(
        formula=FORMULA, repeat=1, horizon=20, probe_milestone_reward=2.5,
        potential_scale=0, time_penalty=0, tag_penalty=0, failure_penalty=0,
    ))
    try:
        arena.reset(7)
        labels = {"in_bounds", "probe_upper"}
        monkeypatch.setattr(arena, "labels", lambda **kwargs: labels)
        arena.step([7])
        labels = set()
        _, reward, _, _ = arena.step([7])
        assert arena.monitor.impossible
        assert reward[0] == 0
        assert arena.metrics["furthest_phase"] == 1
        assert arena.metrics["milestone_reward"] == 2.5
        assert arena.metrics["milestone_times"][1:] == [None] * 5
        labels = {"in_bounds", *MILESTONES}
        arena.step([7])
        assert arena.metrics["milestone_reward"] == 2.5
    finally:
        arena.close()


@pytest.mark.parametrize("phase", [3, 4])
def test_later_curriculum_phases_end_on_their_own_milestones(monkeypatch, phase):
    arena = TeamArena(ArenaConfig(formula=FORMULA, repeat=1, curriculum_phase=phase))
    try:
        arena.reset(7)
        labels = {"in_bounds", *MILESTONES[:phase]}
        monkeypatch.setattr(arena, "labels", lambda **kwargs: labels)
        _, _, done, info = arena.step([7])
        assert done
        assert info["metrics"]["curriculum_success"]
        assert info["metrics"]["furthest_phase"] == phase
        assert not info["metrics"]["ltlf_satisfied"]
        assert info["metrics"]["captures"] == 0
    finally:
        arena.close()


def test_default_milestone_reward_keeps_original_potential_shaping(monkeypatch):
    config = ArenaConfig(formula=FORMULA, repeat=1, time_penalty=0, tag_penalty=0)
    assert config.probe_milestone_reward == 0
    arena = TeamArena(config)
    try:
        arena.reset(7)
        phi_before = arena.potential()
        monkeypatch.setattr(arena, "labels", lambda **kwargs: {"in_bounds", "probe_upper"})
        _, reward, _, _ = arena.step([7])
        expected = config.potential_scale * (config.gamma * arena.potential() - phi_before)
        assert reward[0] == pytest.approx(expected)
        assert arena.metrics["milestone_reward"] == 0
        assert arena.metrics["furthest_phase"] == 1
    finally:
        arena.close()


@pytest.mark.parametrize("phase", [0, 6])
def test_full_mission_acceptance_bonus_is_paid_once(monkeypatch, phase):
    arena = TeamArena(ArenaConfig(
        formula=FORMULA, repeat=1, curriculum_phase=phase, potential_scale=0,
        time_penalty=0, tag_penalty=0, capture_reward=0, acceptance_bonus=24,
    ))
    try:
        arena.reset(7)
        monkeypatch.setattr(arena, "labels", lambda **kwargs: {"in_bounds", *MILESTONES})
        _, reward, _, _ = arena.step([7])
        assert arena.monitor.satisfied
        assert reward[0] == 24
    finally:
        arena.close()
