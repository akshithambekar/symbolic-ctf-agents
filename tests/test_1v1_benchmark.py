import copy
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.run_1v1 import (
    CaseEvents,
    ScenarioRed,
    orbit_start,
    scenario_for,
    validate_suite,
)


SUITE = json.loads((Path(__file__).parents[1] / "benchmarks/1v1/v1.json").read_text())


def test_v1_roster_invariants_and_explicit_starts():
    validate_suite(SUITE)
    assert [case["seed"] for case in SUITE["cases"]] == list(range(51001, 51051))
    assert SUITE["cases"][14]["red_route"]["start"] == [84, 11]
    assert SUITE["cases"][42]["proposed_blue_start"] == [21, 38]
    assert SUITE["cases"][42]["blue_start"] == [23, 38]
    assert SUITE["cases"][44]["red_route"]["start"] == [86, 26]
    assert SUITE["cases"][49]["red_route"]["start"] == [112, 60]


def test_orbit_spawn_radius_and_tangent_heading():
    pos, heading = orbit_start(SUITE["cases"][0]["red_route"])
    assert np.allclose(pos, [150, 40])
    assert heading == 180
    route = SUITE["cases"][13]["red_route"]
    pos, heading = orbit_start(route)
    assert np.isclose(np.linalg.norm(np.asarray(pos) - route["center"]), route["radius"])
    assert heading == 45


def test_reactive_case_pursues_any_red_side_intruder():
    route = SUITE["cases"][49]["red_route"]
    red = ScenarioRed(route, 1.0, reactive=True)
    class Game:
        current_time = 0.1
        max_speeds = [2.0, 2.0]
        state = {"agent_position": np.array([[121.0, 60.0], [112.0, 59.0]]),
                 "agent_heading": np.array([90.0, 180.0]),
                 "agent_is_tagged": np.array([False, False])}
    action = red.action(Game())
    assert red.pursuing
    assert np.isclose(action[0], 2.0)


def test_pursuit_respects_timeout_and_cooldown():
    route = copy.deepcopy(SUITE["cases"][44]["red_route"])
    red = ScenarioRed(route, 0.8)
    class Game:
        max_speeds = [2.0, 2.0]
        state = {"agent_position": np.array([[95.0, 26.0], [86.0, 26.0]]),
                 "agent_heading": np.array([90.0, 0.0]),
                 "agent_is_tagged": np.array([False, False])}
        current_time = 0.1
    red.action(Game())
    assert red.pursuing
    Game.current_time = 8.2
    red.action(Game())
    assert not red.pursuing
    assert red.cooldown_until > Game.current_time
