import copy
import numpy as np
from maritime_nsrl.research_env import TeamArena,ArenaConfig,SharedTeamVecEnv
from maritime_nsrl.shield import PredictiveShield
from maritime_nsrl.translation import translate
from maritime_nsrl.oracle import run_oracle


class TinyGame:
    def __init__(self): self.safe=True
    def step(self, action): self.safe=action["safe"]; return self.safe


def test_shield_rejects_unsafe_actions_and_fails_closed():
    game = TinyGame()
    gate = PredictiveShield()
    gate.initialize({"in_bounds"})
    ground = lambda g: {"in_bounds"} if g.safe else set()
    selected,_,index = gate.transition(game,[{"safe":False},{"safe":True}],ground)
    assert index == 1 and selected.safe and gate.interventions == 1
    assert game.safe
    selected,_,_ = gate.transition(game,[{"safe":False}],ground)
    assert selected is None and game.safe and gate.deadlocks == 1


def test_joint_step_advances_game_once_and_all_agents_move():
    arena = TeamArena(ArenaConfig(team_size=2,horizon=10))
    env = SharedTeamVecEnv(arena)
    try:
        env.seed(3)
        obs = env.reset()
        positions = arena.game.state["agent_position"][:2].copy()
        env.step_async(np.array([0,2]))
        nxt,rewards,dones,infos = env.step_wait()
        assert arena.game.current_time == 5.
        assert obs.shape == nxt.shape == (2,38)
        assert np.all(np.linalg.norm(arena.game.state["agent_position"][:2]-positions,axis=1)>0)
        assert not any(dones) and len(rewards)==2
        _,_,dones,infos = env.step(np.array([0,2]))
        assert all(dones) and arena.game.current_time==0
        assert sum("metrics" in i for i in infos)==1
        assert all("terminal_observation" in i for i in infos)
    finally: env.close()


def test_advanced_english_validation():
    result = translate("Always stay within the arena. Whenever carrying a flag, eventually capture it.",logic_backend="dfa")
    assert result["validation"]["satisfiable"]


def test_role_constraint_is_grounded_for_three_agents():
    arena = TeamArena(ArenaConfig(team_size=3, formula="G(defender_behind_center -> teammate_safe) & F(captured)"))
    try:
        arena.reset(2)
        assert "defender_behind_center" in arena.labels()
        assert "teammate_safe" in arena.labels()
    finally: arena.close()


def test_shielded_actual_game_matches_unshielded_when_safe():
    a,b = TeamArena(ArenaConfig()),TeamArena(ArenaConfig(shield=True))
    try:
        np.testing.assert_array_equal(a.reset(5),b.reset(5))
        for _ in range(3):
            one,two = a.step([0]),b.step([0])
            np.testing.assert_array_equal(one[0],two[0])
            np.testing.assert_array_equal(one[1],two[1])
        assert b.gate.interventions == 0 and b.gate.predictions == 15
    finally: a.close(); b.close()


def test_stage_metrics_expose_offensive_funnel():
    rows = run_oracle(ArenaConfig(team_size=1, opponent="static", scenario="standard",
        horizon=240, formula="F(captured)"), episodes=1, seed=41)
    row = rows[0]
    assert row["reached_enemy_territory"]
    assert row["reached_enemy_flag"]
    assert row["pickup_time"] is not None
    assert row["return_crossed_midfield"]
    assert row["captures"] == 1
    assert row["capture_time"] >= row["pickup_time"]
    assert row["enemy_flag_distance_min"] <= 10
    assert row["trajectory"]


def test_shield_records_discrete_substitution_details():
    arena = TeamArena(ArenaConfig(team_size=1, shield=True, horizon=5,
        formula="F(captured)", trace_stride=1))
    try:
        arena.reset(7)
        _,_,_,info = arena.step([7])
        assert sum(arena.metrics["requested_action_counts"]) == 1
        assert sum(arena.metrics["executed_action_counts"]) == 1
        assert info["metrics"]["trajectory"][0]["requested_actions"] == [7]
    finally: arena.close()


def test_nonfinite_simulator_state_fails_closed_without_nan_observation():
    arena=TeamArena(ArenaConfig(team_size=1,horizon=10))
    try:
        prior=arena.reset(8).copy()
        arena.game.state["agent_position"][0,0]=np.nan
        assert not arena.finite_game_state()
        # The cached last valid observation is the value returned on numerical termination.
        np.testing.assert_allclose(arena.last_valid_observation,prior)
    finally: arena.close()


def test_chase_feature_and_distance_metrics_are_available():
    arena = TeamArena(ArenaConfig(team_size=1, opponent="chase", horizon=5,
        formula="F(captured)", defender_distance_feature=True))
    try:
        observation = arena.reset(12)
        assert observation.shape == (1, arena.observation_space.shape[0])
        observation, _, done, info = arena.step([2])
        assert done
        assert np.all(np.isfinite(observation))
        metrics = info["metrics"]
        assert metrics["enemy_flag_distance_min"] is not None
        assert metrics["defender_distance_min"] is not None
        assert metrics["defender_distance_mean"] >= metrics["defender_distance_min"]
        assert metrics["enemy_territory_time"] >= 0
    finally: arena.close()
