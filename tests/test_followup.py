import numpy as np
import pytest
from maritime_nsrl.followup import Exposure, ExposureArena, ChaseTrainingArena, native_restart
from maritime_nsrl.research_env import ArenaConfig
from maritime_nsrl.tactics import TacticalArena, TacticalPolicy, TacticsConfig, role_contract


def plain(**kwargs):
    return ArenaConfig(guided=False, reward_shaping=False, temporal_state=False, opponent="static",
        formula="G(in_bounds) & F(has_flag & F(captured))", **kwargs)


def test_return_exposure_is_native_possession_not_a_pickup():
    arena = ExposureArena(plain(horizon=10.), Exposure("return", 6., 1.))
    try:
        obs = arena.reset(10)
        assert np.isfinite(obs).all()
        assert arena.game.state['agent_has_flag'][0]
        assert arena.game.state['flag_taken'][1]
        assert arena.metrics['pickups'] == 0 and arena.ever_flag
        assert arena.game.state['agent_position'][0, 0] > arena.size[0]/2
    finally: arena.close()


def test_opt_in_recovery_keeps_feasible_edge_probe_safe():
    arena = ExposureArena(plain(horizon=20., shield=True), Exposure("approach", 6., 1.), "proactive")
    try:
        arena.reset(11)
        while not arena.finished:
            obs, rewards, _, _ = arena.step([2])
            assert np.isfinite(obs).all() and np.isfinite(rewards).all()
        assert arena.metrics['out_of_bounds_steps'] == 0
        assert arena.metrics['shield_deadlocks'] == 0
        assert arena.game.current_time == 20.
    finally: arena.close()


@pytest.mark.parametrize('mode', ['fixed', 'conditional'])
def test_conflicting_carrier_duty_has_explicit_fallback(mode):
    config = plain(team_size=3, horizon=10.)
    config.formula = role_contract(mode)['ltl']
    arena = TacticalArena(config, TacticsConfig(mode, 'escort', 'raid', True))
    try:
        arena.reset(12)
        positions=arena.game.state['agent_position'].tolist()
        positions[0]=[128.,20.];positions[3]=[72.,60.]
        native_restart(arena, {'agent_position':positions,'agent_has_flag':[True,False,False,False,False,False]})
        assert arena.roles()[0] == 'carrier'
        assert arena.roles().count('defender') == 1
        assert 'defender_behind_center' in arena.labels()
        assert TacticalPolicy()(arena)[0] in {1,4,5}
        assert not arena.monitor.impossible
    finally: arena.close()


def test_chase_task_control_preserves_unshaped_reward():
    arena = ChaseTrainingArena(plain(horizon=5.), 'task')
    try:
        arena.reset(13)
        _, rewards, done, info = arena.step([2])
        assert done
        assert info['metrics']['directional_reward'] == 0
        assert float(rewards[0]) == pytest.approx(info['metrics']['base_reward'])
    finally: arena.close()


def test_invalid_exposure_rejected():
    with pytest.raises(ValueError): Exposure('unknown')
    with pytest.raises(ValueError): Exposure(margin=.5)
    with pytest.raises(ValueError): Exposure(speed=float('nan'))


def test_pair_audit_rejects_missing_and_duplicate_arms():
    import runpy
    from pathlib import Path
    validate = runpy.run_path(str(Path(__file__).parents[1]/'scripts/report_followup.py'))['validate_pairs']
    rows=[dict(training_seed=0,condition='approach',opponent='static',evaluation_seed=5,arm=a)
        for a in ['off','original','brake','proactive']]
    assert len(validate(rows)) == 1
    with pytest.raises(ValueError, match='Incomplete'): validate(rows[:-1])
    with pytest.raises(ValueError, match='Duplicate'): validate(rows+[rows[0]])


def test_directional_reward_does_not_reintroduce_nan_after_guard(monkeypatch):
    from maritime_nsrl.research_env import TeamArena
    arena=ChaseTrainingArena(plain(horizon=5.),'directional')
    try:
        arena.reset(13)
        observation=arena.observe().copy()
        def failed_step(self,actions):
            self.game.state['agent_position'][0]=[float('nan'),float('nan')]
            self.metrics['numerical_terminations']=1
            self.metrics['episode_reward']=-20.
            return observation,np.array([-20.],dtype=np.float32),True,{'metrics':dict(self.metrics)}
        monkeypatch.setattr(TeamArena,'step',failed_step)
        obs,rewards,_,info=arena.step([2])
        assert np.isfinite(obs).all() and np.isfinite(rewards).all()
        assert info['metrics']['directional_reward']==0.
        assert info['metrics']['episode_reward']==-20.
    finally:arena.close()
