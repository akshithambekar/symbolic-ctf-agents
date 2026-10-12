from argparse import Namespace
import json
import pytest
from maritime_nsrl.viewer import load_policy_environment, snapshot, watch, replay


def test_record_and_replay_uses_real_renderer(tmp_path,monkeypatch):
    monkeypatch.setenv('SDL_VIDEODRIVER','dummy')
    monkeypatch.setenv('SDL_AUDIODRIVER','dummy')
    directory=tmp_path/'record'
    args=Namespace(policy='oracle',scenario='standard',opponent='static',team_size=1,
        config=None,record=directory,headless=True,speed=100,episodes=1,seed=4,
        stochastic=False,max_frames=2)
    watch(args)
    rows=[json.loads(line) for line in (directory/'replay.jsonl').read_text().splitlines()]
    assert len(rows)==2 and len(rows[0]['positions'])==2
    assert rows[1]['time']>rows[0]['time']
    assert (directory/'frame-000000.png').stat().st_size>1000
    replay(Namespace(trajectory=directory/'replay.jsonl',episode=0,record=tmp_path/'replay',
                     headless=True,speed=100,max_frames=1))
    assert (tmp_path/'replay/frame-000000.png').is_file()
    with pytest.raises(FileExistsError): watch(args)


def test_oracle_config_and_complete_snapshot():
    env,policy,is_team=load_policy_environment('oracle','wide','chase',3)
    try:
        env.reset(4); actions=policy(env); env.step(actions)
        frame=snapshot(env,actions,0,is_team)
        assert len(frame['positions'])==6 and frame['arena_size']==[200,100]
        assert len(frame['cooldown'])==6 and 'contract_violated' in frame
    finally: env.close()


def test_renderer_preserves_global_rng_and_simulator_state(monkeypatch):
    import numpy as np
    import random
    from maritime_nsrl.viewer import Display
    monkeypatch.setenv('SDL_VIDEODRIVER','dummy')
    env,policy,is_team=load_policy_environment('oracle','standard','static',1)
    renderer=None
    try:
        env.reset(7); frame=snapshot(env,policy(env),0,is_team)
        original=env.game.state['agent_position'].copy()
        np.random.seed(12); random.seed(12)
        expected=(np.random.random(),random.random())
        np.random.seed(12); random.seed(12)
        renderer=Display(frame,headless=True); renderer.draw(frame)
        assert (np.random.random(),random.random())==expected
        np.testing.assert_array_equal(original,env.game.state['agent_position'])
    finally:
        if renderer: renderer.close()
        env.close()
