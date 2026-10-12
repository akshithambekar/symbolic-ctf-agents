"""Seed-level reporting and paired-state checks for selected exposure replication."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def summary(rows):
    times=[r['capture_time'] for r in rows if r['captures']>0 and r['capture_time'] is not None]
    return {'episodes':len(rows),'captures':sum(r['captures']>0 for r in rows),'pickups':sum(r['pickups']>0 for r in rows),
        'bounds_episodes':sum(r['out_of_bounds_steps']>0 for r in rows),'deadlocks':sum(r['shield_deadlocks']>0 for r in rows),
        'interventions':sum(r['shield_interventions'] for r in rows),'offensive_events':sum(r['offensive_events'] for r in rows),
        'mean_common_reward':float(np.mean([r['common_task_reward'] for r in rows])),
        'mean_tags':float(np.mean([r['tags_received'] for r in rows])),
        'mission_satisfied':sum(r['ltlf_satisfied'] for r in rows),
        'mean_capture_time_successes':float(np.mean(times)) if times else None}


def csv_rows(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def report(root):
    rows=json.loads((root/'episodes.json').read_text(encoding='utf-8'));assert len(rows)==150
    pairs=defaultdict(dict)
    for r in rows:
        key=(r['training_seed'],r['evaluation_seed'],r['opponent'])
        assert r['arm'] not in pairs[key]
        pairs[key][r['arm']]=r
    assert len(pairs)==75 and all(set(p)=={'off','original'} for p in pairs.values())
    starts=defaultdict(list)
    for line in (root/'traces.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line);i=r['identity'];starts[(i['training_seed'],i['evaluation_seed'],i['opponent'])].append(r)
    assert len(starts)==75
    for pair in starts.values():
        assert len(pair)==2
        for key in ['positions','headings','speeds','has_flag','tagged','flag_home','captures']:
            assert pair[0]['initial'][key]==pair[1]['initial'][key]
        a=dict(pair[0]['config']);b=dict(pair[1]['config']);a.pop('shield');b.pop('shield');assert a==b
    totals={arm:summary([r for r in rows if r['arm']==arm]) for arm in ['off','original']}
    seeds=[]
    for seed in range(5):
        for arm in ['off','original']:seeds.append({'training_seed':seed,'arm':arm,**summary([r for r in rows if r['training_seed']==seed and r['arm']==arm])})
    cells=[]
    for opponent in ['static','patrol','chase']:
        for arm in ['off','original']:cells.append({'opponent':opponent,'arm':arm,**summary([r for r in rows if r['opponent']==opponent and r['arm']==arm])})
    csv_rows(root/'seed_summary.csv',seeds);csv_rows(root/'opponent_summary.csv',cells)
    (root/'summary.json').write_text(json.dumps(totals,indent=2),encoding='utf-8')
    deltas=[]
    for seed in range(5):
        a=next(r for r in seeds if r['training_seed']==seed and r['arm']=='off')
        b=next(r for r in seeds if r['training_seed']==seed and r['arm']=='original')
        deltas.append({'training_seed':seed,'capture_rate_delta':(b['captures']-a['captures'])/a['episodes'],
            'bounds_rate_delta':(b['bounds_episodes']-a['bounds_episodes'])/a['episodes'],
            'common_reward_delta':b['mean_common_reward']-a['mean_common_reward']})
    csv_rows(root/'paired_seed_deltas.csv',deltas)
    lines=['# Five-seed matched offensive-exposure replication','',
        'The original one-step gate was selected from the earlier two-seed screen. This freezes a narrow engineered enemy-territory approach condition: 2.01 m lower-edge margin, 1 m/s outward speed, standard arena. Seeds 0/1 reuse selection checkpoints; seeds 2/3/4 are new independently trained replicas of the same imitation-initialized PPO protocol. All 150 held-out evaluation episodes use fresh seeds and exact same-checkpoint off/on pairing.','',
        '| Arm | Captures / episodes | Pickups | Bounds episodes | Deadlocks | Offensive substitutions | Common reward | Mean tags |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,r in totals.items():lines.append(f"| {arm} | {r['captures']}/{r['episodes']} | {r['pickups']} | {r['bounds_episodes']} | {r['deadlocks']} | {r['offensive_events']} | {r['mean_common_reward']:.3f} | {r['mean_tags']:.3f} |")
    lines+=['','## Independent training units','', '| Seed | Arm | Captures / 15 | Bounds episodes | Deadlocks |','|---|---|---:|---:|---:|']
    for r in seeds:lines.append(f"| {r['training_seed']} | {r['arm']} | {r['captures']} | {r['bounds_episodes']} | {r['deadlocks']} |")
    lines+=['','Seed-level paired deltas and separate points are retained; evaluation episodes are not treated as independent training seeds. Zero sample variance in a seed metric is not certainty about other distributions. No pooled-episode confidence interval is claimed as seed-level uncertainty.','',
        '## Opponent breakdown','', '| Opponent | Arm | Captures / 25 | Pickups | Bounds episodes |','|---|---|---:|---:|---:|']
    for r in cells:lines.append(f"| {r['opponent']} | {r['arm']} | {r['captures']} | {r['pickups']} | {r['bounds_episodes']} |")
    lines+=['','## Scope','',
        'This measures deployment-only filtering of identical task-capable checkpoints, not neuro-symbolic training superiority. Any bounds benefit applies to this selected condition and exact simulator model. It is not an unbiased comparison across all game states: the stress condition was chosen after calibration and deliberately exposes unsafe proposals. Five seeds provide replication of that condition, not a general invariant. Original high-speed deadlocks and rejected proactive/braking variants remain in the separate screen. Chase acquisition in ordinary full missions remains a separate question.','',
        'All 75 paired configurations and initial states were verified. Raw metrics, exact models/configurations/specifications, source hashes and full state/control traces are retained. Earlier results are unchanged.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'analysis_provenance.json').write_text(json.dumps({'pair_count':75,'config_and_initial_state_pairs_verified':True,
        'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8,4))
    for ax,metric,title in zip(axes,['captures','bounds_episodes'],['Capture rate','Bounds episode rate']):
        for seed in range(5):
            values=[next(r[metric]/r['episodes'] for r in seeds if r['training_seed']==seed and r['arm']==arm) for arm in ['off','original']]
            ax.plot([0+(seed-2)*.025,1+(seed-2)*.025],values,marker='o',label=f'seed {seed}')
        ax.set_xticks([0,1],['Off','Original gate']);ax.set_ylim(-.04,1.04);ax.set_title(title)
    axes[0].legend(fontsize=8);fig.suptitle('Five-seed replication of selected offensive edge condition')
    fig.tight_layout();fig.savefig(root/'seed_comparison.png',dpi=200);plt.close(fig)
    print(json.dumps(totals,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);report(p.parse_args().root)
