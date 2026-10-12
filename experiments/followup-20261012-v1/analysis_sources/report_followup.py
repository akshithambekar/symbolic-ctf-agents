"""Verify matched screens and report outcomes without pooling exposure contexts."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def validate_pairs(rows):
    pairs = defaultdict(dict)
    for row in rows:
        key = (row['training_seed'], row['condition'], row['opponent'], row['evaluation_seed'])
        if row['arm'] in pairs[key]: raise ValueError('Duplicate paired arm')
        pairs[key][row['arm']] = row
    for arms in pairs.values():
        if set(arms) != {'off', 'original', 'brake', 'proactive'}: raise ValueError('Incomplete paired arms')
    return pairs


def summarize(rows):
    return {'episodes':len(rows), 'captures':sum(r['captures']>0 for r in rows),
        'pickups':sum(r['pickups']>0 for r in rows), 'bounds_episodes':sum(r['out_of_bounds_steps']>0 for r in rows),
        'deadlocks':sum(r['shield_deadlocks']>0 for r in rows), 'interventions':sum(r['shield_interventions'] for r in rows),
        'offensive_events':sum(r['offensive_events'] for r in rows), 'carrier_events':sum(r['carrier_events'] for r in rows),
        'mean_common_reward':float(np.mean([r['common_task_reward'] for r in rows])),
        'mean_duration':float(np.mean([r['duration'] for r in rows])), 'mean_tags':float(np.mean([r['tags_received'] for r in rows]))}


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def report(root, reference=None):
    root=Path(root)
    rows=json.loads((root/'exposure/episodes.json').read_text(encoding='utf-8'))
    pairs=validate_pairs(rows)
    assert len(rows)==144 and len(pairs)==36
    traces=[json.loads(line) for line in (root/'exposure/traces.jsonl').read_text(encoding='utf-8').splitlines()]
    starts=defaultdict(list)
    for trace in traces:
        identity=trace['identity'];key=tuple(identity[k] for k in ['training_seed','condition','opponent','evaluation_seed'])
        starts[key].append(trace['initial'])
    for frames in starts.values():
        for frame in frames[1:]:
            for key in ['positions','headings','speeds','has_flag','tagged','flag_home','captures']:
                assert frame[key]==frames[0][key],(key,frame,frames[0])
    summaries=[]
    for condition in dict.fromkeys(r['condition'] for r in rows):
        for arm in ['off','original','brake','proactive']:
            summaries.append({'condition':condition,'arm':arm,**summarize([r for r in rows if r['condition']==condition and r['arm']==arm])})
    write_csv(root/'exposure/summary.csv',summaries)
    (root/'exposure/summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    role_rows=json.loads((root/'roles/episodes.json').read_text(encoding='utf-8'));assert len(role_rows)==96
    role_summary=[]
    for style in ['split_flank','escort']:
        for mode in ['none','fixed','conditional','dynamic']:
            for opponent in ['static','patrol','chase','raid']:
                group=[r for r in role_rows if r['attack_style']==style and r['role_mode']==mode and r['tactical_opponent']==opponent]
                assert len(group)==3
                role_summary.append({'attack_style':style,'role_mode':mode,'opponent':opponent,'episodes':len(group),
                    'captures':sum(r['captures']>0 for r in group),'red_captures':sum(r['opponent_captures'] for r in group),
                    'tags':sum(r['tags_received'] for r in group),'satisfied':sum(r['ltlf_satisfied'] for r in group)})
    write_csv(root/'roles/summary.csv',role_summary)
    chase=json.loads((root/'chase/summary.json').read_text(encoding='utf-8'))
    conflict=json.loads((root/'conflicts/episodes.json').read_text(encoding='utf-8'))
    lines=['# Offensive shielding, role control and staged chase follow-up','',
        'Exploratory screens only. Previous studies are unchanged. Exact source snapshots, simulator hashes, policies, configurations and raw traces accompany the runs. Two independently trained source checkpoints are reused unchanged for the shield study. Each exposure cell has one held-out episode per training seed; repetitions are not additional independent seeds.','',
        '## Offensive exposure and deadlock recovery','',
        '| Condition | Arm | Captures / episodes | Pickup episodes | Bounds episodes | Deadlocks | Substitutions | Offensive / carrier substitutions |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:lines.append(f"| {s['condition']} | {s['arm']} | {s['captures']}/{s['episodes']} | {s['pickups']} | {s['bounds_episodes']} | {s['deadlocks']} | {s['interventions']} | {s['offensive_events']} / {s['carrier_events']} |")
    lines+=['','The original gate enforces predicted G(in_bounds). Brake adds zero-speed/turn candidate controls; proactive additionally rejects successors whose current velocity projects outside center bounds in two seconds. That reserve is a heuristic strengthening, not a certified braking model. The learned action space is unchanged. Raw substituted controls are authoritative: emergency controls are labelled hold (7) in legacy waypoint counts.','',
        'Approach/return starts are engineered enemy-territory edge conditions, not ordinary full missions. Return starts initialize possession with PyQuaticus init_dict; their captures test return behavior, not acquisition. All paired initial states were checked. Deadlocks terminate early; reduced time penalties or tags in short episodes are not evidence of improved control. No result supports a general invariant or superiority claim.','',
        '## Role strategies with attack style controlled','',
        '| Attack style | Roles | Opponent | Blue captures / 3 | Red captures | Mission satisfied / 3 |','|---|---|---|---:|---:|---:|']
    for s in role_summary:lines.append(f"| {s['attack_style']} | {s['role_mode']} | {s['opponent']} | {s['captures']} | {s['red_captures']} | {s['satisfied']} |")
    lines+=['','These are scripted tactical policies, not learned coordination. Attack style is identical within each four-role comparison, resolving the earlier role/style confounding. Three episodes per cell are feasibility evidence, not a robust ranking.','',
        '## Carrier/defender conflict','', '| Mode | Reassign carrier duty | Initial roles | Captures | Mission satisfied |','|---|---|---|---:|---:|']
    for r in conflict:lines.append(f"| {r['role_mode']} | {r['carrier_reassignment']} | {r['initial_roles']} | {r['captures']} | {r['ltlf_satisfied']} |")
    lines+=['','Carrier priority is preserved in every variant. Optional fallback assigns another available player to defense when fixed Agent 0 carries; dynamic mode already excludes carriers. These intentionally supplied carrier/intrusion states test contract conflicts, not learned pickups. The role constraint remains instantaneous; reassignment does not guarantee a defender is in position.','',
        '## Staged chase acquisition','', '| Variant | Seed | Added decisions | Completed curriculum | Full-speed pickup % | Full-speed capture % | Static capture % | Candidate |','|---|---:|---:|---|---:|---:|---:|---|']
    for r in chase:lines.append(f"| {r['variant']} | {r['training_seed']} | {r['actual_decisions']} | {r['completed_curriculum']} | {100*r['full_speed_pickup_rate']:.0f} | {100*r['full_speed_capture_rate']:.0f} | {100*r['static_capture_rate']:.0f} | {r['candidate']} |")
    lines+=['','Each training stage adds 2,048 decisions at chase speeds 0.65, 0.85, then 1.0. Advancement requires pickups in at least 20% of five held-out episodes and static captures in at least 60% of five separate episodes. Failed candidates stop; they are still evaluated at full chase speed as transfer diagnostics. Different total budgets after stopping must not be described as matched full-curriculum training.','',
        'Task, flag/home progress, and progress-gated local evasion use identical observations/actions and initial checkpoints. These are task-reward continuation variants without formal training guidance. Common task reward is separate from training return. A candidate flag is a screening recommendation, not a five-seed conclusion.','',
        '## Next decision','', 'Reject any recovery variant that reduces captures through early termination. Promote an offensive shield condition only when it has nonzero offensive substitutions, fewer bounds episodes, unchanged or better captures, and no deadlocks; freeze a separate held-out protocol before a larger comparison. Chase and role results remain exploratory.']
    if reference is not None:
        references=json.loads((reference/'summary.json').read_text(encoding='utf-8'))
        lines += ['', '## Post-hoc easier-speed reference', '',
            'The unmodified source policies were subsequently evaluated on the same lower-speed screen seeds. This reference was not used to alter promotion. A passed capability screen does not establish learning gains.', '',
            '| Seed | Chase speed | Unmodified pickups / 5 |', '|---|---:|---:|']
        for r in references: lines.append(f"| {r['training_seed']} | {r['speed']} | {round(5*r['pickup_rate'])} |")
        lines += ['', 'Compare stage screen files against these references before attributing any lower-speed success to continuation training. The full-speed baseline and continued variants also use matched evaluation seeds.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'analysis_provenance.json').write_text(json.dumps({'paired_cells':len(pairs),'paired_initial_states_verified':True,
        'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    conditions=list(dict.fromkeys(r['condition'] for r in rows))
    for ax,metric,label in zip(axes,['captures','bounds_episodes','deadlocks'],['Capture rate','Bounds episode rate','Deadlock rate']):
        for j,arm in enumerate(['off','original','brake','proactive']):
            values=[next(s[metric]/s['episodes'] for s in summaries if s['condition']==c and s['arm']==arm) for c in conditions]
            ax.plot(range(len(conditions)),values,marker='o',label=arm)
        ax.set_xticks(range(len(conditions)),[c.replace('_','\n') for c in conditions],fontsize=7);ax.set_ylim(-.03,1.03);ax.set_title(label)
    axes[0].legend(fontsize=7);fig.suptitle('Two-seed exploratory exposure screen; initial possession supplied for return probes')
    fig.tight_layout();fig.savefig(root/'exposure.png',dpi=200);plt.close(fig)
    print(json.dumps({'exposure':summaries,'chase':chase},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--reference',type=Path)
    args=p.parse_args();report(args.root,args.reference)
