"""Audit task outcomes versus training return; preserve raw reward definitions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def analyze(root):
    rows=[]
    for cell in sorted((root/'chase').iterdir()):
        if not (cell/'status.json').exists():continue
        status=json.loads((cell/'status.json').read_text(encoding='utf-8'))
        for stage in sorted(cell.glob('stage*')):
            if not stage.is_dir() or not (stage/'episodes.jsonl').exists():continue
            episodes=[json.loads(line) for line in (stage/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
            counts=np.sum([r['requested_action_counts'] for r in episodes],axis=0)
            rows.append({'training_seed':status['training_seed'],'variant':status['variant'],'stage':stage.name,
                'episodes':len(episodes),'capture_episodes':sum(r['captures']>0 for r in episodes),
                'zero_pickup_positive_return_episodes':sum(r['pickups']==0 and r['episode_reward']>0 for r in episodes),
                'mean_training_return':float(np.mean([r['episode_reward'] for r in episodes])),
                'mean_common_reward':float(np.mean([r['common_task_reward'] for r in episodes])),
                'mean_added_reward':float(np.mean([r.get('directional_reward',0.) for r in episodes])),
                'evade_action_fraction':float(counts[6]/max(1,counts.sum())),
                'numerical_terminations':sum(r['numerical_terminations'] for r in episodes)})
    with (root/'chase/reward_audit.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (root/'chase/reward_audit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    lines=['# Chase reward audit','',
        'Training return and common task reward have different definitions. The added progress/evasion feedback is a heuristic, not fully potential-based shaping: it skips tagged transitions and uses pickup history to switch the target. It can therefore reward repeated approaches without flag acquisition, and after possession loss it can keep rewarding return. These limitations are part of the tested definitions, not evidence against all progress or evasion shaping.','',
        '| Variant | Seed | Stage | Episodes | No-pickup positive-return episodes | Mean training return | Mean common reward | Evade action % |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['variant']} | {r['training_seed']} | {r['stage']} | {r['episodes']} | {r['zero_pickup_positive_return_episodes']} | {r['mean_training_return']:.3f} | {r['mean_common_reward']:.3f} | {100*r['evade_action_fraction']:.2f} |")
    lines+=['','Positive training return without pickups demonstrates a reward/task mismatch; it does not establish intentional exploitation or prove the cause of chase failure. Action fractions are descriptive, not causal. The easier-speed reference shows whether capability was already present before continuation.','',
        'Next reward candidate: a current-possession potential with discount-correct shaping through tagged and return transitions, evaluated under matched controls. Separately investigate exploration and richer maneuvering. Do not promote these tested variants on training return alone.','',
        'A post-screen numerical guard was added to the working implementation so reward additions cannot reintroduce NaNs after simulator termination. Run snapshots preserve the code actually used; the numerical-termination count above identifies whether any recorded training episode was affected.','',
        'Legacy enemy_territory_time in raw metrics multiplies step count by 0.1, although this arena advances dt=0.1 times speedup=10 per simulator step. For these runs, corrected seconds equal enemy_territory_steps, not that legacy time column. Raw metrics remain unchanged. Role studies likewise use actual contract satisfaction rather than the legacy fixed-safety column, which counts a missing defender even in no-defender mode.']
    (root/'chase/REWARD_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    positions=np.arange(len(rows))
    ax.bar(positions-.18,[r['mean_training_return'] for r in rows],width=.36,label='Training return')
    ax.bar(positions+.18,[r['mean_common_reward'] for r in rows],width=.36,label='Common task reward')
    ax.set_xticks(positions,[f"{r['variant']}\ns{r['training_seed']}/{r['stage']}" for r in rows],fontsize=7)
    ax.set_ylabel('Mean per training episode');ax.axhline(0,color='black',lw=.7);ax.legend()
    ax.set_title('Descriptive reward audit; different objectives are not interchangeable')
    fig.tight_layout();fig.savefig(root/'chase/reward_audit.png',dpi=200);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);analyze(p.parse_args().root)
