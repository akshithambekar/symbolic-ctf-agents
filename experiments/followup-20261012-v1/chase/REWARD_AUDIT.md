# Chase reward audit

Training return and common task reward have different definitions. The added progress/evasion feedback is a heuristic, not fully potential-based shaping: it skips tagged transitions and uses pickup history to switch the target. It can therefore reward repeated approaches without flag acquisition, and after possession loss it can keep rewarding return. These limitations are part of the tested definitions, not evidence against all progress or evasion shaping.

| Variant | Seed | Stage | Episodes | No-pickup positive-return episodes | Mean training return | Mean common reward | Evade action % |
|---|---:|---|---:|---:|---:|---:|---:|
| directional | 0 | stage0 | 49 | 28 | 30.380 | 3.774 | 0.05 |
| directional | 0 | stage1 | 42 | 42 | 14.911 | -6.400 | 0.00 |
| directional | 1 | stage0 | 51 | 25 | 33.871 | 6.542 | 0.05 |
| directional | 1 | stage1 | 42 | 42 | 14.916 | -6.400 | 0.15 |
| progress | 0 | stage0 | 49 | 28 | 29.912 | 3.774 | 0.05 |
| progress | 0 | stage1 | 42 | 42 | 14.877 | -6.400 | 0.00 |
| progress | 1 | stage0 | 51 | 25 | 33.302 | 6.542 | 0.05 |
| progress | 1 | stage1 | 42 | 42 | 14.926 | -6.400 | 0.15 |
| task | 0 | stage0 | 49 | 0 | 12.602 | 3.815 | 0.10 |
| task | 0 | stage1 | 42 | 0 | -1.200 | -6.400 | 0.00 |
| task | 1 | stage0 | 51 | 0 | 15.839 | 6.541 | 0.10 |
| task | 1 | stage1 | 42 | 0 | -1.200 | -6.400 | 0.20 |

Positive training return without pickups demonstrates a reward/task mismatch; it does not establish intentional exploitation or prove the cause of chase failure. Action fractions are descriptive, not causal. The easier-speed reference shows whether capability was already present before continuation.

Next reward candidate: a current-possession potential with discount-correct shaping through tagged and return transitions, evaluated under matched controls. Separately investigate exploration and richer maneuvering. Do not promote these tested variants on training return alone.

A post-screen numerical guard was added to the working implementation so reward additions cannot reintroduce NaNs after simulator termination. Run snapshots preserve the code actually used; the numerical-termination count above identifies whether any recorded training episode was affected.

Legacy enemy_territory_time in raw metrics multiplies step count by 0.1, although this arena advances dt=0.1 times speedup=10 per simulator step. For these runs, corrected seconds equal enemy_territory_steps, not that legacy time column. Raw metrics remain unchanged. Role studies likewise use actual contract satisfaction rather than the legacy fixed-safety column, which counts a missing defender even in no-defender mode.
