# Project Deliverables and Schedule

Transcribed from the two supplied Gantt chart images. Activity names, IDs, durations, people, and predecessors are retained as shown. Line wrapping has been normalized for readability.

- **Schedule:** shaded weeks in the chart, using its W1–W32 labels. Ranges are inclusive; no calendar dates were supplied.
- **Duration:** the value in the chart's Duration column, expressed in weeks.
- **Predecessors:** original activity or group IDs; `N/A` is retained where shown.
- Group bars and activity bars are recorded separately. Differences between them are preserved, with notes at the end.
- The chart supplies no completion status or acceptance criteria.

## 1. Project Environment and Research

**Duration:** 2 weeks · **People:** Team · **Predecessors:** N/A · **Group bar:** W1–W2

- **1.1 — Install and configure PyQuaticus**
  - Schedule: W1 · Duration: 1 week · People: Team · Predecessors: N/A
- **1.2 — Run MCTF tutorials and sample agents**
  - Schedule: W2 · Duration: 1 week · People: Team · Predecessors: 1.1
- **1.3 — Install and evaluate LTL CodeGen**
  - Schedule: W1 · Duration: 1 week · People: Armin / Zawadul / Tuga · Predecessors: N/A
- **1.4 — Research alternative English-to-LTL tools**
  - Schedule: W2 · Duration: 1 week · People: Armin / Zawadul / Tuga · Predecessors: 1.3
- **1.5 — Study MCTF rules & observation/action spaces**
  - Schedule: W3 · Duration: 1 week · People: Team · Predecessors: 1.2

## 2. System Architecture & Interface

**Duration:** 2 weeks · **People:** Team · **Predecessors:** 1 · **Group bar:** W3–W4

- **2.1 — Define English-to-LTL-to-RL architecture**
  - Schedule: W3 · Duration: 1 week · People: Team · Predecessors: 1
- **2.2 — Define structured LTL output format**
  - Schedule: W4 · Duration: 1 week · People: Team · Predecessors: 2.1
- **2.3 — Define LTL-to-RL interface**
  - Schedule: W5 · Duration: 1 week · People: Team · Predecessors: 2.2
- **2.4 — Define validation and failure-handling**
  - Schedule: W5 · Duration: 1 week · People: Team · Predecessors: 2.2

## 3. LTL Translation Pipeline

**Duration:** 5 weeks · **People:** Armin / Zawadul / Tuga · **Predecessors:** 2 · **Group bar:** W5–W9

- **3.1 — Create NL strategy test cases**
  - Schedule: W5 · Duration: 1 week · People: Armin / Zawadul / Tuga · Predecessors: 2.2
- **3.2 — Implement English-to-LTL translation**
  - Schedule: W6–W7 · Duration: 2 weeks · People: Armin / Zawadul / Tuga · Predecessors: 3.1
- **3.3 — Implement syntax/semantic verification**
  - Schedule: W8 · Duration: 1 week · People: Armin / Zawadul / Tuga · Predecessors: 3.2
- **3.4 — Evaluate translation reliability**
  - Schedule: W9–W10 · Duration: 2 weeks · People: Armin / Zawadul / Tuga · Predecessors: 3.3
- **3.5 — Produce structured LTL output**
  - Schedule: W8 · Duration: 1 week · People: Armin / Zawadul / Tuga · Predecessors: 3.2

## 4. PyQuaticus/RL Pipeline

**Duration:** 7 weeks · **People:** Akshith / Yahya · **Predecessors:** 2 · **Group bar:** W5–W11

- **4.1 — Establish baseline agent control**
  - Schedule: W5 · Duration: 1 week · People: Akshith / Yahya · Predecessors: 1.2
- **4.2 — Define RL states, actions and rewards**
  - Schedule: W6–W7 · Duration: 2 weeks · People: Akshith / Yahya · Predecessors: 4.1
- **4.3 — Integrate symbolic/LTL commands**
  - Schedule: W8–W9 · Duration: 2 weeks · People: Akshith / Yahya · Predecessors: 2.3, 4.2
- **4.4 — Tier 1: static-opponent scenario**
  - Schedule: W10–W11 · Duration: 2 weeks · People: Akshith / Yahya · Predecessors: 4.3
- **4.5 — Tier 2: patrolling-opponent scenario**
  - Schedule: W12–W13 · Duration: 2 weeks · People: Akshith / Yahya · Predecessors: 4.4
- **4.6 — Tier 3: reactive-opponent scenario**
  - Schedule: W14–W15 · Duration: 2 weeks · People: Akshith / Yahya · Predecessors: 4.5
- **4.7 — Evaluate training & performance**
  - Schedule: W16 · Duration: 1 week · People: Akshith / Yahya · Predecessors: 3.5, 4.6

## 5. Integration

**Duration:** 3 weeks · **People:** Team · **Predecessors:** 3.5, 4.6 · **Group bar:** W10–W12

- **5.1 — Connect LTL pipeline to RL layer**
  - Schedule: W10 · Duration: 1 week · People: Team · Predecessors: 3.5, 4.3
- **5.2 — Test end-to-end commands**
  - Schedule: W11 · Duration: 1 week · People: Team · Predecessors: 5.1
- **5.3 — Test invalid/contradictory LTL**
  - Schedule: W12 · Duration: 1 week · People: Team · Predecessors: 5.2, 2.4
- **5.4 — Evaluate constraint compliance**
  - Schedule: W12 · Duration: 1 week · People: Team · Predecessors: 5.2, 2.4
- **5.5 — Record demonstrations**
  - Schedule: W13 · Duration: 1 week · People: Team · Predecessors: 5.3, 5.4

## 6. Environment Scale-up

**Duration:** 2 weeks · **People:** Akshith / Yahya · **Predecessors:** 5 · **Group bar:** W14–W15

- **6.1 — Configure 6-agent environment**
  - Schedule: W14 · Duration: 1 week · People: Akshith / Yahya · Predecessors: 5.5
- **6.2 — Adapt observation space**
  - Schedule: W15 · Duration: 1 week · People: Akshith / Yahya · Predecessors: 6.1
- **6.3 — Per-agent actions and rewards**
  - Schedule: W16 · Duration: 1 week · People: Akshith / Yahya · Predecessors: 6.2
- **6.4 — Smoke test with stock agents**
  - Schedule: W17 · Duration: 1 week · People: Akshith / Yahya · Predecessors: 6.3

## 7. Team Roles and Coordination

**Duration:** 4 weeks · **People:** Team · **Predecessors:** 6 · **Group bar:** W18–W21

- **7.1 — Define roles**
  - Schedule: W19 · Duration: 1 week · People: Team · Predecessors: 6.4
- **7.2 — Map roles to policies**
  - Schedule: W20 · Duration: 1 week · People: Akshith / Yahya · Predecessors: 7.1
- **7.3 — LTL role statements**
  - Schedule: W20 · Duration: 1 week · People: Armin / Zawadul / Tuga · Predecessors: 7.1
- **7.4 — LTL coordination vocabulary**
  - Schedule: W19 · Duration: 1 week · People: Armin / Zawadul / Tuga · Predecessors: 6.4
- **7.5 — Implement coordination in LTL**
  - Schedule: W20 · Duration: 1 week · People: Armin / Zawadul / Tuga · Predecessors: 7.4

## 8. Multi-agent Training

**Duration:** 7 weeks · **People:** Akshith / Yahya · **Predecessors:** 7 · **Group bar:** W21–W27

- **8.1 — Choose MARL paradigm**
  - Schedule: W21 · Duration: 1 week · People: Akshith / Yahya · Predecessors: 7
- **8.2 — Initial training runs**
  - Schedule: W21–W22 · Duration: 2 weeks · People: Akshith / Yahya · Predecessors: 7.2, 7.5
- **8.3 — Iterate to convergence**
  - Schedule: W23–W25 · Duration: 3 weeks · People: Akshith / Yahya · Predecessors: 8.2
- **8.4 — Checkpoint selection and evaluation**
  - Schedule: W26 · Duration: 1 week · People: Akshith / Yahya · Predecessors: 8.3
- **8.5 — Pure MARL baseline setup**
  - Schedule: W26 · Duration: 1 week · People: Akshith / Yahya · Predecessors: 8.3
- **8.6 — Baseline training**
  - Schedule: W21–W23 · Duration: 3 weeks · People: Akshith / Yahya · Predecessors: 6.4
- **8.7 — Baseline evaluation**
  - Schedule: W24 · Duration: 1 week · People: Akshith / Yahya · Predecessors: 8.6

## 9. Integration and Evaluation

**Duration:** 3 weeks · **People:** Team · **Predecessors:** 8 · **Group bar:** W27–W29

- **9.1 — Integrate symbolic agents**
  - Schedule: W27 · Duration: 1 week · People: Team · Predecessors: 8.4, 8.7
- **9.2 — Round-robin tournament simulation**
  - Schedule: W28 · Duration: 1 week · People: Team · Predecessors: 9.1
- **9.3 — Simulate possible ablations**
  - Schedule: W28–W29 · Duration: 2 weeks · People: Team · Predecessors: 9.1

## 10. Research Paper

**Duration:** 6 weeks · **People:** Team · **Predecessors:** 9.2, 9.3 · **Group bar:** no distinct bar visible against the green group-row background

- **10.1 — Related works draft**
  - Schedule: W25–W27 · Duration: 3 weeks · People: Team · Predecessors: 8
- **10.2 — Methodology section**
  - Schedule: W28–W29 · Duration: 2 weeks · People: Team · Predecessors: 10.1
- **10.3 — Experiments and Results section**
  - Schedule: W30 · Duration: 1 week · People: Team · Predecessors: 9.2, 9.3
- **10.4 — Reviews and Revisions**
  - Schedule: W31 · Duration: 1 week · People: Team · Predecessors: 10.3

## 11. Final Delivery

**Duration:** 2 weeks · **People:** Team · **Predecessors:** 9 · **Group bar:** W30–W31

- **11.1 — 3v3 Live Demo Preparation**
  - Schedule: W30 · Duration: 1 week · People: Team · Predecessors: 9.2, 9.3
- **11.2 — Final Documentation**
  - Schedule: W31 · Duration: 1 week · People: Team · Predecessors: 10.4, 11.1
- **11.3 — Capstone handoff**
  - Schedule: W32 · Duration: 1 week · People: Team · Predecessors: 11.2

## Source schedule notes

These observations describe the supplied chart. No durations, bars, or dependencies have been adjusted to reconcile them.

- Group 1's bar ends at W2; activity 1.5 is at W3.
- Group 2's bar ends at W4; activities 2.3 and 2.4 are at W5.
- Group 3's bar ends at W9; activity 3.4 continues through W10.
- Group 4's bar covers W5–W11; its activities continue through W16.
- Group 5's bar ends at W12; activity 5.5 is at W13. The group lists 4.6 as a predecessor even though 4.6 is scheduled at W14–W15 and group 5 begins at W10.
- Group 6's bar ends at W15; its activities continue through W17.
- Group 7's bar covers W18–W21; its activity bars occupy W19–W20.
- Group 8's bar ends at W27; its latest activity bars end at W26. Activity 8.1 lists group 7 as a predecessor and is scheduled at W21, which overlaps group 7's bar.
- Group 10 lists a duration of 6 weeks; its activity bars span W25–W31, covering 7 weeks. Its group-level predecessors are 9.2 and 9.3, while activity 10.1 begins at W25 and lists group 8 as its predecessor; group 8's bar continues through W27.
- Group 11's bar ends at W31; activity 11.3 is at W32. Activity 11.2 lists 10.4 as a predecessor, and both are scheduled at W31.
- The second image omits the week-header row. Its week labels are transcribed by continuing the W1–W32 column order from the first image.
