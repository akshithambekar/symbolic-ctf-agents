# Maritime CTF Translation Benchmark — candidate edition

This prepares the dataset and evaluation design for a possible paper extension.
It does not change the published-draft pilot results or claim independent review
has occurred. The neutral dataset title is separate from its provenance metadata.

## Files ready for review

- `candidate_v1/dataset.json`: **120 instructions / 60 paired semantic families**.
- `candidate_v1/splits.json`: **40 development items and 80 test candidates**.
- `candidate_v1/reviewer_A.csv` and `reviewer_B.csv`: independent first-pass forms,
  with instructions but no proposed answers. Reviewer identities and judgments
  are blank. There are no fabricated approvals.
- `candidate_v1/context_review.csv`: approval/rejection of the predicate scope and
  proposed grounding invariant by the two reviewers.
- `candidate_v1/reference_review.csv`: candidate formulas, rationale, sources,
  and machine-generated satisfying/violating abstract traces for second-pass review.
- `candidate_v1/adjudication.csv`: worksheet for disagreements; it is not consumed
  as an automatic approval by the current tool.
- `sources.json`: source types, URLs, provenance and adaptation limits.
- `PROTOCOL.md`: intended experiment and reporting rules before model calls.

The 40 historical cases are development-only. The 80 new instructions form 40
new families: 32 formula families and eight ambiguity/unsupported families.
There are 96 formula-expected and 24 clarification/rejection items overall.
All proposed formulas compile, and no test formula is equivalent to an exposed
historical formula or a different test family under the proposed profile.
This is a compositional same-vocabulary split, not an unseen-domain evaluation.

## Source-backed examples

The official MCTF rules motivate tagging, boundary, recovery and Powerplay cases.
PyQuaticus's official attacker/defender/combined policies motivate tactical
conditions. A general CTF/LTL research paper supplies concepts involving goals,
territory and opponent proximity. These sources do **not** provide ready-made
English/LTL gold labels for our PyQuaticus profile. Wording and candidate
formalizations are our own adaptations and must be reviewed.

Some source ideas deliberately become unsupported-request tests. For example,
four-meter collision avoidance cannot be represented using `tagged`, and a
particular carrier returning to an official corner base cannot be replaced by
the existing team-level flag-stand `home` proposition. An exact per-agent rule
must not silently become a stronger or weaker aggregate team constraint.

## Important grounding issue

The simulator defines `teammate_safe` as the absence of any controlled tagged
boat. Thus it is the complement of `tagged`. Treating both as independent Boolean
variables can incorrectly reject otherwise equivalent translations. This draft
proposes the explicit trace invariant in `dataset.json` and asks reviewers to
approve it. The planned study reports both invariant-constrained agreement and
the original unconstrained Boolean metric; neither proves physical realizability.
No historical scores have been recomputed under this new profile.

## Commands — no model/API calls

From the repository root with the paper environment:

```powershell
.\.venv-paper\Scripts\python.exe paper_benchmark/study/manage.py check
.\.venv-paper\Scripts\python.exe paper_benchmark/study/manage.py review-status
```

`check` verifies hashes, grouping, compileability, and semantic overlap. Review
forms are excluded from the immutable draft manifest so people can fill them in.
`review-status` checks declared distinct identities, decisions and formula
agreement; it cannot authenticate humans or certify annotation quality.

To reconstruct a separate draft rather than overwrite this one:

```powershell
.\.venv-paper\Scripts\python.exe paper_benchmark/study/manage.py build --directory paper_benchmark/runs/new-candidate-draft
```

Existing outputs are refused. Changing references or split membership requires
a new version and manifest. No benchmark run on the 80 test candidates has been
performed. The current `paper_benchmark/run.py compare` continues to use the
original 40-item dataset; this candidate dataset is not connected to that live
runner yet. Integrate a reviewed, versioned release only after adjudication.

## Next human action

Give one reviewer `reviewer_A.csv`, the other `reviewer_B.csv`, and give both
the glossary/profile. Ask them to classify each instruction and supply their
own formula when expressible, before consulting `reference_review.csv`.
Then compare the independent interpretations, review the proposed answers and
abstract traces, and document disagreements. The final reviewed release and
experiment settings need a separate explicit freeze before evaluation.
