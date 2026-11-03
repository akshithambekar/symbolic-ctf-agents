# Restoration verification — November 3, 2026

- Original source/dataset manifest: all 25 entries match SHA-256.
- Complete preserved experiment: all 35 files match the additional manifest.
- Raw outcomes: 120 per translator, 240 total; 96 formula-reference outputs per
  translator. Summary values independently recomputed from those records match.
- Offline finite-trace DFA re-scoring: all 240 saved assessments match, with zero
  differences. Recomputed CSV/JSON summaries match the historical values.
- Legacy normalization, temporal equivalence, ordered-goal counterexamples,
  generated-code restrictions, rejection handling, and mocked provider repair
  tests: 11 passed.
- Restoration integrity and isolated execution checks: 2 passed.
- New live model/API calls: zero.

Checks used a separate Python 3.13.3 environment with the saved translation
dependency versions. `.venv-paper/` and generated verification runs are ignored.
The complete historical artifacts, not these temporary environments, are committed.
The checked-in original report and raw records were not rewritten.

Re-run the verification commands in `README.md`. Fresh live measurements will
depend on provider availability and must be stored as new experiments. IBM NL2LTL
and new-model comparisons remain future extensions, not completed results.
