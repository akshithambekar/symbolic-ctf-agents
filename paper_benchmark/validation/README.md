# IBM integration evidence — November 3, 2026

The two archived runs below contain four real provider calls in total. Both used
the paper's `nvidia/nemotron-3-ultra-550b-a55b:free` model and the same two CTF cases.
They are **integration smoke tests**, not a completed translator-quality study.

- `ibm-native-200-20261103-v1`: native 200-token limit. Both calls ended with
  `finish_reason: length`, with all 200 completion tokens consumed by reasoning
  and no final pattern text. Both outputs were retained as unusable.
- `ibm-tokens-4096-20261103-v1`: explicit larger-token variant. Both calls returned
  usable, reference-equivalent Absence/Existence formulas. The implementation ID
  records `tokens-4096`; these outcomes are not pooled with the native-cap run.

Each archive preserves raw output and usage, request payloads, configuration,
implementation/model IDs, summaries, review sheets, plots, source hashes, and
snapshots. The archive manifest covers exact saved bytes, including line endings.
Original September artifacts remain unchanged in `../reference_run/`.

The current implementation passed 13 outer pytest cases, including the fake-HTTP
three-method comparison and the subprocess that runs the original 11 legacy tests.
It also passed historical SHA-256 and raw-summary verification. The full new
40-case × three-trial × three-method experiment has not been launched.
