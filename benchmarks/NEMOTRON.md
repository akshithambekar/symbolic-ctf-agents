# Nemotron CTF translation benchmark

This benchmark runs the CTF instruction set in `ctf_instruction_set.json` through
the repository's current translator and finite-trace validator. It compares model
responses with the authored references over all nonempty finite traces. The gold
interpretations are project-authored and still need independent human review.

## Configure and inspect

Put the OpenRouter key in the repository-root `.env` file, or export it in the
shell. The runner reads only `OPENROUTER_API_KEY` and never writes it to results.

```dotenv
OPENROUTER_API_KEY=your-key
```

List the Nemotron model shortcuts without making API calls:

```bash
PYTHONPATH=src python -m benchmarks.nemotron --list
```

Available aliases are `nano-9b-v2`, `nano-12b-v2-vl`, `nano`, `nano-omni`,
`super`, `ultra`, `lightning`, and `super-vl`. A run defaults to the Nemotron 3
size trio (`nano`, `super`, `ultra`)
with three independent trials. Specify other models or run all eight with `--models`:

```bash
PYTHONPATH=src python -m benchmarks.nemotron --models nano super ultra
PYTHONPATH=src python -m benchmarks.nemotron --models all --trials 3
```

Each new run writes a uniquely named directory under `benchmarks/results/nemotron/`.
It records the model IDs, resolved provider model, request settings, raw attempts,
translation outputs, semantic assessments, aggregate CSV/JSON, paraphrase checks,
and a report. Raw results are ignored by Git. If OpenRouter reaches a rate limit or
a provider becomes unavailable, measured cases remain saved. Resume the run with
the same model selection and trial count:

```bash
PYTHONPATH=src python -m benchmarks.nemotron --models all --trials 3 \
  --output benchmarks/results/nemotron/RUN_DIRECTORY --resume
```

OpenRouter free variants are rate-limited, and unsuccessful calls may count toward
the limit. The default call timeout is 120 seconds. Use `--pause SECONDS` to space
requests further apart. If a free endpoint returns HTTP 404 or 429, the runner tries
that model's corresponding paid route unless you pass `--stop-on-http-errors`. Paid
fallback can incur charges. The result records both model IDs and estimates cost from
recorded usage and listed prices when possible.

The runner makes requests in sequence and waits at least 3.1 seconds between them.
A per-key file lock makes other benchmark processes on this computer wait for the
active run to finish. Use `--stop-on-http-errors` to stop on the first 404 or 429
without trying the paid route.

## Run the benchmark

The completed three-trial comparison uses the four routes that responded during
route checks. It excludes the unavailable 9B v2 and 12B v2 VL routes. Nano's free
route returned 404, and Super's free route returned 404. Earlier paid-route checks
received one Nano response and a 429 for Super. See [results](NEMOTRON_RESULTS.md).

To repeat the completed comparison without paid fallbacks, run:

```bash
PYTHONPATH=src ./.venv/bin/python -m benchmarks.nemotron \
  --models nano-omni ultra lightning super-vl --trials 3 \
  --stop-on-http-errors
```

If a run stops or encounters a connection error, resume the same model set and trial
count with the printed result directory. The runner skips measured cases and retries
pending ones:

```bash
PYTHONPATH=src ./.venv/bin/python -m benchmarks.nemotron \
  --models nano-omni ultra lightning super-vl --trials 3 \
  --output benchmarks/results/nemotron/RUN_DIRECTORY --resume \
  --stop-on-http-errors
```

Results remain local because `benchmarks/results/` is gitignored. The tracked
summary is in [results](NEMOTRON_RESULTS.md).

## Metrics

- **Syntax-valid rate:** formula-expected items that parse as supported LTL.
- **Compilable rate:** formula-expected items that compile to a nonempty finite-trace
  DFA.
- **Semantic-correct rate:** formula-expected items equivalent to the authored
  reference; a failed or invalid formula counts as incorrect.
- **Appropriate-rejection rate:** ambiguous or unsupported items explicitly rejected.
- **Repair rate:** measured items for which the translator made at least one
  validation repair request.
- **Latency:** end-to-end translator time, including validation repairs.
- **Estimated cost:** token usage multiplied by the listed input/output price for
  paid fallback attempts; free-route requests are priced at zero.

Formula equivalence uses the project's DFA product checker, not string equality or
sampled traces. A semantic match proves agreement with the authored formula under
the benchmark semantics. It does not validate that the formula captures every
reasonable reading of its English instruction. Treat reviewer approval and
counterexamples in `raw.jsonl` as part of interpretation.
