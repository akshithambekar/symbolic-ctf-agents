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
the limit. The runner stops after a 429 response and leaves the rest pending. The
default call timeout is 120 seconds. Use `--pause SECONDS` to space requests further
apart. If a free endpoint returns HTTP 404 or 429, the runner retries that model on
its corresponding paid route and uses that route for its remaining cases. This can
incur charges. The result records both model IDs and estimates cost from recorded
usage and listed prices when possible.

## Complete a benchmark run

The initial run is incomplete. It measured three responses out of 960 scheduled
model, case, and trial outcomes. OpenRouter reported that the 9B v2 free and paid
routes no longer have a provider. It gave the same response for the 12B v2 VL free
and paid routes. The Nano free route returned 404, then its paid route produced one
measured response. The Nano Omni and Ultra free routes each produced one measured
response. The Super free route returned 404, and its paid route returned 429. The
remaining models have no measured cases.

The small 9B and 12B v2 aliases are unavailable on OpenRouter. Skip them in the full
run. First test the remaining unmeasured routes one at a time:

```bash
PYTHONPATH=src ./.venv/bin/python -m benchmarks.nemotron --models lightning --trials 1
PYTHONPATH=src ./.venv/bin/python -m benchmarks.nemotron --models super-vl --trials 1
PYTHONPATH=src ./.venv/bin/python -m benchmarks.nemotron --models super --trials 1
```

Each command starts a new run. The runner tests up to 40 instructions for the model
unless OpenRouter returns an error that stops that model or the whole run. Review
the printed result directory and its `summary.json` before continuing. Paid fallbacks
can incur charges.

After the routes respond, run all available models for three trials. Use the aliases
that succeeded in the single-trial checks. For example:

```bash
PYTHONPATH=src ./.venv/bin/python -m benchmarks.nemotron \
  --models nano nano-omni super ultra lightning super-vl --trials 3
```

If a run stops on a rate limit, wait for the OpenRouter limit to reset, then resume
the same model set and trial count with the printed result directory:

```bash
PYTHONPATH=src ./.venv/bin/python -m benchmarks.nemotron \
  --models nano nano-omni super ultra lightning super-vl --trials 3 \
  --output benchmarks/results/nemotron/RUN_DIRECTORY --resume
```

The runner skips measured cases and retries pending ones. Results remain local because
`benchmarks/results/` is gitignored. See [partial results](NEMOTRON_RESULTS.md) for
the observed errors and measured sample counts.

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
