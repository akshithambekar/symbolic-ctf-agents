# English-to-LTL translation smoke test

Stage 1 runs our restricted AST implementation on the single instruction in `cases.json`: "Eventually reach object_7."
The reference is `F reach(object_7)`, with `reach(object_7)` as one atomic proposition.
Under standard LTL over infinite traces, `F` includes the current step.
The proposition may hold now or later.
This is a pipeline smoke test, not a measurement of translation accuracy.
Successful translations now pass through one shared Spot evaluator over infinite traces.
Failed translations retain `semantic_evaluation.outcome: "not_run"`.
Install the checker and read the [semantic evaluation guide](SEMANTICS.md) for grammar, limits, result categories, and license/platform details.

`comparison_cases.json` provides four shared examples for manual inspection, including the smoke case, avoidance, reaching both objects, and reaching either object.
Run them with `python -m benchmarks.run --cases benchmarks/comparison_cases.json`.
This makes one live request per case using the team translator by default; select `--implementation ltlcodegen-container` to use the isolated baseline.
Do not use `--offline` to assess these cases: the offline model returns the same canned response for every instruction.
These examples are not a validated accuracy dataset.
The fixed-response integration tests exercise them with both implementations and shared semantic checking.

## Setup and offline verification

Use the project virtual environment or install the Python requirements:

```bash
python -m pip install -r benchmarks/requirements.txt
brew install spot  # macOS/Linux Homebrew; see SEMANTICS.md for other installs
# Also install legacy dependencies to run the baseline restoration tests:
python -m pip install -r translators/ltl_codegen/requirements.txt
python -m pytest -q
python -m benchmarks.run --list
python -m benchmarks.run --offline
```

`--list` displays cases without initializing a provider client.
`--offline` uses a canned fake-model response and writes a result with zero provider requests.
The tests exercise fenced and unfenced responses, unsafe-code rejection, JSON redaction, configuration errors, and the real OpenAI SDK over a fake HTTP transport.
They verify that HTTP 429 and HTTP 503 responses cause only one request, with no retries.
These tests establish pipeline behavior, not live translation accuracy.

## One live attempt

Create a `.env` file in the repository root with your OpenRouter key and chosen model ID:

```dotenv
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=provider/model-name
```

The file is ignored by Git and is read automatically using `python-dotenv`.
Existing environment variables take precedence over `.env`, and `--model` takes precedence over both.
Install or update the benchmark dependencies using the setup command above, then run:

```bash
python -m benchmarks.run
# Alternatively:
python -m benchmarks.run --model 'your-provider/your-model-id'
```

There is no default live model.
Both an explicit model ID and a nonempty `OPENROUTER_API_KEY` are required.
Missing configuration is recorded as `configuration_error`, with zero attempts and a message identifying what is missing.
The exit status remains 0 when all translations succeed and 1 for translation/provider/configuration failures.
Semantic mismatches and checker errors do not change this status; inspect `semantic_evaluation.outcome`.

The client uses OpenRouter's [OpenAI-compatible endpoint](https://openrouter.ai/docs/quickstart), a temperature of 0, a 1024-token output limit, and a 60-second timeout.
Our restricted translator calls the model once per case and rejects attempts to enable retries.
The SDK is configured with `max_retries=0` and HTTP redirects disabled.
OpenRouter [provider fallbacks](https://openrouter.ai/docs/guides/routing/provider-selection) are disabled with `provider.allow_fallbacks=false`.
There is no retry, alternate model, or local fallback after a failure.
`provider_requests` counts client completion calls; it cannot observe a provider's internal processing.

## Translation boundary

The model receives our copy of the migrated few-shot prompt with added AST response constraints and returns Python-shaped text.
The interpreter accepts one `def question():`, an optional docstring, local formula assignments, and a final return.
Expressions can refer to previously assigned formula variables and call `ap(reach, 'object_7')` or approved LTL operators.
Supported operators are `ltl_and`, `ltl_or`, `ltl_not`, `ltl_next`, `ltl_until`, `ltl_eventually`, `ltl_always`, and `ltl_imply`.
Object arguments must be literal identifiers.
Imports, attributes, indexing, decorators, annotations, loops, conditionals, arbitrary expressions, arbitrary calls, and surrounding prose are rejected.
The interpreter never compiles, executes, imports, or writes model-generated Python.
Code length, AST node count, and formula expansion are bounded.

Both plain code and one Markdown Python code fence are supported.
For example:

```python
def question():
    target = ap(reach, 'object_7')
    return ltl_eventually(target)
```

With our translator's default listed-prefix format, this produces `['EVENTUALLY', 'reach(object_7)']`.
The quoted `reach(object_7)` is one proposition, and the first element is a temporal operator.
`translated` means the restricted interpreter produced a formula.
It does not establish that the formula matches the English instruction or reference.

## Inspecting results

A run prints only the result file path, never the result payload or generated code.
Results default to `benchmarks/results/<timestamp>-<unique-id>.json`, which Git ignores.
Open that JSON file in an editor to inspect the case, `implementation_id`, model ID, settings, raw generated code, formula, attempt and request counts, outcome, failure details, and elapsed seconds.
Use `--output PATH` to choose another result path; keep custom output outside tracked files.
Use `--cases PATH` to select a different case list.

Outcomes distinguish `translated`, `translation_error`, `provider_error`, `rate_limit_error`, `configuration_error`, and unexpected local `runner_error` failures.
Rejected generated code is retained with the interpreter's reason, while provider failures retain the exception type, HTTP status when available, and a fixed diagnostic message.
Provider exception bodies, request objects, headers, and raw exception text are omitted.
The configured API key is redacted from every serialized string, including echoed generated code and error details.

Stage 2 uses Spot's parser and infinite-trace equivalence filter on normalized formula data.
See [SEMANTICS.md](SEMANTICS.md) for independent lecture-property tests, deterministic validation precedence, and error/timeout handling.

## Implementation identity and deferred baselines

The benchmark directly imports `translators.team_translator.translator` and records `implementation_id: "restricted-ast-v1"` in every result, including configuration and provider failures.
The ID identifies the implementation variant separately from the provider's `model_id`.
Existing result fields and translation/provider failure categories are preserved.
New runs use `schema_version: 2` to identify the populated semantic evaluation structure and its new outcomes, with `semantic_timeout_seconds` added to settings.
Older local JSON files are left untouched and may lack implementation attribution.
The stable ID does not replace a future source revision/configuration fingerprint for reproducible comparisons.

The migrated LTLCodeGen baseline is available as `ltlcodegen-container-v1` through the isolated container runner described below.
IBM NL2LTL is only referenced and remains unintegrated.
See the [implementation inventory and history audit](../translators/README.md) for provenance, deliberate import changes, mutable settings, and baseline decisions.

The one shared case and its reference remain independent of translator code.
The shared evaluator converts both implementations' flat listed-prefix output and the infix reference into fully parenthesized Spot syntax.
It quotes each complete proposition spelling, so `reach(object_7)` is one atom even when undeclared in case metadata.
Evaluation distinguishes equivalence, mismatch, invalid generated/reference formulas, checker errors, and whole-evaluation timeouts.
No string-based semantic comparison or sampled-trace substitute is performed.

## Isolated LTLCodeGen baseline

Build the image with Docker Desktop running, then run the controlled isolation checks before using live model output:

```bash
python -m benchmarks.ltlcodegen_runner.runner
LTL_DOCKER_TESTS=1 python -m pytest -q tests/test_ltlcodegen_isolation.py
python -m benchmarks.run --implementation ltlcodegen-container --offline
python -m benchmarks.run --implementation ltlcodegen-container
```

The final command makes one live OpenRouter request per case using the existing configuration above.
The original baseline prompt is formatted on the host, and only the response is sent to the container on standard input.
The executor runs in a fresh container with no network or host mounts, a non-root user, a read-only root filesystem, dropped capabilities, a bounded temporary filesystem, a 512 MiB memory limit, a 64-process limit, one CPU, and a 20-second execution deadline.
The baseline writes its generated module into a temporary copy of the imported source inside the container.
Docker Desktop and its isolation mechanisms must be functioning correctly; containers are not a guarantee against kernel or runtime vulnerabilities.
The image build uses the imported dependency ranges, which are not pinned and may resolve differently in future builds.
Results from arbitrary generated code should be treated as untrusted, including any formula or error it produces.
The baseline and provider retries are disabled for benchmark runs; the existing baseline implementation is unchanged.
Successful baseline translations use the same host-side semantic evaluator as the team translator; failed translations retain `not_run`.
