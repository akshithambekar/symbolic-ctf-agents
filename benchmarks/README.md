# English-to-LTL translation smoke test

Stage 1 runs our restricted AST implementation on the single instruction in `cases.json`: "Eventually reach object_7."
The reference is `F reach(object_7)`, with `reach(object_7)` as one atomic proposition.
Under standard LTL over infinite traces, `F` includes the current step.
The proposition may hold now or later.
This is a pipeline smoke test, not a measurement of translation accuracy.
Semantic checking is deferred to Stage 2, and every result explicitly records `semantic_evaluation.outcome: "not_run"`.
No checker installation is needed for Stage 1.

`comparison_cases.json` provides four shared examples for manual inspection, including the smoke case, avoidance, reaching both objects, and reaching either object.
Run them with `python -m benchmarks.run --cases benchmarks/comparison_cases.json`.
This makes one live request per case and still runs only the team translator.
Do not use `--offline` to assess these cases: the offline model returns the same canned response for every instruction.
These examples are not a validated accuracy dataset, and semantic evaluation remains `not_run`.

## Setup and offline verification

Use the project virtual environment or install the Python requirements:

```bash
python -m pip install -r benchmarks/requirements.txt
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
The exit status is 0 for successful translation and 1 for any recorded failure.

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

Stage 2 will add a separate maintained parser and equivalence checker, explicit notation conversion, and independent equivalence tests.
No string comparison or semantic score is performed in Stage 1.

## Implementation identity and deferred baselines

The benchmark directly imports `translators.team_translator.translator` and records `implementation_id: "restricted-ast-v1"` in every result, including configuration and provider failures.
The ID identifies the implementation variant separately from the provider's `model_id`.
Existing result fields, schema version, and failure categories are preserved; the identity field is additive.
Older local JSON files are left untouched and may lack implementation attribution.
The stable ID does not replace a future source revision/configuration fingerprint for reproducible comparisons.

The original migrated LTLCodeGen code is restored under `translators/ltl_codegen/` and has no selectable benchmark execution path.
Its generated-Python executor needs a verified isolation design before it can be benchmarked with model output.
IBM NL2LTL is only referenced and remains unintegrated.
See the [implementation inventory and history audit](../translators/README.md) for provenance, deliberate import changes, mutable settings, and baseline decisions.

The one shared case and its reference remain independent of translator code.
A future shared evaluator must normalize each implementation's native output and the infix reference into one maintained checker's notation, including a consistent single-proposition encoding of `reach(object_7)`.
It must use standard infinite-trace LTL, with `F` including the current step, and distinguish invalid formulas, checker errors, semantic mismatch, and equivalence.
Until that evaluator is integrated and independently tested, all results preserve `semantic_evaluation.outcome: "not_run"`.
No string-based semantic comparison is performed.
