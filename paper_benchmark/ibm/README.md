# IBM NL2LTL comparison arm

Implementation ID: `ibm-nl2ltl-0.0.6-ctf-openrouter-v1`.

This adapter uses the real **IBM NL2LTL 0.0.6** package, not a newly invented
direct-English-to-LTL prompt bearing IBM's name. It invokes the inherited
`GPTEngine.translate` path through `nl2ltl.translate`, its native output parser,
DECLARE-pattern grounding, and each template's `to_ltlf()` constructor.

## Exact evaluated portion and adaptations

1. The upstream few-shot pattern-classification prompt is preserved in
   `upstream/prompt.json`, with its MIT license and pinned Git revision recorded
   in `provenance.json`. The PyPI wheel omitted that prompt file, so the source
   copy is supplied explicitly.
2. A CTF glossary and explicit refusal protocol are appended, without adding gold
   formulas or reference answers to requests. The appended instructions expose
   all eight DECLARE templates actually implemented by the library; the upstream
   prompt's examples originally list only five. The original prompt stays intact.
3. A small `GPTEngine` subclass replaces its provider/model consistency gate so
   the same NVIDIA/OpenRouter model can be evaluated. The parent's request
   construction, native sampling/token settings, output parsing, grounding, and
   formula construction remain in use. A recorded HTTP transport makes the live
   request; the library runs offline in a separate worker with a stub transport.
4. Official Pylogics AST objects are converted structurally to the shared scorer's
   notation. Strong `Next` becomes the scorer's strong `X`; unsupported AST nodes
   are rejected rather than silently reinterpreted.
5. `UNSUPPORTED: explanation` is a domain-adapter refusal, not an upstream native
   output format. It is recorded as such. Multiple equally ranked different
   candidates are not selected using the gold formula; all candidates are kept
   and the adapter reports ambiguity if there is no unique top formula.

This is an **IBM-based CTF/OpenRouter adaptation**, not a reproduction of the
original paper's GPT-3.5/GPT-4 setup. It does not use IBM's Rasa engine or retrain
the library. `provenance.json` records the PyPI versions, upstream prompt revision,
and installed IBM module hashes; `requirements.lock` pins the isolated runtime.

## Capability limits and pinned upstream behavior

The engine selects one of Existence, ExistenceTwo, Absence, RespondedExistence,
Response, Precedence, ChainResponse, and NotCoExistence. It is not an unrestricted
generator of arbitrary nested conjunctions and missions. A refusal on a
formula-expected item counts as unusable under the shared benchmark; it is not
automatically credited as a correct translation. The same reference interpretations
and human review requirements apply to all three methods.

The native 0.0.6 grounders consume the first symbol for unary templates and the
first two for most binary templates; additional symbols can be dropped. Raw
symbols, chosen templates, and constructed candidates are retained for review.
The pinned `ground_notcoexistence` implementation uses its second symbol twice.
These are observed upstream behaviors, preserved rather than silently repaired.
Semantic scoring, not the template name or successful parsing, determines agreement
with the reference. No claim is made that every supported pattern is implemented
correctly upstream.

## Isolated setup

The paper benchmark itself uses Python 3.13. IBM 0.0.6 requires Python below 3.11
and pins OpenAI 1.12.0, so install it separately:

```powershell
uv venv --no-project --python 3.10 .venv-ibm
uv pip install --python .venv-ibm/Scripts/python.exe -r paper_benchmark/ibm/requirements.lock
```

On Linux/macOS, use `.venv-ibm/bin/python`. The parent selects the appropriate
default path; `compare --ibm-python PATH` can select a different interpreter.
The worker receives no API credential, performs no network request, and never
executes generated model code. Missing/broken worker dependencies are infrastructure
failures and remain `Not measured`, distinct from invalid model output.

## Matched new comparison

After configuring the repository `.env`, use the paper environment's interpreter:

```powershell
.\.venv-paper\Scripts\python.exe paper_benchmark/run.py compare
```

Defaults: original 40 cases, three trials, all three methods, and the original
`nvidia/nemotron-3-ultra-550b-a55b:free` model. This schedules **360 outcomes**
plus the two original methods' validation repairs. New results go in a fresh
ignored `paper_benchmark/runs/compare-...` directory. September results are never
pooled into this run. The order of the three methods rotates across cases/trials.

For a small smoke test, not a quality study:

```powershell
.\.venv-paper\Scripts\python.exe paper_benchmark/run.py compare --methods ibm --trials 1 --case-id capture-0 --case-id tag-0
```

`--methods current ltlcodegen ibm`, repeated `--case-id`, `--env-file`, `--output`,
`--model`, `--endpoint`, `--timeout`, `--ibm-python`, and `--ibm-max-tokens` make conditions explicit.
There is no application-level paid fallback or alternate model selection.

### Reasoning-model token budget

The default native 200-token limit may be consumed entirely by reasoning, leaving
no final pattern text. This occurred in the recorded Nemotron smoke test. It is
retained as an unusable measured output, with the raw finish reason and token usage.
For an explicitly adapted larger-budget run:

```powershell
.\.venv-paper\Scripts\python.exe paper_benchmark/run.py compare --methods ibm --trials 1 --case-id capture-0 --case-id tag-0 --ibm-max-tokens 4096
```

For all three methods with that IBM cap, omit `--methods` and the case filters.
An override receives a separate implementation ID ending in `tokens-4096` and is
recorded alongside native settings. It must not be described as an unmodified
200-token IBM setup. It still uses one provider call, not a repair loop.

## Fairness and metrics

The same natural-language input, glossary, underlying model, finite-trace scorer,
and reference interpretation are used. IBM keeps its native one-call behavior,
temperature 0.5, 200-token output cap, and stop sequence. The two frozen paper
adapters retain their original provider defaults and maximum 3/4 calls.
This compares complete adapted workflows with different native request budgets;
it is not an equal-token/equal-call ablation. IBM's zero repair rate is mechanical,
not evidence of greater reliability. An equal-budget ablation would be a separate
explicitly named experiment.

## Recorded live integration checks

The November 3 checks used the same free Nemotron model on two simple cases:

| IBM token cap | Requests | Usable / reference-equivalent outputs | Observation |
| --- | --- | --- | --- |
| Native 200 | 2 | 0 | All completion tokens were reasoning; no final text |
| Explicit 4096 | 2 | 2 | Native pattern parser produced correct Absence/Existence formulas |

These four requests are smoke checks, not an accuracy benchmark or a statistical
comparison. The full three-method matrix has not been launched. Raw outputs,
configuration, usage, source snapshots, and summaries are under
`../validation/ibm-native-200-20261103-v1` and
`../validation/ibm-tokens-4096-20261103-v1`.

Outputs include raw responses/requests, model metadata, source snapshots and hashes,
worker dependency provenance, validation/semantic results, errors, repairs, latency,
paraphrase comparisons, CSV/JSON summaries, a plot, and a human review sheet.
Formula-reference and rejection-reference denominators are reported separately.
Provider failures are excluded from quality denominators and counted as missing.
No superiority claim follows from smoke tests or a single trial.

## Verification

With both environments prepared:

```powershell
.\.venv-paper\Scripts\python.exe -m pytest paper_benchmark/tests -q
.\.venv-paper\Scripts\python.exe paper_benchmark/run.py verify
```

Tests cover official Existence/Absence/Response/ChainResponse/ExistenceTwo output,
strong next, malformed/refused output, native extra-symbol behavior, unsupported
predicates, provider/worker failures, unchanged model identity, secret redaction,
and an end-to-end three-method run against a local fake HTTP provider. The
original historical integrity/scoring checks remain in the suite.

Primary source: https://github.com/IBM/nl2ltl
