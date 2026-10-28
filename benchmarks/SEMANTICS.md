# Shared infinite-trace LTL evaluation

The benchmark uses [Spot](https://spot.lre.epita.fr/), a maintained LTL and omega-automata toolkit, through its `ltlfilt` executable.
Spot 2.16.1 was installed and tested for this integration on Apple Silicon macOS.
The [official release page](https://spot.lre.epita.fr/install.html) identifies that release as published on 2026-10-16.
Spot's [documented equivalence filter](https://spot.lre.epita.fr/ltlfilt.html) decides language equivalence over infinite traces, rather than comparing formula strings or finite trace samples.
Both `restricted-ast-v1` and `ltlcodegen-container-v1` use the same evaluator on the host after translation.
The translator implementations and their safety boundaries are unchanged.

## Installation and platform requirements

Install the Python benchmark dependencies and the external native checker separately:

```bash
python -m pip install -r benchmarks/requirements.txt
brew install spot
ltlfilt --version
python -m pytest -q tests/test_semantics.py
python -m benchmarks.run --offline
```

The [Homebrew formula](https://formulae.brew.sh/formula/spot) supplies Apple Silicon macOS and Linux packages and identifies its license as GPL-3.0-or-later.
For other Linux installations, follow Spot's [official Debian/Fedora, Conda, or source installation instructions](https://spot.lre.epita.fr/install.html).
Source builds require a C++20 compiler; Python bindings are not needed by this integration.
Do not substitute an unrelated PyPI package named `spot` for the native `ltlfilt` executable.
The benchmark invokes an externally installed program and does not bundle or link Spot's library.
Spot's [license documentation](https://spot.lre.epita.fr/#orgdf77a8b) explains GPL terms for distribution and linked derivatives.

`ltlfilt` must be on `PATH`, or set `LTLFILT` to its executable path in the process environment:

```bash
export LTLFILT=/opt/homebrew/bin/ltlfilt
```

The semantic worker requires POSIX process groups and resource limits, supported here on macOS and Linux.
Native Windows is not supported by this worker; use a POSIX environment with Spot installed.
Linux is supported by the design but was not exercised in this integration run.
A missing executable, failed version probe, unsupported platform, launch failure, or failed checking produces `checker_error`, never a semantic mismatch or a skipped check for a successful translation.
The semantic test suite requires a working checker and fails explicitly if it is missing.

## Input notation and proposition grammar

`formula_notation.py` converts a small, explicit input language into fully parenthesized Spot infix syntax.
Its parser handles notation only; Spot performs the independent LTL parsing and equivalence decision.
All inputs are treated as data.
No `eval`, Python literal evaluation, generated-code execution, shell interpretation, or lenient Spot parsing is used.

Both benchmark implementations emit flat listed-prefix strings by default, for example:

```text
['AND', 'EVENTUALLY', 'reach(object_1)', 'EVENTUALLY', 'reach(object_2)']
```

A prefix input must be one flat bracketed list of single- or double-quoted tokens, separated by commas.
Whitespace between tokens and an optional trailing comma are accepted.
Nested lists, unquoted values, escapes inside tokens, arbitrary expressions, extra operands, missing operands, and unknown operator tokens are rejected.
The operators have fixed arity:

| Prefix token | Reference infix | Arity |
| --- | --- | --- |
| `AND` | `&` | 2 |
| `OR` | `\|` | 2 |
| `NEGATION` | `!` | 1 |
| `NEXT` | `X` | 1 |
| `UNTIL` | `U` | 2 |
| `EVENTUALLY` | `F` | 1 |
| `ALWAYS` | `G` | 1 |
| `IMPLY` | `->` | 2 |

References use those infix symbols, parentheses, and lowercase `true` or `false` constants.
Prefix leaves can also be the quoted tokens `'true'` and `'false'`.
From strongest to weakest, precedence is unary `! X F G`, `U`, `&`, `|`, and `->`.
`U` and `->` associate to the right; `&` and `|` associate to the left.
Separate an operator from an identifier with whitespace or parentheses: write `G F p` or `G(F(p))`.
Compact `GFp` is an identifier, not a chain of temporal operators.
No PSL, finite-trace, past-time, release, weak-until, Unicode operator, or alternate symbolic syntax is accepted at this input boundary, even when Spot supports it.

Atomic propositions have this case-sensitive ASCII grammar:

```text
identifier  := [A-Za-z_][A-Za-z_0-9]*
proposition := identifier | identifier(identifier)
```

The exact operator names in the table, `F`, `G`, `X`, `U`, `true`, and `false` are reserved and cannot be used as proposition names or predicate names.
A predicate argument is one identifier; multiple arguments, nested predicates, spaces inside the predicate spelling, quotes, punctuation, and arbitrary Python expressions are unsupported.
Examples include `p`, `GoalReady`, `reach(object_7)`, and `visit(undeclared_object)`.
The case's `propositions` dictionary supplies descriptions, not an allowlist.
Syntactically valid undeclared propositions are accepted and participate in equivalence checking.
All propositions are independent Boolean variables; no physical action constraints or mutual exclusions are inferred.

The converter encloses each complete proposition spelling in double quotes for Spot.
For example, both generated `'reach(object_7)'` and reference `reach(object_7)` become `"reach(object_7)"`, one atomic proposition.
Bare `reach_object_7` becomes `"reach_object_7"` and remains a different proposition.
Quoting preserves spelling injectively and prevents names such as `GFp` from becoming temporal operators in Spot.
The accepted grammar excludes quote and escape characters, so formula text cannot break out of the quoted atom.
The common normalized strings and the sorted union of proposition spellings are recorded for inspection.

## Semantics, validation, and limits

Checking uses standard LTL over infinite traces.
`X` refers to the next step, `U` is strong until, and `F` includes the current step.
A proposition that holds only now satisfies `F p` but need not satisfy `X F p`.
Spot's strict parser is invoked on each normalized formula with `--ltl --count` before checking equivalence with `--equivalent-to`.
The evaluator accepts an equivalence result only for the documented return-status/count pairs: `(0, 1)` for equivalent and `(1, 0)` for mismatch.
Nonzero parse errors, native crashes, unexpected output, or checker diagnostics are checker errors.
No sampled traces, syntactic simplification heuristic, or string-equality fallback decides the result.

The default deadline is 10 seconds for the entire evaluation of one pair, including worker startup, notation conversion, validation, native parsing, checker initialization, and equivalence checking.
The host starts a fresh worker process group and kills the group on timeout, including a running `ltlfilt` child.
The deadline is measured from entry into `evaluate`, rather than reset for individual checker calls.
Process launch, operating-system scheduling, and cleanup add small overhead to the observed elapsed time; this is not a hard real-time scheduling guarantee.
The Python API accepts a positive finite `timeout_seconds`; the benchmark records its configured `semantic_timeout_seconds` in settings.

Each input is limited to 16,384 characters and 2,048 tokens.
Formula tree depth and recursive grouping depth are limited to 64.
These checks also apply to baseline output, even if its executor reports success.
Oversized values are replaced by invalid-input markers before worker serialization, and the worker validates generated input before reference input.
Native stdout/stderr use temporary files under a 128 KiB per-file limit, and raw diagnostics are not persisted.
The worker receives a minimal environment without provider credentials.
Limits can reject otherwise meaningful formulas; this is a bounded smoke-test evaluator, not an unrestricted formula service.
Hard formulas can still exhaust native resources or exceed the deadline; such failures are not evidence of inequivalence.
The worker is a timeout and data-processing boundary, not a sandbox for arbitrary executable code.
`LTLFILT` must identify a trusted installed checker.

Validation precedence is deterministic:

1. A failed translation leaves semantic evaluation `not_run`; no checker is started.
2. Validate the generated formula's supported grammar and limits.
   If both inputs are invalid, report `invalid_generated_formula`.
3. Validate the reference grammar and limits.
   A valid generated formula with an invalid reference reports `invalid_reference_formula`.
4. Initialize Spot and parse/check the normalized formulas.
   Missing or failed initialization reports `checker_error` when the inputs are valid.
   A failure to parse a converter-produced canonical formula also indicates a checker/converter error.

Infrastructure failures or the overall deadline can interrupt any of these steps and produce `checker_error` or `checker_timeout`.
Input validation takes precedence over checker availability when the worker can run.

## Results and CLI behavior

The result envelope is now `schema_version: 2`.
All previous result fields, implementation IDs, translation/provider failure categories, retry settings, raw generated code, and raw formula remain available.
Version 1 result files are not rewritten.
The existing `semantic_evaluation` field now contains real evaluation data for successful translations:

| Outcome | Meaning |
| --- | --- |
| `equivalent` | Same language over all infinite traces |
| `semantic_mismatch` | Valid formulas with different infinite-trace languages |
| `invalid_generated_formula` | Unsupported, malformed, or over-limit generated formula |
| `invalid_reference_formula` | Generated input valid, reference unsupported, malformed, or over-limit |
| `checker_error` | Missing/failed checker, worker failure, unexpected native output, or infrastructure failure |
| `checker_timeout` | The whole evaluation exceeded its deadline |
| `not_run` | Translation or configuration failed, so no formula pair can appropriately be checked |

Evaluation records include checker identity, detected version when available, the deadline, elapsed seconds, normalized strings and propositions when conversion succeeds, and a fixed diagnostic on failure.
The enclosing case's `elapsed_seconds` covers translation and checking; semantic elapsed time is also available separately.
Credential redaction applies to these added fields as well as the preexisting fields.
`outcome` and `failure` at the translation level are never replaced by a semantic mismatch or checker failure.
The CLI retains its existing exit behavior: 0 when every translation succeeds, 1 for a translation/provider/configuration failure, regardless of semantic outcomes.
Inspect `semantic_evaluation.outcome` to assess semantic success; exit status alone is insufficient.

## Independent verification and integration tests

The tests use the lecture properties supplied with this task, drawn from [Caltech's LTL lecture](https://www.cds.caltech.edu/~murray/courses/afrl-sp12/L3_ltl-24Apr12.pdf), as well as independent LTL identities:

- Slide 5: `F p` is equivalent to `true U p`, and `G p` is equivalent to `!F !p`.
- Slide 12: `F (p & q)` differs from `F p & F q`, and `G (p | q)` differs from `G p | G q`.
- Slide 5: progress `G F p` differs from stability `F G p`.
- Independently: `F p` is equivalent to `!G !p`, and differs from `X F p` because of current-step satisfaction.

Additional tests cover implication, strong-until unfolding, next/conjunction distribution, nested operators, all native operator tokens, shared atom mapping, undeclared atoms, malformed and oversized inputs, conversion and checker deadlines, and missing/crashed/failed checkers.
Both equivalent and non-equivalent pairs are required, and every infix equivalence pair is syntactically different, so constant answers and string comparison cannot pass.

```bash
python -m pytest -q
# With Docker Desktop running and the isolated baseline image already built:
LTL_DOCKER_TESTS=1 python -m pytest -q tests/test_semantic_benchmarks.py tests/test_ltlcodegen_isolation.py
```

The fixed-response integration tests run the same four existing comparison cases through each implementation and the same evaluator.
They also exercise intentional mismatch and invalid-reference cases, while baseline tests include an executor-success response containing invalid formula data.
No live provider credentials are needed.
The comparison examples remain illustrative and are not a validated accuracy dataset.
No aggregate accuracy measurement is made.
