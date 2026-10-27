# Team translator

This is our locally modified derivative of the migrated LTLCodeGen core, not the original baseline.
Its stable benchmark identity is `restricted-ast-v1`, exported as `IMPLEMENTATION_ID` from this package.
The benchmark imports this package directly; no implementation-selection framework is needed for the single-case MVP.
See the [history audit](../README.md) for migration provenance and the separate baseline.
The inherited prompt and operators retain the [migrated MIT license](LICENSE-LTLCodeGen).

`translator.py` preserves the recent local changes: one model invocation, fenced or unfenced text handling, restricted AST interpretation, bounded code/formula size, and rejection of unsafe or unsupported structures.
It never executes, imports, or writes model-generated Python.
It accepts one argument-free `question()` function, an optional docstring, local formula assignments, approved LTL calls, and a final return.
The stricter response instructions are in our own `prompt_template.py`.
The model is supplied by the caller through `invoke(prompt)`; OpenRouter configuration and result recording belong to the benchmark.

Supported calls are `ap(reach, 'object_identifier')`, `ltl_and`, `ltl_or`, `ltl_not`, `ltl_next`, `ltl_until`, `ltl_eventually`, `ltl_always`, and `ltl_imply`.
Imports, attributes, indexing, decorators, annotations, loops, conditionals, and arbitrary Python are rejected.
Translation returns `(formula_or_failure_reason, success, generated_code)`.
The retained `retry_on_error` and `max_retries` arguments reject requests to enable retries.
Successful interpretation establishes that a formula was constructed, not that it matches the instruction.

Our separate `ltl_operators.py` defaults to `prefix=True` and `listed=True`.
The translator's default `listed=True` adds outer brackets, producing `['EVENTUALLY', 'reach(object_7)']` for the smoke case.
These operator settings are mutable process globals within this implementation; do not change them concurrently.
They are independent of the baseline's globals.
No prompt or operator module is shared with the original executor.

Install the [benchmark dependencies](../../benchmarks/requirements.txt), run the [offline tests and smoke command](../../benchmarks/README.md), and inspect the ignored JSON output in an editor.
Semantic evaluation explicitly remains `not_run` pending one shared checker and implementation-specific notation conversion.
