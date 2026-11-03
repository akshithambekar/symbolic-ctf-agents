# Demo comparison: English-to-LTL translators

This comparison describes the implementations in this repository.
The LTLCodeGen baseline is a migrated subset, not a verified complete or unchanged upstream release.
The team translator is a local derivative of that migration.
See the [provenance audit](../translators/README.md) for details.

| Aspect | Migrated LTLCodeGen baseline | Team implementation |
| --- | --- | --- |
| Benchmark ID | `ltlcodegen-container-v1` | `restricted-ast-v1` |
| Prompt | Imported few-shot navigation prompt | Separate copy with restrictions for the AST interpreter |
| Model response | Python code written to a generated module and executed | Python-shaped text interpreted through a restricted AST; generated code is not executed |
| Execution boundary | Benchmark runs the executor in a resource-limited Docker container | Host-side interpreter accepts only supported constructs and operators |
| Response formatting | Imported executor removes the first and last response lines | Accepts plain code or a single Markdown Python fence |
| Retries in benchmark | Disabled | Disabled |
| Output and evaluation | Listed-prefix formula checked by the shared Spot evaluator | Listed-prefix formula checked by the same evaluator |
| Dependencies | Imported legacy LangChain dependencies and Docker for the benchmark | `langchain-core` and benchmark dependencies |

Both benchmark paths use the same cases, provider settings, and semantic evaluator when run with the same configuration.
Translation success and semantic equivalence are recorded separately.
The baseline's normal direct executor supports retries and executes generated Python with normal process permissions; the benchmark disables retries and uses the container.

## Demo cases

| Instruction | Reference formula | LTLCodeGen result | Team result |
| --- | --- | --- | --- |
| Eventually reach object_7 | `F reach(object_7)` | Not measured here | Not measured here |
| Never reach object_3 | `G !reach(object_3)` | Not measured here | Not measured here |
| Eventually reach object_1 and eventually reach object_2 | `F reach(object_1) & F reach(object_2)` | Not measured here | Not measured here |
| Eventually reach object_1 or object_2, or both | `F (reach(object_1) \| reach(object_2))` | Not measured here | Not measured here |
| Reach object_4 at the next step | `X reach(object_4)` | A direct `ltl_next` call fails in the baseline template | Not measured here |
| Do not reach object_3 until object_7 is reached, and ensure object_7 is eventually reached | `!reach(object_3) U reach(object_7)` | Not measured here | Not measured here |
| Release: avoid object_3 unless object_7 was reached earlier | `!(!reach(object_7) U reach(object_3))` | Not measured here | Not measured here |
| Weak until: avoid object_3 until object_7, or forever | `(!reach(object_3) U reach(object_7)) \| G !reach(object_3)` | Not measured here | Not measured here |
| Strong release: avoid object_3 through an eventual reach of object_7 | `!reach(object_3) U (reach(object_7) & !reach(object_3))` | Not measured here | Not measured here |

These nine examples are for inspection, not a validated accuracy dataset.
`F` includes the current step, and the two-object conjunction does not impose an order.
`X` refers to the immediately following step.
The baseline defines `ltl_next` but its execution template does not import it, so a direct call fails; live results remain unmeasured.
Strong `U` requires object_7 eventually and allows both propositions to hold when object_7 is reached.
Release, weak until, and strong release use equivalent expressions built from supported operators; neither translator nor the checker accepts those operator symbols directly.

## Run the comparison

Set up the provider, Spot, and the isolated baseline as described in the [benchmark guide](../benchmarks/README.md).
With Docker running and the same model configured for both runs, execute:

```bash
python -m benchmarks.run --cases benchmarks/comparison_cases.json --implementation ltlcodegen-container
python -m benchmarks.run --cases benchmarks/comparison_cases.json --implementation restricted
```

Each command prints the path to an ignored JSON result file.
Compare each case's translation outcome, generated formula, `semantic_evaluation.outcome`, and elapsed time in those files.
The offline mode uses a fixed response for every case and must not be used to assess translation quality.
