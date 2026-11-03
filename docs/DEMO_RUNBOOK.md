# Translation demo runbook

## Goal and scope

Show how two English-to-LTL translators handle the same instructions and how a shared semantic checker compares their outputs with reference formulas.
The translators use different prompts and execution methods, so this compares complete implementations, not only their execution methods.
The optional adapted baseline wraps plain Python and supplies the missing `ltl_next` import for the executor; it is a separately identified variant.
The nine cases are illustrative examples, not a validated accuracy dataset.
The migrated LTLCodeGen baseline is a subset of a historical import, not a verified complete or unchanged upstream release.

For the recorded GPT-6 Luna results and a presentation-ready walkthrough, use the [demo script](DEMO_SCRIPT.md).
Allow roughly five to eight minutes, depending on discussion.
The translation demo is separate from the capture-the-flag agent video.

## Before recording

1. Use the demo branch and confirm its revision with `git status --short --branch` and `git log -1 --oneline`.
2. Install the benchmark dependencies, Spot, and Docker as described in the [benchmark guide](../benchmarks/README.md) and [semantic guide](../benchmarks/SEMANTICS.md).
3. Configure `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` in the ignored root `.env` file.
   Use the same model and configuration for both runs.
   Keep the key off screen.
4. Start Docker Desktop, verify the Docker server is reachable, and build the baseline image.
5. Run the tests below, then run both translators on the shared cases and inspect the saved results.
6. Prepare a table and a backup recording or screenshots from the actual results.
   Keep failures visible and distinguish them from semantic mismatches.

From the repository root, with an existing `venv` containing the benchmark dependencies:

```bash
venv/bin/python -m benchmarks.run --cases benchmarks/comparison_cases.json --list
ltlfilt --version
docker info --format '{{.ServerVersion}}'
venv/bin/python -m benchmarks.ltlcodegen_runner.runner
venv/bin/python -m pytest -q tests/test_semantics.py tests/test_semantic_benchmarks.py
LTL_DOCKER_TESTS=1 venv/bin/python -m pytest -q tests/test_semantic_benchmarks.py tests/test_ltlcodegen_isolation.py
```

The image build may download dependencies.
If using a different Python environment, replace `venv/bin/python` with its Python executable.
A successful Docker test requires Docker to be running and reachable.

## Collect results

Use distinct paths so both outputs remain easy to find:

```bash
venv/bin/python -m benchmarks.run --cases benchmarks/comparison_cases.json --implementation ltlcodegen-adapted --output benchmarks/results/demo-adapted.json
venv/bin/python -m benchmarks.run --cases benchmarks/comparison_cases.json --implementation restricted --output benchmarks/results/demo-team-new.json
```

Each run makes one live request per case, for 18 requests in total.
These paths are ignored by Git and are overwritten if reused, so rename or copy results before rerunning.
The commands print the paths, not the results.
A nonzero exit means at least one translation failed, but a zero exit does not establish semantic success.
Inspect `outcome`, `formula`, `failure`, and `semantic_evaluation.outcome` for every result in both JSON files.
Also verify matching model IDs, settings, and cases.
Do not use `--offline` for the comparison: it uses a canned response.

For a quick, read-only summary of the two files:

```bash
venv/bin/python - <<'PY'
import json
from pathlib import Path

for label, path in (
    ('Adapted LTLCodeGen baseline', 'benchmarks/results/demo-adapted.json'),
    ('Team translator', 'benchmarks/results/demo-team-new.json'),
):
    payload = json.loads(Path(path).read_text())
    print(f'\n{label}')
    for result in payload['results']:
        print(f"{result['case']['id']}: {result['outcome']} / {result['semantic_evaluation']['outcome']}")
        print(f"  formula: {result['formula']}")
        if result['failure']:
            print(f"  failure: {result['failure']}")
PY
```

Use the actual outputs to fill the [comparison table](DEMO_COMPARISON.md), or copy them to presentation slides.
Do not replace a failed run with a successful one without identifying the rerun and its conditions.

## Suggested narration

### 1. Introduce the question

> We are exploring how to turn a natural-language instruction into a temporal logic formula whose meaning can be checked.
> Today we compare our translator with a migrated LTLCodeGen baseline on the same examples and model.
> This is a separate demonstration from the agent video.

### 2. Show the data

> Each example has an instruction and a reference formula that represents the intended meaning.
> For instance, “reach object 4 at the next step” has the reference `X reach(object_4)`.
> The `X` operator means the immediately following step.

Show the instruction, reference, and description from [comparison_cases.json](../benchmarks/comparison_cases.json).
Choose one of the actual results for a detailed walkthrough.

### 3. Compare the translators

> Both implementations ask the same configured model to generate Python-shaped code that constructs a formula.
> The migrated baseline executes the generated code in an isolated container.
> Our implementation interprets an allowed subset of that code without executing it.
> The prompts differ, so this is a comparison of the full translator approaches.

The original baseline does not import `ltl_next`, so a direct call fails; the adapted variant supplies the import.
Show each actual generated formula and translation outcome.
Explain any failed translation instead of presenting it as an incorrect formula.

### 4. Explain semantic checking

> We then compare each generated formula with the reference using the same Spot evaluator.
> It checks whether the formulas describe the same behavior over infinite traces, even when their text differs.
> The check tells us whether the output matches the reference, so the reference itself still needs to accurately represent the instruction.

Show the actual `semantic_evaluation.outcome` for each translator.
An infrastructure error or timeout is not a semantic mismatch.
A failed translation is marked `not_run` because there is no formula to compare.

### 5. Show richer temporal requirements

> Strong until requires the target to eventually occur, while the condition holds beforehand.
> Weak until also permits the condition to hold forever without the target occurring.
> Release and strong release let us express closely related requirements with different conditions on whether and when the target must occur.

Use the case descriptions in the comparison table to explain timing, including whether both events may occur at the same step.
The current translators and checker do not directly accept release, weak-until, or strong-release operator symbols.
Those examples use equivalent expressions built from supported operators.
Show the actual results without claiming success before checking them.

### 6. Close

> These examples demonstrate the translation and checking workflow.
> They do not establish general accuracy: that would require a larger reviewed dataset and repeated evaluation.
> The translator demo and the agent pipeline are currently separate.

## Questions to anticipate

- **Why can formulas that look different be equivalent?** Spot checks their behavior over infinite traces rather than comparing their text.
- **Does equivalence prove the English was understood?** It proves equivalence to the reference formula, not that the reference is correct.
- **Are the advanced operators native?** No.
  Their meanings are represented with equivalent supported operators.
- **Are the runs a controlled accuracy study?** No.
  The examples are few, and the implementations use different prompts.
- **Does a successful translation guarantee a semantic match?** No.
  Inspect the translation and semantic outcomes separately.
