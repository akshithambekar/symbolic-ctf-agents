# Recording script: English-to-LTL demo

Read only the words in quotation marks.
Each **DO** line tells you exactly what to show next.
It's fine to pause or cut the recording between sections.

## Before you record

Open these three files in your editor:

1. `benchmarks/comparison_cases.json`
2. `benchmarks/results/live-baseline.json`
3. `benchmarks/results/live-team.json`

In each file, search for `release-avoid-until-target` and leave that result visible.
The existing result files show a baseline mismatch and a team match for this case.
If you run the benchmark again, the output can change, so check the new files before using the specific explanation in section 5.

Open a terminal at the repository root.
Make sure Docker Desktop is running and the OpenRouter key is configured.
Keep your `.env` file and API key off screen.

## 1. Introduce the work

**DO:** Show `comparison_cases.json`.

> “We built two ways to translate English instructions into temporal logic.
> We connected both to GPT-6 Luna through OpenRouter, and we use Spot to check whether their output matches a reference formula.”

## 2. Explain the example

**DO:** Point to the `instruction` in `release-avoid-until-target`.

> “Let's look at this example.
> It says we can only reach object 3 if we've already reached object 7 at an earlier step.
> We don't have to reach object 7, but if we never do, we must avoid object 3 forever.”

**DO:** Point to `reference_formula`.

> “This is the formula for that intended meaning.
> We use it as the reference for checking each translator.”

## 3. Run the translators

**DO:** Switch to the terminal.
Paste the complete block below once and press Enter.

> “I'll run all nine cases through both translators using the same model.
> Each translator saves its results in a separate file.”

```bash
venv/bin/python -m benchmarks.run \
  --model openai/gpt-6-luna \
  --cases benchmarks/comparison_cases.json \
  --implementation ltlcodegen-adapted \
  --output benchmarks/results/live-baseline.json

venv/bin/python -m benchmarks.run \
  --model openai/gpt-6-luna \
  --cases benchmarks/comparison_cases.json \
  --implementation restricted \
  --output benchmarks/results/live-team.json
```

**DO:** Wait until both result paths have been printed and the terminal prompt returns.
Pause the recording while the commands run if you like.
The second command runs even if the first reports an error.
The commands overwrite the two result files and make up to 18 model requests in total.
If a run fails or an outcome differs from the example below, use the alternate wording at the end of this script.

## 4. Show the result fields

**DO:** Open `live-baseline.json` and find `release-avoid-until-target`.
Point to `formula`.

> “This is the formula the baseline produced.”

**DO:** Point to `outcome`, then `semantic_evaluation.outcome`.

> “The translation outcome tells us whether it produced a formula.
> The semantic outcome tells us whether Spot found that formula equivalent to our reference.”

## 5. Compare the release results

**Before saying this section, verify that the baseline says `semantic_mismatch` and the team says `equivalent`, and that the formulas match the supplied example.**
The model may produce a different answer on a new run.

**DO:** In `live-baseline.json`, point to `semantic_evaluation.outcome`.

> “Here, the baseline produced a valid formula, but Spot found a mismatch.
> Its formula allows the two objects to be reached at the same step.
> The instruction requires object 7 to have been reached earlier.”

**DO:** Open `live-team.json` at the same case.
Point to `semantic_evaluation.outcome`.

> “Our team translator's formula rules out that simultaneous case.
> Spot found it equivalent to the reference.”

## 6. Close

**DO:** Leave the team result visible.

> “This is why the semantic check is useful.
> A translator can produce a valid formula that still allows behavior the instruction rules out.
> These nine cases are examples for the demo, not a general accuracy study.”

## If the new results differ

Read the outcomes that are actually on screen.
If both say `equivalent`, say: “For this run, Spot found both formulas equivalent to the reference.”
If a result says `semantic_mismatch`, explain its actual formula before saying why it differs.
If it says `not_run`, show `failure` and say: “This translation did not produce a formula, so the semantic check did not run.”
If a live run is slow or fails, show earlier results and identify them as earlier results.

The baseline is an adapted version of the migrated LTLCodeGen subset: the adapter accepts plain or fenced code and supplies a missing next-operator import.
The implementations use different prompts, and release is expressed through supported operators rather than a native release operator.
