# Translator benchmark

This experiment is independent of RL. It never writes into Phase 2 or Phase 3.
The dataset contains 40 CTF statements (20 paraphrase pairs) covering safety,
eventual goals, ordering, conditionals, designated roles, combinations, difficult
temporal operators, ambiguity, and unsupported concepts. Gold interpretations are
project-authored and explicitly await independent human review.

## Run

From this project directory, using the existing virtual environment:

```powershell
mctf benchmark-translators --output results/translator-benchmark-live-01 --model gpt-4o-mini-2024-07-18 --trials 3 --logic-backend dfa
```

Set `LTL_API_KEY` or `OPENAI_API_KEY` in your own environment. Never put credentials
in configuration files. `--base-url` accepts an OpenAI-compatible endpoint.
The output directory must be new. Model calls require outbound network access.
`--logic-backend fragment` separately evaluates the original restricted validator;
do not pool that arm with the DFA-backed pipeline. The existing translation adapter
is reused without changing its prompts or repair logic; a common CTF glossary is
included with the instruction for both systems. References are withheld.

There are 240 scheduled outcomes at three trials. A failed backend connection stops
further calls to that backend, records remaining outcomes as unmeasured, and still
produces all artifacts. This avoids hundreds of repeated connection failures. An
infrastructure-blocked run is not a completed empirical translator comparison.

Outputs: `dataset.json`, `config.json`, `manifest.json`, exact source snapshot,
dependency versions, append-only `raw.jsonl`, `summary.csv/json`, `paraphrases.json`,
`human_review.csv`, `coverage.png`, and `REPORT.md`. Raw attempts include request
messages (without headers/keys), returned text, actual provider model identifier,
usage when provided, errors and timing. The pinned requested model is recorded even
if the provider cannot return a resolved identifier. Native retry limits differ:
three total calls for the current pipeline, four for LTLCodeGen.

## Scope and semantic scoring

The LTLCodeGen translation-only adapter pins upstream commit
`b5eb6a19769c3b06c5757136bef3bd444533e328`. Upstream few-shot prompt and operator
functions are bundled unchanged under the MIT license. The CTF glossary maps
`reach(object_i)` to game predicates. An appended prompt note provides that mapping,
explicit rejection handling and the strong-next operator. It replaces LangChain's
transport with requests. The generated `question()` function is interpreted through
a bounded AST subset (assignments, strings, logical operator calls, return), not
executed as arbitrary Python. Imports, loops and other constructs are recorded as
unsupported generated code. Thus it is a constrained integration of the actual
translation portion, not a claim to run the complete upstream system.

ROS/Gazebo, speech, perception, semantic maps and motion planning are excluded.
Upstream uses infinite-trace LTL/Buchi planning; this benchmark deliberately scores
both outputs under the project's finite-episode LTLf semantics. These adaptations
limit transfer of conclusions to the original published robotics system.

The scorer explores the reachable product of two DFAs, checking acceptance on every
nonempty finite Boolean trace. Equivalent formulas do not need matching strings.
Counterexamples explain strict strengthening, weakening, or incompatible meaning.
Resource exhaustion is inconclusive, never a proof of equivalence. Generated unknown
APs are rejected. Compilation establishes Boolean satisfiability, not physical
realizability. The arbitrary Boolean alphabet includes physically impossible states;
it deliberately does not assume that a score logically implies prior possession.

Correct = equivalent to reference, or explicit rejection for an ungrounded item.
Partially correct = provisional strict strengthening/weakening, subject to review.
Incorrect = incompatible meaning, tautology, or unjustified precision on ambiguity.
Invalid/unusable = no accepted formula or failed syntax/grounding/compilation.
Not measured = infrastructure prevented evaluation. Rejections of ambiguous prompts
require review of the wording. Ordering checks use full reference equivalence and
counterexample traces. Paraphrase consistency compares formulas semantically; missing
outputs are not counted as agreement. Reviewers should edit a copy of the review
sheet, retain their rationale and reference approval, and not overwrite raw outputs.

## Current decision

The local run encountered Windows socket-permission errors before either model
returned an answer. Consequently no measured accuracy/latency/repair-rate ranking
is supported. Keep the current translator operationally while collecting live data.
A hybrid (code generation followed by our validator) is a feasible separately
labelled third arm, not an already demonstrated improvement.

Sources: [project](https://existentialrobotics.org/LTLCodeGen/) and
[pinned source](https://github.com/ExistentialRobotics/LTLCodeGen/tree/b5eb6a19769c3b06c5757136bef3bd444533e328/speech_to_ltl).
