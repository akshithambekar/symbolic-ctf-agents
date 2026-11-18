# Pinned GPT-4o Pan-domain pilot

This is a fresh **exploratory controlled reconstruction**, not an exact replication
of Table II. It replaces neither historical CTF benchmarks nor the earlier generic
prompt pilot. IBM is excluded. The two arms are:

* **Custom/direct JSON, domain-adapted**: the project's declarative JSON-output,
  grounded vocabulary, validation-feedback design, adapted to Pan vocabulary and
  standard infinite-trace LTL. This is not the unchanged CTF prompt or finite-trace
  game validator. The frozen system prompt is the exact evaluated prompt.
* **Paper direct-LTL baseline reconstruction**: public LTLCodeGen
  `scenegraph_prompt.get_translate_prompt` operator/task contract, flat listed-prefix
  output and `reach(object_i)` predicates, replacing its generic robotics examples
  with the shared Pan few-shot examples. This is the public direct-translation path,
  **not IBM NL2LTL**, and not LTLCodeGen's Python-generation method. The paper's
  NL2LTL baseline cites Dai et al.; the unrelated IBM template library is excluded.

Upstream LTLCodeGen source commit: `b5eb6a19769c3b06c5757136bef3bd444533e328`.
Pan source commit: `7c778144c6bfcfecd14b3df5fbb6a40d2e15fd17` at
<https://github.com/UM-ARM-Lab/Efficient-Eng-2-LTL>.

## Freeze and evaluation

`freeze.py` selects 20 Drone, 12 Cleanup and 5 Pick examples from Pan's
`train_seed.jsonl`, pairing utterances with Pan's canonical reference formulas.
Both arms receive the same instruction/reference pairs, rendered in their native
output notation. Seed: 20261118. The authors' original random example identities
were not recovered; these must not be described as the exact paper sample.

The pilot selects 20/20/5 instructions per human/synthetic split: **90 inputs,
180 outputs**. Normalized exact prompt-instruction overlap is excluded. Formula
families can overlap, particularly Pick's five patterns; this is a few-shot
instruction-generalization pilot, not unseen-formula generalization. Independent
reference review remains pending. Corpus annotation errors can affect scores.

Both arms use `openai/gpt-4o-2024-11-20`, OpenAI provider without fallback,
temperature 0, 4,000 output tokens, initial request plus at most three repairs.
The publication's exact GPT-4o snapshot is not established. Repair uses validation
errors only, with common conversational feedback rather than claiming byte-exact
upstream repair prompts. Gold formulas never enter generation or repair requests.

The bounded Spot 2.16.3 checker in WSL compares **infinite-trace logical
equivalence**, separately from syntax and allowed predicates. Native Cleanup `X`
is a proposition, not next. Mapping is frozen. Prefix `EQUAL` is normalized to
bidirectional implication. Refusals and invalid formulas fail task accuracy;
checker/provider failures are explicit and stop the pilot, not semantic errors.

No generated Python is executed. `ast.literal_eval` parses lists of strings only.
The Python-code-generation arm is outside this two-method comparison. If later
added, generated code must run in a separately isolated execution environment.

## Cost and artifacts

The runner refuses directories that already exist. Each request reserves a
conservative input-byte/output-token cost before transmission; missing usage
retains the full reservation. The pilot ceiling is **$5**; there is no automatic
paid fallback or unbounded retry. Model pricing is checked before starting.
Reported provider usage/cost and conservative accounting are distinct records.

Run files contain frozen examples/prompts/specification, source hashes, exact
model/provider config, source snapshot, all raw requests/responses, retries,
latency, semantic checks, budget ledger, summary and status. API keys are never
written. Raw downloaded datasets stay in ignored run directories pending license
review. Historical experiments remain unchanged.

```powershell
.\.venv-paper\Scripts\python.exe paper_benchmark/pinned_study/run.py `
  --frozen paper_benchmark/runs/gpt4o-frozen-1118-v1 `
  --output paper_benchmark/runs/gpt4o-pilot-1118-v1 `
  --env-file ../maritime-nsrl/.env --budget-usd 5
```

Interpretation is limited to this model, shared examples and pilot samples.
Do not claim full-corpus Table II reproduction, IBM comparison, code-generation
performance, or general translator superiority from this pilot.
