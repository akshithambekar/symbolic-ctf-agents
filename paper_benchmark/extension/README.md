# Translator research extension — November 17, 2026

This extension keeps the four research questions separate. It does not change
the preserved CTF experiments or claim that parser validity proves English meaning.

## 1. Failure analysis

`audit.py` reads the completed matrix roots recorded by the Ultra supervisor.
The first measured output stays fixed, including incorrect/unusable outputs.
It separates explicit refusal, output truncation, code interpretation failure,
unsupported predicates, and strict strengthening/weakening of the reference.
These are observable symptoms; the human review sheet has blank causal labels.

Ultra/IBM's 360 outcomes include 172 refusals on formula-expected items and 30
no-formula outcomes stopped at the output cap. Its one-template restriction and
reasoning budget are therefore distinct issues. LTLCodeGen's 360 outcomes include
52 strict strengthenings, 9 weakenings, 21 other semantic mismatches, 51 invented
formulas for ambiguous/unsupported requests, 20 code-interpretation failures,
and 1 output-limit failure. The remaining 206 agree with formula references.
Do not collapse these categories into an undifferentiated syntax-error claim.

## 2. Token and cost accounting

`usage_attempts.csv` includes every actual saved request from the source roots,
including validation repairs and infrastructure-failed recovery attempts.
Consolidated copies are excluded. Main summaries exclude smoke calls, which remain
available in the ledger. Reported cost is distinct from estimated cost. Missing
cost is unknown, not a zero-price observation. Reasoning tokens are normally a
subset of completion tokens, not an additional amount to add to total tokens.

Current zero-price NVIDIA routes report zero cost on successful metered calls.
Token totals still quantify their resource demands. Source file hashes and the
number of attempts with usage/cost accompany every aggregate.

```powershell
.\.venv-paper\Scripts\python.exe paper_benchmark/extension/audit.py --output paper_benchmark/runs/new-audit
```

## 3. Free model coverage

`free_models.py` snapshots the live catalog and serving endpoints. It requires
zero prompt/completion prices and a text-generation route. A historical model
webpage alone does not establish live availability.

The confirmed new family is Google Gemma:

| Model ID | Total parameters | Active per token |
|---|---:|---:|
| `google/gemma-4-26b-a4b-it:free` | 25.2B | 3.8B |
| `google/gemma-4-31b-it:free` | 30.7B | 30.7B, dense |

NVIDIA Lightning/Nano Omni are approximately 30B total / 3B active; Super VL is
120B / 12B; Ultra is 550B / 55B. Gemma supplies a small/large active-size contrast,
not an exact match in total parameters, architecture, or inference compute.
These sizes come from the archived OpenRouter descriptions and model endpoints.

On November 17 the live catalog lists no second non-NVIDIA family with two free
text models. The checked OpenAI GPT-OSS 20B/120B and Qwen 30B/235B free aliases
have zero serving endpoints despite historical webpages. The second pair remains
pending rather than silently substituting paid models, a random router, or two
unrelated families. Re-run discovery when availability changes.

`gemma_study.py` runs six smoke outcomes per model, then the same 120-item,
three-trial, three-translator exploratory protocol (2,160 full-study outcomes).
Prompts, native generation settings, repair caps and the existing Boolean LTLf
scorer stay fixed. It does not tune prompts using new results. Infrastructure
failures leave a visibly incomplete study for recovery rather than quality zeros.

## 4. Public upstream evaluation

The LTLCodeGen project page's Data link points to
`real_world_evaluation_prompts.yaml`, absent from current main. We recovered the
exact source from commit `e152ebd0d0b328f945e07b6c050106ce89bdb794`:

https://raw.githubusercontent.com/ExistentialRobotics/LTLCodeGen/e152ebd0d0b328f945e07b6c050106ce89bdb794/real_world_evaluation_prompts.yaml

It contains 45 instructions: 10 Tier 1, 10 Tier 2, 15 Tier 3 and 10 Tier 4.
It supplies **no gold LTL formulas**. The file and upstream MIT license are
preserved under `upstream_data/`. Instructions remain unchanged.

`upstream.py` tests the original codegen prompt/operators on these instructions,
with an explicit eight-object navigation glossary. The generated function is
interpreted using the existing restricted AST interpreter rather than executed.
This is a translation-only, object-profile adaptation. It does not recreate the
robot's semantic map, manipulation capabilities, ROS/Gazebo planner, GPT-4o paper
configuration, or paper success rate. Syntax and LTLf compilation are diagnostic
outputs, and semantic accuracy remains null pending reference construction/review.
No CTF result is pooled with the upstream results.

```powershell
.\.venv-paper\Scripts\python.exe paper_benchmark/extension/upstream.py --source paper_benchmark/extension/upstream_data/real_world_evaluation_prompts.yaml --env-file ../maritime-nsrl/.env --output paper_benchmark/runs/new-upstream
```

The LTLCodeGen paper also evaluates Drone, Cleanup and Pick datasets from
UM-ARM-Lab/Efficient-Eng-2-LTL. Those are separate resources with different
predicate grammars and semantics. A scored reproduction needs their gold data,
operator normalization and semantic checker; the instruction-only 45-case test
must not be misreported as such a reproduction.

Primary sources:

- https://openrouter.ai/api/v1/models
- https://openrouter.ai/google/gemma-4-26b-a4b-it:free
- https://openrouter.ai/google/gemma-4-31b-it:free
- https://existentialrobotics.org/LTLCodeGen/
- https://github.com/ExistentialRobotics/LTLCodeGen
- https://github.com/UM-ARM-Lab/Efficient-Eng-2-LTL
