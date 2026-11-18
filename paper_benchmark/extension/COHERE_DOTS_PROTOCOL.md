# Cohere/Dots CTF model extension

Run: `runs/cd-1118-v1`. Existing experiment results remain unchanged.

## Frozen comparison

- `cohere/north-mini-code:free`: 30B total / 3B active parameters.
- `dots-studio/dots-3-note-preview:free`: 280B total / 16B active parameters.
- 120 existing CTF instructions, three trials, three adapted translator workflows:
  custom JSON, CTF LTLCodeGen code adapter, and IBM NL2LTL.
- 2,160 scheduled main outcomes, plus 12 smoke outcomes excluded from main scores.
- Exact dataset bytes/hash, catalog, model IDs, source commit, source snapshots,
  native settings and prompts are preserved. No prompt tuning or reference changes.
- Native request caps remain custom 3, codegen 4, IBM 1; IBM output cap remains
  4,096 tokens. These are workflow comparisons, not equal-token-budget comparisons.
- Both models must be advertised at zero input/output price before launch.
  No paid substitutions or random free-model router is used.

This extends the old **CTF** experiment; it does not add IBM to the separate
two-arm GPT-4o Pan-domain pilot. Restricted AST interpretation of generated code
is retained; arbitrary model-generated Python is never executed.

## Meeting-note requirements

1. **Explain failures:** automated categories distinguish refusal, output-token
   truncation, code interpretation, unsupported predicates, compilation, and
   stronger/weaker/different reference meanings. A review CSV retains exact source
   rows and leaves causal conclusions for review. Categories are observed symptoms,
   not proof that a specific prompt or template caused failure.
2. **Track tokens/cost:** `usage_attempts.csv` records all actual requests, including
   repairs and infrastructure failures; `usage_summary.csv` aggregates by model and
   translator. Input, completion, reasoning and total tokens are separate fields.
   Reasoning is a subset of completion, not added twice. Missing usage/cost is
   unknown; reported zero is distinguished from missing. Smoke is marked and
   excluded from main usage aggregates. Every completed condition produces a fresh
   audit directory; raw provider usage remains intact.
3. **Broaden model coverage:** add two working free alternatives with explicitly
   different size classes. This is not a within-family scaling experiment and does
   not isolate parameter count from architecture/training/provider differences.
4. **Keep upstream tests separate:** Pan-domain evaluation and recovered upstream
   robotics instructions are independent studies, not pooled into this dataset.

## Integrity and interpretation

Provider errors stop that model's remaining conditions rather than creating an
unbounded retry loop or counting unserved requests as incorrect answers. Failed
raw calls remain available for a later missing-outcome continuation. Incorrect
but measured outcomes are retained, never retried for a better score.

Primary semantic scoring remains unconstrained Boolean **LTLf** reference
agreement with the frozen scorer. Supported-predicate validity is not proof of
natural-language correctness. Independent reference approval remains pending;
game-grounding relationships can change equivalence judgments and must be reported
as separate sensitivity analyses. Do not claim general superiority from smoke
tests, incomplete runs or this dataset alone.

```powershell
.\.venv-paper\Scripts\python.exe paper_benchmark/extension/alternative_study.py `
  --output paper_benchmark/runs/FRESH_DIRECTORY `
  --env-file ../maritime-nsrl/.env
```

Every target directory must be new. Check `status.json` for main measured count;
smoke outputs deliberately do not increase that count. Per-trial quality tables,
plots, counterexamples and human-review sheets are emitted by the existing runner.
