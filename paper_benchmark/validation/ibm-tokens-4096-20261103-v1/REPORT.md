# New exploratory translator comparison

These are new calls, separate from the preserved September results.
No historical rows were pooled into this run.

All methods use the same inputs, requested model, CTF glossary, and frozen LTLf scorer.
Native workflow call caps differ: direct JSON 3; CTF codegen 4; IBM 1.
IBM therefore has no validation repair loop; zero repairs is not evidence of better quality.
IBM uses temperature 0.5 and native stop settings; its explicit token cap and variant ID are recorded in config.json. Native cap is 200.
This compares adapted workflows, not isolated prompts at equal call/token budgets.
Gold interpretations require independent approval. Syntax and satisfiability do not prove English meaning.

| Method | Measured/scheduled | Formula-reference outputs | Correct/reference |
| --- | --- | --- | --- |
| ibm | 2/2 | 2 | 2/2 |

Read raw.jsonl, counterexamples, model metadata, and the human review sheet before interpretation.
Subset/smoke runs cannot establish translator superiority. No automatic quality ranking is made.
