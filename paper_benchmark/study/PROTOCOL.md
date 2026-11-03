# Proposed publication experiment protocol

Status: prepared before new study calls; references/profile await independent
review. This document is not a claim of preregistration in an external registry.

## Questions and implementations

Primary: how reliably do the three pipelines preserve the intended CTF mission
under a common proposition profile and finite-trace interpretation?

- Frozen September direct-JSON translator.
- Frozen September CTF-adapted LTLCodeGen translator.
- IBM NL2LTL 0.0.6 CTF/OpenRouter adaptation, explicitly identified by token cap.

Prompt improvement is a second experiment: frozen direct-JSON baseline versus
documented variants developed only on the development portion. Do not tune a
variant after inspecting test results and still call that result held out.

## Data and annotation

Candidate version: `mctf-translation-candidate-2.0`, 120 items, 60 families.
Development: original 40 items, already exposed through previous measurements.
Test candidates: 80 new instructions, 40 families, two phrasings per family.
Keep paired meanings together. The test is compositional within the same eight
propositions; do not claim unseen-domain or arbitrary-natural-language coverage.

1. Two reviewers first judge English plus glossary/profile without seeing the
   proposed formulas. Each records formula/ambiguous/unsupported, their own LTLf
   interpretation where applicable, and rationale.
2. They separately approve or reject the aggregate predicate meanings and proposed
   invariant `teammate_safe <-> !tagged` across every step. Its encoding uses only
   the frozen parser's operators. Resolving profile objections precedes scoring.
3. Compare independent judgments and inspect proposed references and abstract
   traces. An adjudicator resolves disagreements with a recorded rationale.
4. Publish actual reviewer process, agreement/disagreement counts, adjudications,
   changes and exclusions. Automation checks declarations, not reviewer identity.
5. Create a new reviewed release with dataset hash and fixed splits. Do not edit
   this candidate version or the historical reference run to conceal corrections.

Source-backed ideas are newly worded/adapted, not copied authoritative gold
labels. Dataset construction used AI assistance. No claim is made that public
source ideas are absent from the language model's pretraining data.

## Semantics and annotation traps

All episodes are nonempty finite traces; F includes the current step, X is strong,
and U is strong. A deadline in seconds, a distance, distinct-boat identity,
Powerplay actions, or a new region cannot be inferred from an unrelated atom.
The eight propositions are team aggregates; `home` means flag-stand proximity,
not official corner capture zones or necessarily the current carrier.

Primary proposed semantic metric: agreement under the reviewed grounding
invariant, using the existing product-DFA checker on `(invariant & output)` versus
`(invariant & reference)`. Report the original unconstrained Boolean result too,
as a separate metric. Resource limits remain inconclusive errors, never proofs
of equivalence. Reviewers must approve the profile before this choice is frozen.
SAT witnesses and counterexamples are abstract Boolean traces, not guaranteed
simulator trajectories or proof that the English interpretation is correct.

## Model, prompts and budgets

Use the paper's exact model ID if still served:
`nvidia/nemotron-3-ultra-550b-a55b:free`. Record requested/resolved model IDs,
provider metadata, date, all prompts and outputs, usage, errors and latencies.
If unavailable, define a new model condition for all methods; do not silently
substitute another model or pool it with the historical results.

Propose three repeated trials per test instruction/method: 720 outcomes for
80 test instructions and three methods. Repeats measure generation variability,
not 720 independent instructions. Development calls are recorded separately.

Primary workflow comparison preserves the frozen two methods' native repair
caps (3 and 4 calls). IBM uses one call and an explicitly labeled 4096-token
variant because its native 200-token smoke exhausted the budget on reasoning.
Native sampling/token settings differ: this is an adapted-workflow comparison,
not an isolated prompt or equal-resource causal comparison. Record all budgets.

A separately labeled equal-resource ablation would require new wrappers for all
methods, the same maximum calls, token budget, sampling and timeout, and a
documented repair protocol. Do not change the frozen methods to achieve it.
No automatic paid fallback is permitted. Interleave methods across cases/trials;
record rate limits and temporal service changes. Retain missing/infrastructure
outcomes and report coverage rather than rerunning selectively until successful.

## Metrics and statistics

- Primary: semantic/reference agreement on formula-expected test items, with
  invalid or missing model output counted as incorrect. Report counts and rates.
- Appropriate clarification/rejection on the separately counted non-formula items.
- Syntax, grounded predicate support, compilation, strong/inclusive ordering.
- Added constraints, omitted conditions, reversed implication/order, unexpected
  predicates, and refusal reasons; human-audited error categories.
- Repair frequency, total calls, prompt/completion/reasoning tokens, latency, and
  cost where provider data is available. A one-call method's zero repairs is not
  inherently better. Track infrastructure coverage separately from quality.
- Paraphrase consistency with failures/missing outputs and denominator stated.

Report per-category/per-family outcomes and paired method differences. Estimate
uncertainty by resampling semantic families, keeping both phrasings, methods and
all trials together in each resample. A pre-specified family-cluster bootstrap
(10,000 replicates, fixed seed 491) can produce 95% intervals. Do not treat trials
or paraphrases as independent samples. Identify small subgroup uncertainty.
Any multiple-comparison procedure and confirmatory hypotheses must be fixed
before evaluating the reviewed test release. No best-method claim from a smoke run.

## Release and reproducibility

Freeze exact dataset/profile/split/model/prompt/adapter/scorer versions before
new test calls. Save raw outputs, hashes, snapshots, dependency locks, traces,
review records and all exclusions in new directories. Keep the September pilot
separate from the new study, and exploratory prompt selection separate from final
test evaluation. The reviewed dataset export and evaluation/statistics integration
are subsequent implementation steps; this preparation does not launch them.
