# Interpretation hold: upstream-domain pilot

The 270 saved outputs are retained as an exploratory adapter diagnostic.
Do not describe their ranking as our translator outperforming the LTLCodeGen
authors or reproducing the paper. No primary score has been rewritten.

## Verified issues

1. The pilot used NVIDIA Ultra and Super VL, rather than the paper's GPT-4o.
2. It used the preserved generic code-generation prompt plus a domain glossary
   and object-to-predicate mapping. The paper reports dataset-specific prompt
   examples (20 Drone, 12 Cleanup, 5 Pick). We did not reproduce those prompts.
3. The test is 20/20/5 deterministically sampled instructions, one per sampled
   reference-string group, not the original full evaluation or its trial design.
4. Semantic equivalence to a supplied reference is not independent validation
   of that reference's agreement with the English instruction.
5. Pick's canonical labels provide limited grounding semantics for manipulation
   events. Our navigation-derived object interface is not a validated recreation
   of the paper's dataset-specific task representation.

## Concrete reference/English discrepancy

Source row `drone-planning-1191`:

> always avoid the orange room to navigate to landmark one .

The source reference becomes `F(p3 & !p6)` after renaming, with p3 denoting
landmark one and p6 denoting the orange room. The codegen result is
`G(!p6) & F(p3)`. This preserves a plausible literal reading of "always avoid".
The reference allows entering the orange room after a successful destination
visit, whereas the generated formula prohibits it throughout the infinite trace.
Spot's mismatch is mathematically correct; English correctness needs review.

Every Drone codegen output in this pilot produced a usable formula. The 0/20
scores for both models therefore do not mean code generation or parsing failed.
They mean none of those generated trace languages equaled the supplied reference.

## Required before a paper claim

- Recover the authors' dataset-specific prompts, symbol grounding, split and
  evaluation setup. If unavailable, label any reconstruction explicitly.
- Independently review English/reference discrepancies and ambiguous phrases.
- Test the adapters on published reference examples with known constructions.
- Run matched prompt/representation ablations, keeping backend and budgets fixed.
- Report exact source references separately from any reviewed alternative labels.
- Keep this pilot separate from CTF results and published upstream performance.
