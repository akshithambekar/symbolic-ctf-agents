# Reference-scored Drone, Cleanup and Pick comparison

This is a separate exploratory pilot, not the 45-instruction real-world validity
test and not a reproduction of the LTLCodeGen paper's GPT-4o results.

## Frozen inputs

Source: UM-ARM-Lab/Efficient-Eng-2-LTL at commit
`7c778144c6bfcfecd14b3df5fbb6a40d2e15fd17`.

| Dataset | Source rows | Raw reference-string groups | Pilot instructions |
|---|---:|---:|---:|
| Drone | 6,185 | 355 | 20 |
| Cleanup | 3,382 | 39 | 20 |
| Pick | 744 | 5 | 5 |

The raw files and their checksums are under the ignored
`paper_benchmark/runs/gold-datasets-1117-v1/` directory. These are exact source
row/string counts, not a claim about the number of semantically distinct formulas.
Sampling uses seed 20261117, chooses raw formula groups without replacement,
then picks one instruction per group. All three translators receive the same
sample. The gold formulas are used only by the checker, never in model requests.

## Semantics and domain adaptation

The checker uses Spot 2.16.3 `ltlfilt` over **infinite traces** through the
repository's existing bounded `benchmarks.semantics.evaluate` implementation.
Windows invokes it through Ubuntu WSL. The CPU-independent parser bridge and
checker version are recorded with results. No sampled trace or exact-string
matching substitutes for semantic equivalence.

The domain importer parses Drone's infix and Cleanup/Pick's prefix reference
notations. In Cleanup, `X` is an atomic proposition meaning blue room with chair,
not a temporal operator. The importer renames all propositions bijectively to
`p0`, `p1`, etc. and exposes the domain glossary to every translator. It preserves
source references, source row indices and original instruction wording, including
any source annotation inconsistencies. Native source meanings come from the
canonical domain vocabulary, not the generated output.

Each gold formula must pass a Spot parsing/self-equivalence check before the
first live translation. Unit tests include equivalent formulas with different
syntax, a genuine mismatch, and an infinite/finite trace distinction using next.

## Compared variants

- Direct JSON: domain prompt with the original direct-LTL interface and up to
  three validation attempts.
- Codegen: preserved upstream operator prompt and the existing restricted AST
  interpreter, with domain object-to-predicate mapping, up to four attempts.
- IBM: real pinned NL2LTL 0.0.6 parser and constructors, with an explicit domain
  profile, one-template limitation and one call capped at 4096 tokens.

These domain-adapted variants are distinct from the frozen CTF conditions.
Default CTF prompt behavior remains unchanged. IBM's constructors produce
LTLf syntax; for this comparison the resulting operator expression is explicitly
interpreted under the same infinite-trace semantics as all other outputs. This
is an adaptation, not a claim to reproduce IBM's original experimental setup.

The pilot uses Ultra and Super VL, one trial: **45 x 3 x 2 = 270 outcomes**.
Both model IDs end in `:free` and are checked for zero advertised token pricing.
Gemma is still being collected separately on CTF when shared free capacity permits.
No paid fallback is enabled. Native repair budgets differ and remain disclosed.

## Outputs and recovery

Each run preserves dataset/configuration, source hashes and source copies,
reference validation, raw requests/responses, usage metadata, semantic outcomes
and per-domain/model/method summaries. Unknown predicates, refusal, translation
failure, semantic mismatch and infrastructure failures remain distinguishable.
Checker failures stop scoring instead of becoming incorrect translations.

`finish_gold.py` supervises the pilot and resumes only missing or infrastructure
outcomes into fresh directories. Measured incorrect/unusable outcomes are retained.
Retries wait 30 minutes after interruption. Source roots remain read-only.

```powershell
.\.venv-paper\Scripts\python.exe paper_benchmark/extension/finish_gold.py --source paper_benchmark/runs/gold-datasets-1117-v1 --env-file ../maritime-nsrl/.env --output paper_benchmark/runs/new-gold-supervisor
```

Spot binary packages came from the Ubuntu 24.04 build linked by Spot's official
installation page. Package hashes were checked against that repository's Packages
index before extraction into `/opt/mctf-spot`; system package versions were not
replaced. The package manifest is archived at
`paper_benchmark/runs/spot-2.16.3-packages/manifest.json`.

Sources:

- https://github.com/UM-ARM-Lab/Efficient-Eng-2-LTL
- https://existentialrobotics.org/LTLCodeGen/
- https://spot.lre.epita.fr/install.html
- https://spot.lre.epita.fr/ltlfilt.html
