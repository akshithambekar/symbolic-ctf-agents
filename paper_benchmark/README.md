# Yahya's original CTF translator benchmark

The additive [candidate study](study/README.md) contains 120 instructions,
source citations, separate development/test-candidate families, independent
review worksheets, and a proposed publication protocol. It has not replaced
the frozen benchmark or been run as a new measured comparison.

This branch restores the exact local benchmark reported in **Evaluating
Neuro-Symbolic Reinforcement Learning in Maritime Capture-the-Flag**, Table 3.
The preserved run is `translator-benchmark-20260926-021055-4de671`.
It is a separate experiment from the newer repository's `benchmarks/` tools.

## What is preserved

- 40 project-authored CTF statements, nine categories, 20 paraphrase pairs.
- Three independent trials per statement per method: 120 outputs per method.
- The original direct English-to-LTLf JSON translator and validation/repair logic.
- The original CTF-adapted translation-only LTLCodeGen implementation, including
  its upstream prompt/operators, object-to-CTF mapping, restricted AST interpreter,
  license, and repair behavior. It is not the complete robotics system.
- The exact requested model: `nvidia/nemotron-3-ultra-550b-a55b:free`, through
  `https://openrouter.ai/api/v1`; 120-second request timeout and provider sampling
  defaults. Current translator: up to 3 calls; LTLCodeGen adapter: up to 4 calls.
- Finite-trace DFA scoring, raw requests/responses, attempts, latencies, human
  review sheet, summary CSV/JSON, plots, configuration, dependencies, and hashes.

`reference_run/` is copied byte-for-byte from the local experiment. All 25 entries
in its original source/dataset manifest matched before import. An additional
manifest covers the complete preserved folder. Do not edit historical artifacts.
Git line-ending conversion is disabled for that folder to preserve its hashes
on Windows and POSIX checkouts.
The runner imports `reference_run/source` explicitly, so neither the latest RL
translator nor `translators/team_translator` replaces the original implementation.

## Paper results and denominators

| Metric | Original direct translator | Original CTF LTLCodeGen adapter |
| --- | --- | --- |
| Measured outputs | 120 | 120 |
| Formula-reference outputs | 96 | 96 |
| Semantic correctness | 90/96 = 93.75% | 69/96 = 71.875% |
| Syntax valid | 95/96 = 98.9583% | 96/96 = 100% |
| Compilable | 95/96 = 98.9583% | 96/96 = 100% |
| Repair rate | 52/120 = 43.3333% | 18/120 = 15% |
| Mean latency, including repairs | 34.9052 s | 53.6617 s |
| Comparable paraphrase equivalence | 42/50 = 84% | 42/52 = 80.7692% |

Eight statements have no gold formula and require rejection/review, accounting for
the other 24 outputs per method. Syntax/compilation/semantic/predicate rates use
the 96 formula-reference outputs; repairs and latency use all measured outputs.
The paper's rounded values are preserved, with these denominator clarifications.
Gold interpretations are project-authored and still need independent approval.
Equivalence is agreement with those formulas under LTLf, not proof of English
meaning or physical realizability. No universal superiority claim follows.

## Separate lightweight setup

The historical live run used the local Python 3.13.3 translator environment.
Its saved dependency versions are in `reference_run/dependencies.json`.
The translation-only tools do not require PyQuaticus, PyTorch, ROS, Spot, or Docker.

```powershell
uv venv --no-project --python 3.13 .venv-paper
uv pip install --python .venv-paper/Scripts/python.exe -r paper_benchmark/requirements-dev.txt
.\.venv-paper\Scripts\python.exe paper_benchmark/run.py verify
.\.venv-paper\Scripts\python.exe -m pytest paper_benchmark/tests/test_reference.py -q
.\.venv-paper\Scripts\python.exe paper_benchmark/run.py rescore
```

Verification and re-scoring make **zero model/API requests**. Re-scoring writes to
a unique ignored `paper_benchmark/runs/` directory. The supporting legacy tests
run in a subprocess to avoid mixing this package with current repository code.

## New live trials

Create a repository-root `.env` (ignored by Git), or export your key:

```dotenv
OPENROUTER_API_KEY=your-key
```

Then run the original two methods on the original dataset and model:

```powershell
.\.venv-paper\Scripts\python.exe paper_benchmark/run.py run
```

This schedules 240 measured outcomes plus any repairs. It is not started by
verification. Model availability, provider routing, and model behavior can change;
a new run cannot recreate historical responses exactly. There is no
application-level switch to a different or paid model after failure.

Use `--trials 1` for a smaller exploratory run. `--model`, `--endpoint`,
`--timeout`, `--env-file`, and a fresh `--output` directory can be explicit.
Changing a model or trial budget creates a new experimental condition; the
historical results remain the reference, not measurements of the new condition.

## Extending this branch

Add IBM NL2LTL or additional models as explicitly named new implementations and
conditions, reusing the dataset, glossary, finite-trace validation, scorer, and
review protocol. Preserve raw outputs and exact model/backend/version metadata.
Match request budgets or state unequal repair budgets. Do not replace the direct
translator with the newer LTLCodeGen-derived team translator under the same name.

## IBM third arm added

The separately labeled IBM NL2LTL arm is now implemented. See
[IBM setup, adaptations, limitations, and commands](ibm/README.md).
It uses the real pinned package in its own Python 3.10 environment, the original
dataset/scorer, and the same requested model. After its setup, run all tests with
`python -m pytest paper_benchmark/tests -q` from the paper environment.

```powershell
.\.venv-paper\Scripts\python.exe paper_benchmark/run.py compare
```

This starts a fresh three-method comparison, not an update of the preserved
two-method study. Use `--methods ibm --trials 1 --case-id capture-0` for a smoke
check. The native request/repair budgets differ and are recorded explicitly.
For reasoning models, `--ibm-max-tokens 4096` is a separately identified adapter
variant; the native 200-token smoke returned no final text, whereas the 4096-token
smoke returned correct formulas for both checked cases. These are integration
checks, not a completed IBM accuracy benchmark.
The newer main branch's IBM dependency lockfile alone did not constitute a
completed comparison. The new
`llm-ltl-benchmark` branch's partial model experiment is also separate from this
paper's measured two-translator benchmark.

The frozen `reference_run/REPORT.md` is retained unchanged, including its older
generic architecture-decision wording. Use the actual raw results and summaries,
this README's denominators, and the paper's stated limitations for interpretation.
