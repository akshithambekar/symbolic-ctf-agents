# English-to-LTL implementations

Shared cases and orchestration live in `benchmarks/`.
Translator code lives separately so a future comparison can attribute each output to its implementation.
The single-case benchmark currently runs only our restricted implementation.

| Implementation | Location | Relationship and availability |
| --- | --- | --- |
| Migrated LTLCodeGen baseline | [`ltl_codegen/`](ltl_codegen/README.md) | Imported core restored from repository commit `698677f`; unavailable through the benchmark because it executes generated Python |
| Team translator | [`team_translator/`](team_translator/README.md) | Locally modified derivative; current benchmark default with stable ID `restricted-ast-v1` |
| IBM NL2LTL | Reference in [`../DOCUMENTS.md`](../DOCUMENTS.md) | Future baseline only; no integration, dependencies, adapter, or runnable benchmark path |

## Provenance and history audit

The source of truth for restoration is repository migration commit `698677f9639d3601112df21a13997ae34b8370cc`.
The earlier available commit `e463365ca7c1fdc612282c052b0027393f4a0d70` has identical `ltl/` contents, but `698677f` is the migration in the current HEAD's ancestry.
The imported README identifies [ExistentialRobotics/LTLCodeGen](https://github.com/ExistentialRobotics/LTLCodeGen), source directory `speech_to_ltl/speech_to_ltl/`, and upstream revision `b5eb6a1`.
That upstream revision is recorded provenance from the migration, not a fresh comparison against an upstream checkout.
The available local history establishes what this repository imported; it does not establish that those files were unmodified relative to upstream.

| Repository state | Evidence and changes |
| --- | --- |
| `1e348281c755838dce09c98a33dba8fa298b457d`, "Library added" | Added only the IBM NL2LTL reference in `DOCUMENTS.md`, despite the commit message describing a library addition |
| `698677f`, original migration | Added the translator, prompt, operators, `reach` action, import template, model helper, requirements, license, and README under `ltl/` |
| `e8762d603cbb404d18264b9271eaf8299787797f` | Added four fake-model/operator tests and pytest configuration; did not change the translator |
| `89bfbfb88f618ceb1074463aaf9e3a7a9265a726` | Added a single-case benchmark skeleton and documentation; did not change the translator |
| Working tree before this reorganization | Added the AST interpreter, stricter prompt, one-attempt OpenRouter runner, fake HTTP tests, result redaction, ignored JSON outputs, and inclusive-current-step case wording |
| This reorganization | Preserves those local changes in `team_translator/` and `benchmarks/`, restores the baseline code and dependencies to the migration, and adds implementation attribution |

The imported subset is not a complete upstream distribution.
Its README explicitly excluded ROS2 integration, scene/YOLO prompts, robot stack components, and an upstream `ltl_translator.py` described as an NL2LTL baseline.
The local repository does not establish that this omitted upstream module is the IBM project; do not conflate those references.
The imported executor already uses this repository's `ltl.*` package paths, and the imported model helper cites GPT-Synthesizer as inspiration.
These are properties of the migration snapshot.
Moving the baseline to `translators/ltl_codegen/` changes only the executor's generated-module import path and the import template's package paths; its prompt and execution behavior are preserved.
The imported prompt retains navigation examples and object filtering; no capture-the-flag adaptation is claimed.

## Imports, dependencies, and mutable settings

The affected in-repository callers were `benchmarks/run.py` and `tests/test_ltl.py`.
Both now import `translators.team_translator` explicitly.
The function names and tuple return shape of our translator are preserved.
The baseline is now imported from `translators.ltl_codegen.code_ltl_translator`, including its unsafe executor and normal-use retries.
The historical `ltl.code_ltl_translator` path is no longer available; no compatibility alias is provided.
Callers wanting our AST behavior must use `translators.team_translator.translator`.

The two implementations have separate prompt and operator copies.
There is no shared translator code or shared operator state between them.
Both operator modules retain process-wide `prefix` and `listed` globals, so callers must not change an implementation's settings concurrently.
The benchmark leaves our operator defaults at listed-prefix format; changing them in-process would also affect formula output and make its recorded format setting inaccurate.
The baseline also retains its module cache and generated-file state.
Trusted baseline tests run in temporary package copies and subprocesses to avoid source-tree writes and cross-test reload state.
That test arrangement provides no security isolation for untrusted code.

The baseline dependency list remains exactly as imported in `translators/ltl_codegen/requirements.txt`, including the legacy LangChain chain dependency.
Our translator uses `langchain-core`, while `benchmarks/requirements.txt` adds the OpenAI SDK and pytest.
These are version ranges, not a reproducibility lockfile.
Both implementations retain the migrated MIT license attribution.

## Future baseline comparison

Before exposing the original executor to model output, choose and verify an isolation design with explicit filesystem, network, process, resource, and credential boundaries.
A subprocess alone is insufficient because it retains the user's process permissions.
Replacing its executor with our AST interpreter would change the baseline and must be identified as a separate variant.
When a baseline adapter is added, disable both translator and provider-client retries for benchmark runs even though normal baseline usage supports retries.

IBM NL2LTL still needs source/revision and license review, dependency setup, a minimal adapter, and output-notation documentation.
Do not infer an integration from the existing link or the historical commit message.

Semantic checking remains deferred with an explicit `not_run` result.
A future shared evaluator must convert each implementation's native output and the infix reference into one checker's input language, map `reach(object_7)` consistently to one atomic proposition, and preserve standard infinite-trace semantics including current-step `F`.
Verify that checker independently with equivalent and non-equivalent formulas, then distinguish invalid formulas, checker failures, equivalence, and semantic mismatch for every implementation.
Do not compare formula strings as a substitute for equivalence.
No additional cases or accuracy claims are introduced here.
