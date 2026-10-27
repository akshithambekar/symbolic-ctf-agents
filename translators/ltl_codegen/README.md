# Migrated LTLCodeGen baseline

This directory preserves the behavior originally imported into this repository at migration commit `698677f9639d3601112df21a13997ae34b8370cc`.
The imported files retain that snapshot's behavior, with only the executor and import-template package paths adjusted for relocation to `translators/ltl_codegen/`.
The prompt, operators, dependency list, and license are unchanged; this README explains the current organization.
The migration README named [ExistentialRobotics/LTLCodeGen](https://github.com/ExistentialRobotics/LTLCodeGen), directory `speech_to_ltl/speech_to_ltl/`, upstream revision `b5eb6a1`.
This is an identifiable migrated subset, not a verified untouched or complete upstream copy.
See the [history audit and remaining uncertainty](../README.md).
MIT attribution is retained in [LICENSE-LTLCodeGen](LICENSE-LTLCodeGen).

Our recent AST interpreter and stricter prompt now live separately in [`translators/team_translator/`](../team_translator/README.md).
The benchmark runs only that restricted implementation and does not import this executor.
Import the baseline from `translators.ltl_codegen.code_ltl_translator`.
The old `ltl` package path is removed without a compatibility layer; callers wanting our restricted behavior must use `translators.team_translator.translator`.

| File | Imported behavior |
| --- | --- |
| `code_ltl_translator.py` | Legacy `LLMChain`, first/last response-line removal, generated module writing/import/reload, printed output, optional retries enabled by default |
| `code_ltl_exec_template.py` | Repository-qualified `translators.ltl_codegen.*` imports prepended to generated code |
| `code_ltl_prompt_template.py` | Imported few-shot navigation prompt, without our added AST response constraints |
| `ltl_operators.py` | String-building operators with process-wide `prefix=True` and `listed=True` defaults |
| `available_actions.py` | The `reach` action |
| `model.py` | Imported legacy OpenAI helper, including its model default and YAML configuration example |
| `requirements.txt` | Original dependency ranges, including `langchain>=0.3,<1` |

The original executor runs model-generated Python with normal process permissions.
It is unavailable as a live benchmark execution path until an isolation approach has been selected and verified.
Do not send live model output to it.
The required work includes enforcing filesystem, network, process, resource, and credential boundaries; simply launching a subprocess does not provide those boundaries.
Using our AST interpreter in its place would constitute a modified variant rather than the restored baseline.
A future benchmark adapter must explicitly disable its translator retries and provider-client retries.

Offline restoration tests use only trusted, fixed code and fake models in temporary package copies.
They cover the original fenced success, unfenced failure, and normal-use retry behavior without modifying this source directory.
Those tests depend on the baseline requirements as well as the benchmark test environment:

```bash
python -m pip install -r translators/ltl_codegen/requirements.txt -r benchmarks/requirements.txt
python -m pytest -q
```

The legacy chain API remains deprecated; restoring it is intentional for baseline fidelity.
The baseline's mutable operator globals, generated module, and reload cache are separate from our implementation's state but remain limitations for concurrent baseline usage.
No baseline benchmark adapter or semantic checker has been added.
IBM NL2LTL is a distinct future baseline, currently only referenced in `DOCUMENTS.md`.
