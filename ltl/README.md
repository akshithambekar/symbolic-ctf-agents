# ltl — English → LTL translation

Core of the LTL code-generation translator, migrated from
[ExistentialRobotics/LTLCodeGen](https://github.com/ExistentialRobotics/LTLCodeGen)
(`speech_to_ltl/speech_to_ltl/`, upstream commit `b5eb6a1`). MIT licensed — see `LICENSE-LTLCodeGen`.

The LLM writes Python that calls the operator functions below; running that code
produces the LTL formula, so the formula's syntax is always valid. If the code
errors, the error is sent back to the LLM to retry (up to `max_retries`).

| File | Purpose |
| --- | --- |
| `ltl_operators.py` | `ap()`, `ltl_and`, `ltl_or`, `ltl_not`, `ltl_next`, `ltl_until`, `ltl_eventually`, `ltl_always`, `ltl_imply` |
| `available_actions.py` | Action names the LLM may use in `ap(action, obj)` (currently only `reach`) |
| `code_ltl_prompt_template.py` | Few-shot prompt (examples of instruction → code) |
| `code_ltl_exec_template.py` | Import header prepended to the generated code before it runs |
| `code_ltl_translator.py` | Generate → execute → retry loop; entry point `code_ltl_translator(llm, instruction)` |
| `model.py` | `llm_init()` — builds the OpenAI chat model via LangChain |

## Not migrated

ROS2 node and launch files, YOLO/scene-graph prompts, the NL2LTL baseline
(`ltl_translator.py`), and the robot stack (SSMI, solar_planner, jackal sim).

## Setup

```bash
pip install -r ltl/requirements.txt
```

## Known follow-ups

- `ltl_operators.py`: `prefix = True` / `listed = True` produce the upstream planner's list format.
  Set both to `False` for standard infix LTL (Spot-readable).
- `available_actions.py` + prompt examples are still robot navigation (`reach object_x`);
  replace with the MCTF vocabulary from the rules team.
- `code_ltl_translator.py` writes the LLM's code to `ltl/code_ltl_exec.py` and imports it
  (gitignored). Consider `exec()` in a restricted namespace instead.
- `LLMChain` is deprecated in LangChain; `model.py` expects the API key passed in.
