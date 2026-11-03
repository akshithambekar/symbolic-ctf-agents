import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import importlib


code_ltl_exec = None


def get_code_ltl_translate_prompt():
    template_file = os.path.join(os.path.dirname(__file__), "code_ltl_prompt_template.py")
    with open(template_file, "r") as file:
        template = file.readlines()
        template = "".join(template)

    return PromptTemplate(template=template, input_variables=["instruction"])


def execute_code_ltl_translate(code: str):
    template_file = os.path.join(os.path.dirname(__file__), "code_ltl_exec_template.py")
    with open(template_file, "r") as file:
        template = file.readlines()
    # the actual code is contained in ```python code```
    code = code.split("\n")[1:-1]  # remove the first and last line
    output_file = os.path.join(os.path.dirname(__file__), "code_ltl_exec.py")
    with open(output_file, "w") as file:
        for line in template:
            file.write(line)
        for line in code:
            file.write(line + "\n")

    try:
        global code_ltl_exec
        if code_ltl_exec is not None:
            code_ltl_exec = importlib.reload(code_ltl_exec)
        else:
            code_ltl_exec = importlib.import_module("speech_to_ltl.code_ltl_exec")
    except Exception as e:
        print(e)
        return str(e), False
    try:
        ltl = code_ltl_exec.question()
    except Exception as e:
        print(e)
        return str(e), False
    return ltl, True


def code_ltl_translator(
    llm,
    instruction: str,
    verbose: bool = False,
    retry_on_error: bool = True,
    max_retries: int = 3,
    listed: bool = True,
):
    translate_chain = LLMChain(llm=llm, prompt=get_code_ltl_translate_prompt(), verbose=verbose)
    code = translate_chain(dict(instruction=instruction, previous_answer="", failure_reason=""))["text"]
    ltl_or_reason, success = execute_code_ltl_translate(code)
    print(code)
    if not success and retry_on_error:
        for i in range(max_retries):
            code = translate_chain(
                dict(
                    instruction=instruction,
                    previous_answer=f"\nHere is your previous answer:\n{code}",
                    failure_reason=f"\nBut it failed with the following error:\n{ltl_or_reason}",
                )
            )["text"]
            print(f"Failed: {ltl_or_reason}")
            print(f"Retrying {i + 1}/{max_retries}...")
            print(code)
            ltl_or_reason, success = execute_code_ltl_translate(code)
            if success:
                break
    ltl = ltl_or_reason
    if listed:
        ltl = f"[{ltl}]"
    return ltl, success, code
