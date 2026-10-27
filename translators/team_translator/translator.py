"""Single-attempt code generation with a deliberately small AST interpreter."""

import ast
import re
from pathlib import Path

from langchain_core.prompts import PromptTemplate

from . import ltl_operators as ops


OPERATORS = {
    "ltl_and": (ops.ltl_and, 2),
    "ltl_or": (ops.ltl_or, 2),
    "ltl_not": (ops.ltl_not, 1),
    "ltl_next": (ops.ltl_next, 1),
    "ltl_until": (ops.ltl_until, 2),
    "ltl_eventually": (ops.ltl_eventually, 1),
    "ltl_always": (ops.ltl_always, 1),
    "ltl_imply": (ops.ltl_imply, 2),
}
MAX_CODE = 65_536
MAX_FORMULA = 100_000


class UnsupportedCode(ValueError):
    """The response is outside the supported translation language."""


def get_code_ltl_translate_prompt():
    template = Path(__file__).with_name("prompt_template.py").read_text()
    return PromptTemplate.from_template(template)


def generate_code(llm, instruction: str) -> str:
    prompt = get_code_ltl_translate_prompt().format(
        instruction=instruction, previous_answer="", failure_reason=""
    )
    response = llm.invoke(prompt)
    code = response if isinstance(response, str) else response.content
    if not isinstance(code, str):
        raise UnsupportedCode("Expected a text response.")
    return code


def interpret_code(code: str) -> str:
    """Interpret assignments and approved calls; never compile, exec, or import code."""
    if not isinstance(code, str) or len(code) > MAX_CODE:
        raise UnsupportedCode("Response must be text of at most 65536 characters.")
    source = code.strip()
    if source.startswith("```"):
        match = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", source, re.DOTALL)
        if not match:
            raise UnsupportedCode("Expected one Python code fence with no surrounding prose.")
        source = match.group(1)
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError, ValueError) as error:
        raise UnsupportedCode("Invalid Python syntax.") from error
    if sum(1 for _ in ast.walk(tree)) > 4096:
        raise UnsupportedCode("Response exceeds the AST node limit.")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise UnsupportedCode("Expected only def question().")
    function = tree.body[0]
    args = function.args
    if (function.name != "question" or function.decorator_list or function.returns
            or function.type_comment or getattr(function, "type_params", [])
            or args.posonlyargs or args.args or args.kwonlyargs or args.vararg or args.kwarg):
        raise UnsupportedCode("Expected undecorated, unannotated def question() without arguments.")
    body = function.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]  # Optional function docstring, treated only as data.
    if not body or not isinstance(body[-1], ast.Return):
        raise UnsupportedCode("question() must end with a return statement.")
    variables = {}

    def expression(node):
        if isinstance(node, ast.Name) and node.id in variables:
            return variables[node.id]
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            name = node.id if isinstance(node, ast.Name) else type(node).__name__
            raise UnsupportedCode(f"Unsupported expression or undefined variable: {name}.")
        name = node.func.id
        if node.keywords:
            raise UnsupportedCode("Keyword arguments are unsupported.")
        if name == "ap":
            if len(node.args) != 2:
                raise UnsupportedCode("ap requires two arguments.")
            action, obj = node.args
            if not isinstance(action, ast.Name) or action.id != "reach":
                raise UnsupportedCode("Only the reach action is supported.")
            if (not isinstance(obj, ast.Constant) or not isinstance(obj.value, str)
                    or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", obj.value)):
                raise UnsupportedCode("ap requires a literal object identifier.")
            return ops.ap("reach", obj.value)
        if name not in OPERATORS:
            raise UnsupportedCode(f"Unsupported call: {name}.")
        operator, arity = OPERATORS[name]
        if len(node.args) != arity:
            raise UnsupportedCode(f"{name} requires {arity} arguments.")
        values = [expression(arg) for arg in node.args]
        if sum(map(len, values)) + 64 > MAX_FORMULA:
            raise UnsupportedCode("Formula exceeds the size limit.")
        return operator(*values)

    for statement in body[:-1]:
        if (not isinstance(statement, ast.Assign) or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
                or statement.targets[0].id in {*OPERATORS, "ap", "reach", "question"}):
            raise UnsupportedCode("Only local formula assignments followed by return are supported.")
        variables[statement.targets[0].id] = expression(statement.value)
    return expression(body[-1].value)


def execute_code_ltl_translate(code: str):
    try:
        return interpret_code(code), True
    except (UnsupportedCode, RecursionError) as error:
        return str(error), False


def code_ltl_translator(
    llm, instruction: str, verbose: bool = False,
    retry_on_error: bool = False, max_retries: int = 0, listed: bool = True,
):
    if retry_on_error or max_retries:
        raise ValueError("Translation retries are disabled; use retry_on_error=False, max_retries=0.")
    code = generate_code(llm, instruction)
    formula, success = execute_code_ltl_translate(code)
    if success and listed:
        formula = f"[{formula}]"
    return formula, success, code
