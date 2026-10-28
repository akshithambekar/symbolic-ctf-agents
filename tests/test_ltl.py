"""Translator tests use fake models and never require an API key or network."""

import pytest
from langchain_core.language_models import FakeListLLM

from translators.team_translator import ltl_operators as ops
from translators.team_translator.translator import code_ltl_translator, execute_code_ltl_translate


@pytest.fixture(autouse=True)
def infix_format(monkeypatch):
    monkeypatch.setattr(ops, "prefix", False)
    monkeypatch.setattr(ops, "listed", False)


def fenced(code: str) -> str:
    return f"```python\n{code}\n```"


GOOD_CODE = (
    "def question():\n"
    "    flag = ap(reach, 'flag_blue')\n"
    "    zone = ap(reach, 'zone_red')\n"
    "    return ltl_and(ltl_eventually(flag), ltl_always(ltl_not(zone)))"
)
EXPECTED = "(F (reach(flag_blue))) & (G (! (reach(zone_red))))"
CASE_CODE = "def question():\n    target = ap(reach, 'object_7')\n    return ltl_eventually(target)"


def test_operators():
    a, b = ops.ap("reach", "a"), ops.ap("reach", "b")
    assert ops.ltl_eventually(a) == "F (reach(a))"
    assert ops.ltl_until(a, b) == "(reach(a)) U (reach(b))"


@pytest.mark.parametrize("wrap", [lambda code: code, fenced])
def test_execute_good_code(wrap):
    formula, ok = execute_code_ltl_translate(wrap(GOOD_CODE))
    assert ok and formula == EXPECTED


@pytest.mark.parametrize("wrap", [lambda code: code, fenced])
def test_fake_translation_case(wrap, monkeypatch, capsys):
    monkeypatch.setattr(ops, "prefix", True)
    monkeypatch.setattr(ops, "listed", True)
    code = wrap(CASE_CODE)
    llm = FakeListLLM(responses=[code, "must not be used"])
    formula, ok, generated = code_ltl_translator(llm, "Eventually reach object_7.")
    assert ok and formula == "['EVENTUALLY', 'reach(object_7)']"
    assert generated == code
    assert llm.i == 1  # Exactly one invocation.
    assert capsys.readouterr().out == ""


def test_execute_bad_code():
    msg, ok = execute_code_ltl_translate(fenced("def question():\n    return ltl_bogus(1)"))
    assert not ok and "ltl_bogus" in msg


def test_translator_does_not_retry():
    llm = FakeListLLM(responses=["def question():\n    return undefined_thing", GOOD_CODE])
    _, ok, _ = code_ltl_translator(llm, "Eventually reach object_7.", listed=False)
    assert not ok and llm.i == 1


def test_cannot_enable_retries():
    with pytest.raises(ValueError, match="retries are disabled"):
        code_ltl_translator(FakeListLLM(responses=[GOOD_CODE]), "instruction", retry_on_error=True)


@pytest.mark.parametrize("code", [
    "import os\n" + CASE_CODE,
    "@print('side effect')\n" + CASE_CODE,
    "def question(x=open('file', 'w')):\n    return x",
    "def question() -> print('side effect'):\n    return ap(reach, 'object_7')",
    "def question():\n    import os\n    return ap(reach, 'object_7')",
    "def question():\n    while True: pass\n    return ap(reach, 'object_7')",
    "def question():\n    if True: return ap(reach, 'object_7')\n    return ap(reach, 'object_7')",
    "def question():\n    return __import__('os')",
    "def question():\n    return ap.__globals__",
    "def question():\n    return ap(reach, 'object_7')[0]",
    "def question():\n    return [ap(reach, 'object_7') for x in range(10)]",
    "def question():\n    return 'F reach(object_7)'",
    "def question():\n    return ltl_eventually('reach(object_7)')",
    "def question():\n    return ap(reach, 'object_7) | true')",
    "def question():\n    return ap(other_action, 'object_7')",
    "def question():\n    return ltl_eventually(ap(reach, 'object_7'), 1)",
    "def question():\n    return ltl_eventually(a=ap(reach, 'object_7'))",
    "def question():\n    return ltl_eventually(*[ap(reach, 'object_7')])",
    "def question():\n    reach = ap(reach, 'object_7')\n    return reach",
    CASE_CODE + "\nprint('side effect')",
    "Here is your code:\n" + fenced(CASE_CODE),
    fenced(CASE_CODE) + "\n" + fenced(CASE_CODE),
    "def question():\n    return ap(reach, 'object_7')\n    x = ap(reach, 'object_8')",
])
def test_reject_unsupported_code(code):
    _, ok = execute_code_ltl_translate(code)
    assert not ok


def test_rejected_code_has_no_side_effects(tmp_path):
    sentinel = tmp_path / "unsafe"
    code = f"def question():\n    value = open({str(sentinel)!r}, 'w').write('unsafe')\n    return value"
    assert execute_code_ltl_translate(code)[1] is False
    assert not sentinel.exists()


def test_docstring_and_reassignment():
    code = "def question():\n    '''instruction'''\n    a = ap(reach, 'object_7')\n    a = ltl_next(a)\n    return ltl_eventually(a)"
    assert execute_code_ltl_translate(code) == ("F (X (reach(object_7)))", True)


def test_formula_expansion_is_bounded():
    code = "def question():\n    a = ap(reach, 'object_7')\n" + "    a = ltl_and(a, a)\n" * 30 + "    return a"
    message, ok = execute_code_ltl_translate(code)
    assert not ok and "size limit" in message
