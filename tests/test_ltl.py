"""Minimal unit tests for the LTL translator. No API key or network needed:
the LLM is replaced by LangChain's FakeListLLM, which returns canned answers."""
import pytest
from langchain_core.language_models import FakeListLLM

from ltl import ltl_operators as ops
from ltl.code_ltl_translator import code_ltl_translator, execute_code_ltl_translate


@pytest.fixture(autouse=True)
def infix_format(monkeypatch):
    # Standard infix LTL instead of upstream's prefix/list format
    monkeypatch.setattr(ops, "prefix", False)
    monkeypatch.setattr(ops, "listed", False)


def fenced(code: str) -> str:
    # The translator strips the first and last line (the ```python fences)
    return f"```python\n{code}\n```"


GOOD_CODE = fenced(
    "def question():\n"
    "    flag = ap(reach, 'flag_blue')\n"
    "    zone = ap(reach, 'zone_red')\n"
    "    return ltl_and(ltl_eventually(flag), ltl_always(ltl_not(zone)))"
)
EXPECTED = "(F (reach(flag_blue))) & (G (! (reach(zone_red))))"


# 1. Operators build the right formula
def test_operators():
    a, b = ops.ap("reach", "a"), ops.ap("reach", "b")
    assert ops.ltl_eventually(a) == "F (reach(a))"
    assert ops.ltl_until(a, b) == "(reach(a)) U (reach(b))"


# 2. Running LLM-written code produces the formula
def test_execute_good_code():
    ltl, ok = execute_code_ltl_translate(GOOD_CODE)
    assert ok and ltl == EXPECTED


# 3. Broken code is reported as a failure, not a crash
def test_execute_bad_code():
    msg, ok = execute_code_ltl_translate(fenced("def question():\n    return ltl_bogus(1)"))
    assert not ok and "ltl_bogus" in msg


# 4. End to end: first answer fails, retry succeeds
def test_translator_retries():
    llm = FakeListLLM(responses=[fenced("def question():\n    return undefined_thing"), GOOD_CODE])
    ltl, ok, _ = code_ltl_translator(llm, "Capture the blue flag and never enter the red zone", listed=False)
    assert ok and ltl == EXPECTED
