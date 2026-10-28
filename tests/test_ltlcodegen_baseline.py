"""Exercise only trusted fixed inputs in a temporary copy of the legacy package."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from translators.team_translator import ltl_operators as restricted_ops
from translators.ltl_codegen import ltl_operators as baseline_ops


BASELINE_DIR = Path(__file__).resolve().parents[1] / "translators" / "ltl_codegen"


@pytest.mark.parametrize("mode", ["fenced", "unfenced", "retry"])
def test_imported_executor_with_trusted_fixed_input(tmp_path, mode):
    # Keep legacy generated modules and reload state out of the source tree and
    # the pytest process. This is test hygiene, not isolation for untrusted code.
    package = tmp_path / "translators"
    package.mkdir()
    shutil.copyfile(BASELINE_DIR.parent / "__init__.py", package / "__init__.py")
    shutil.copytree(BASELINE_DIR, package / "ltl_codegen", ignore=shutil.ignore_patterns(
        "code_ltl_exec.py", "__pycache__",
    ))
    script = r'''
import sys
from langchain_core.language_models import FakeListLLM
from translators.ltl_codegen import ltl_operators as ops
from translators.ltl_codegen.code_ltl_translator import code_ltl_translator, execute_code_ltl_translate
ops.prefix = False
ops.listed = False
code = "def question():\n    target = ap(reach, 'object_7')\n    return ltl_eventually(target)"
fenced = "```python\n" + code + "\n```"
if sys.argv[1] == "fenced":
    assert execute_code_ltl_translate(fenced) == ("F (reach(object_7))", True)
elif sys.argv[1] == "unfenced":
    # Preserve the imported executor's behavior, including its fence assumption.
    assert execute_code_ltl_translate(code)[1] is False
else:
    bad = "```python\ndef question():\n    return undefined_thing\n```"
    model = FakeListLLM(responses=[bad, fenced, "must not be used"])
    formula, ok, generated = code_ltl_translator(model, "Eventually reach object_7.", listed=False)
    assert ok and formula == "F (reach(object_7))" and generated == fenced
    assert model.i == 2  # Baseline normal-use retry behavior is intentionally retained.
'''
    subprocess.run(
        [sys.executable, "-c", script, mode], cwd=tmp_path,
        capture_output=True, text=True, check=True,
    )


def test_operator_settings_are_independent(monkeypatch):
    assert restricted_ops is not baseline_ops
    monkeypatch.setattr(baseline_ops, "prefix", False)
    monkeypatch.setattr(baseline_ops, "listed", False)
    assert restricted_ops.prefix is True and restricted_ops.listed is True
    assert baseline_ops.ap("reach", "object_7") == "reach(object_7)"
    assert restricted_ops.ap("reach", "object_7") == "'reach(object_7)'"


def test_benchmark_does_not_import_original_executor():
    subprocess.run([
        sys.executable, "-c",
        "import sys; from benchmarks import run; "
        "assert run.IMPLEMENTATION_ID == 'restricted-ast-v1'; "
        "assert 'translators.ltl_codegen.code_ltl_translator' not in sys.modules",
    ], capture_output=True, text=True, check=True)
