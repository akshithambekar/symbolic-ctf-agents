"""Run one response through the migrated executor in a disposable container."""
import contextlib
import io
import json
import os
import shutil
import sys

response = sys.stdin.read(65537)
if len(response) > 65536:
    raise ValueError("Response exceeds limit")
os.mkdir('/tmp/work')
shutil.copytree('/opt/runner/translators', '/tmp/work/translators')
sys.path.insert(0, '/tmp/work')
from translators.ltl_codegen.code_ltl_translator import execute_code_ltl_translate

with contextlib.redirect_stdout(io.StringIO()):
    formula, success = execute_code_ltl_translate(response)
print(json.dumps({'formula': f'[{formula}]', 'success': success}))
