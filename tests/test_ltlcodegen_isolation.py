"""Docker integration checks; enable explicitly after building the runner image."""
import os
from pathlib import Path

import pytest

from benchmarks.ltlcodegen_runner import runner

pytestmark = pytest.mark.skipif(os.environ.get('LTL_DOCKER_TESTS') != '1', reason='Docker integration tests are opt-in')


def fenced(body):
    return '```python\ndef question():\n' + '\n'.join('    ' + line for line in body.splitlines()) + '\n```'


def test_fixed_responses():
    assert runner.execute(fenced("return ltl_eventually(ap(reach, 'object_7'))")) == ("['EVENTUALLY', 'reach(object_7)']", True)
    formula, success = runner.execute('def question():\n    return 1')
    assert not success
    assert isinstance(formula, str)


def test_isolation(monkeypatch, tmp_path):
    secret = 'isolation-test-secret-41941'
    monkeypatch.setenv('OPENROUTER_API_KEY', secret)
    sentinel = tmp_path / 'host-only-sentinel'
    sentinel.write_text(secret)
    body = f'''import os, socket
assert os.getuid() != 0, 'uid'
assert 'OPENROUTER_API_KEY' not in os.environ, 'key'
assert not os.path.exists({str(sentinel)!r}), 'host file'
assert not os.path.exists('/root/.env'), 'root env'
s = socket.socket()
s.settimeout(1)
try:
    s.connect(('1.1.1.1', 53))
except OSError:
    pass
else:
    raise AssertionError('network available')
try:
    open('/opt/runner/probe', 'w')
except OSError:
    pass
else:
    raise AssertionError('root filesystem writable')
assert open('/sys/fs/cgroup/pids.max').read().strip() == '64', 'pids'
assert open('/sys/fs/cgroup/memory.max').read().strip() == '536870912', 'memory'
open('/tmp/probe', 'w').write('ok')
return 'isolated' '''
    assert runner.execute(fenced(body)) == ('[isolated]', True)


def test_timeout(monkeypatch):
    monkeypatch.setattr(runner, 'TIMEOUT', 3)
    with pytest.raises(TimeoutError):
        runner.execute(fenced('while True: pass'))
