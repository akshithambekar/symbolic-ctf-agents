"""Build and invoke an offline, resource-limited container for one response."""
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import tempfile
import threading
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve().parent
IMAGE = 'ltlcodegen-isolated:local'
TIMEOUT = 20
OUTPUT_LIMIT = 65536


def docker_env():
    # Do not forward the host process environment or provider credentials.
    allowed = ('PATH', 'HOME', 'DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_CONFIG',
               'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH', 'TMPDIR')
    return {key: os.environ[key] for key in allowed if key in os.environ}


def build_image():
    with tempfile.TemporaryDirectory(prefix='ltlcodegen-build-') as temp:
        context = Path(temp)
        shutil.copy(RUNNER / 'Dockerfile', context)
        shutil.copy(RUNNER / 'worker.py', context)
        shutil.copy(ROOT / 'translators/ltl_codegen/requirements.txt', context / 'requirements.txt')
        (context / 'translators').mkdir()
        shutil.copy(ROOT / 'translators/__init__.py', context / 'translators')
        shutil.copytree(ROOT / 'translators/ltl_codegen', context / 'translators/ltl_codegen',
                        ignore=shutil.ignore_patterns('code_ltl_exec.py', '__pycache__', '*.pyc'))
        subprocess.run(['docker', 'build', '-t', IMAGE, str(context)], check=True, env=docker_env())


def execute(response):
    if len(response.encode('utf-8')) > OUTPUT_LIMIT:
        raise ValueError('Model response exceeds execution limit.')
    name = 'ltlcodegen-' + uuid4().hex
    cmd = ['docker', 'run', '--pull', 'never', '--name', name, '--rm', '-i', '--network', 'none',
           '--read-only', '--user', '65534:65534', '--cap-drop', 'ALL',
           '--security-opt', 'no-new-privileges', '--pids-limit', '64',
           '--memory', '512m', '--memory-swap', '512m', '--cpus', '1',
           '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=32m,mode=1777',
           '--log-driver', 'none', IMAGE]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=docker_env())
    try:
        def send_input():
            try:
                process.stdin.write(response.encode('utf-8'))
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass

        threading.Thread(target=send_input, daemon=True).start()
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        output = bytearray()
        deadline = time.monotonic() + TIMEOUT
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Container execution timed out.')
            events = selector.select(remaining)
            if not events:
                raise TimeoutError('Container execution timed out.')
            chunk = os.read(process.stdout.fileno(), 8192)
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > OUTPUT_LIMIT:
                raise ValueError('Container output exceeds limit.')
        if process.wait(timeout=max(0.1, deadline - time.monotonic())) != 0:
            raise RuntimeError('Container execution failed.')
        result = json.loads(output)
        if not isinstance(result, dict) or not isinstance(result.get('formula'), str) or type(result.get('success')) is not bool:
            raise ValueError('Invalid container result.')
        return result['formula'], result['success']
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, env=docker_env(), timeout=10, check=False)


if __name__ == '__main__':
    build_image()
