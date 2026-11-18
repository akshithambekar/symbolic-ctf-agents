"""WSL/POSIX entry point to the repository's bounded infinite-trace checker."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.semantics import evaluate

if __name__ == '__main__':
    value = json.load(sys.stdin)
    print(json.dumps(evaluate(value['generated'], value['reference'],
        generated_notation='infix', checker_path='/opt/mctf-spot/ltlfilt', timeout_seconds=10)))
