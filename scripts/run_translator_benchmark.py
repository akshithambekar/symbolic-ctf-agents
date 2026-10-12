"""Standalone benchmark launcher; all artifacts stay inside this project."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from maritime_nsrl.provider_config import load_env
    from maritime_nsrl.translator_benchmark.runner import run
    # Check plotting dependency before spending API calls.
    import matplotlib
    load_env(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    output = ROOT / 'experiments' / f'translator-benchmark-{stamp}-{uuid.uuid4().hex[:6]}'
    print('Saving all benchmark artifacts to:', output, flush=True)
    print('Model:', os.environ.get('OPENROUTER_MODEL'), flush=True)
    run(output, model=os.environ.get('OPENROUTER_MODEL'),
        endpoint=os.environ.get('LTL_BASE_URL', 'https://openrouter.ai/api/v1'),
        trials=args.trials, timeout=args.timeout)
    print('Finished. Report:', output / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
