"""Run with any Python >=3.10; requires git and uv on PATH."""
from pathlib import Path
import subprocess
import os

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "475d670650ec5d49691ba707cd1054c4e45d8053"


def run(*args): subprocess.run(args, cwd=ROOT, check=True)


if __name__ == "__main__":
    run("uv", "venv", "--python", "3.10", ".venv")
    vendor = ROOT / "vendor" / "pyquaticus"
    if not vendor.exists():
        run("git", "clone", "https://github.com/mit-ll-trusted-autonomy/pyquaticus.git", str(vendor))
    run("git", "-C", str(vendor), "checkout", COMMIT)
    python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # The simulator-only requirements exclude the optional MOOS bridge, which is
    # irrelevant to local RL and otherwise pulls a native pymoos build on Linux.
    run("uv", "pip", "install", "--python", str(python), "-r", "requirements-lock.txt")
    run("uv", "pip", "install", "--python", str(python), "--no-deps", "-e", str(vendor), "-e", ".")
    print(f"Ready: {python} -m maritime_nsrl.cli --help")
