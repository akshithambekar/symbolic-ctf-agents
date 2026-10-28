"""Shared infinite-trace equivalence using Spot, under one wall-clock deadline."""

import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from time import perf_counter

from benchmarks.formula_notation import InvalidFormula, MAX_CHARACTERS, to_spot


DEFAULT_TIMEOUT_SECONDS = 10.0
ROOT = Path(__file__).resolve().parents[1]
OUTCOMES = {
    "equivalent", "semantic_mismatch", "invalid_generated_formula",
    "invalid_reference_formula", "checker_error", "checker_timeout",
}


def evaluate(generated, reference, *, generated_notation="listed_prefix",
             timeout_seconds=DEFAULT_TIMEOUT_SECONDS, checker_path=None):
    """Evaluate untrusted strings. Conversion and every checker call run in a worker.

    Precedence: validate generated, validate reference, initialize/check with Spot.
    The parent enforces one deadline and kills the worker's entire process group.
    """
    start = perf_counter()
    result = {
        "outcome": "checker_error", "checker": "spot-ltlfilt", "checker_version": None,
        "timeout_seconds": timeout_seconds, "elapsed_seconds": 0.0,
    }
    process = None
    try:
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("Invalid evaluation timeout.")
        if os.name != "posix":
            raise ValueError("The bounded checker worker requires a POSIX platform.")
        executable = checker_path if checker_path is not None else os.environ.get("LTLFILT", "ltlfilt")
        if not isinstance(executable, str) or len(executable) > 4096:
            raise ValueError("Invalid checker executable setting.")
        # Bound serialization before spawning. Invalid values become invalid
        # formula data in the worker, preserving generated-before-reference order.
        def bounded(value):
            return value if isinstance(value, str) and len(value) <= MAX_CHARACTERS else None

        payload = json.dumps({
            "generated": bounded(generated), "reference": bounded(reference),
            "generated_notation": bounded(generated_notation), "checker_path": executable,
        })
        environment = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C"}
        process = subprocess.Popen(
            [sys.executable, "-m", "benchmarks.semantics"], cwd=ROOT, env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        remaining = timeout_seconds - (perf_counter() - start)
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout_seconds)
        output, _ = process.communicate(payload, timeout=remaining)
        if process.returncode != 0:
            raise ValueError("Checker worker failed.")
        checked = json.loads(output)
        if not isinstance(checked, dict) or checked.get("outcome") not in OUTCOMES:
            raise ValueError("Invalid checker worker response.")
        result.update(checked)
    except subprocess.TimeoutExpired:
        result.update(outcome="checker_timeout", message="Semantic evaluation exceeded its wall-clock deadline.")
    except Exception:
        # Neither raw subprocess diagnostics nor exceptions are safe result data.
        result.update(outcome="checker_error", message="Could not start or communicate with the checker worker.")
    finally:
        if process is not None:
            # Also terminate descendants if the worker failed while Spot was alive.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
        result["elapsed_seconds"] = perf_counter() - start
    return result


def _spot(command):
    # Regular files bound native diagnostics even if the executable misbehaves.
    # The worker's inherited RLIMIT_FSIZE also applies to its Spot subprocesses.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        stdout.seek(0)
        stderr.seek(0)
        return completed.returncode, stdout.read(65_537), stderr.read(65_537)


def _evaluate_worker(request):
    result = {"outcome": "checker_error", "checker_version": None}
    atoms = set()
    for side, notation in (("generated", request["generated_notation"]), ("reference", "infix")):
        try:
            formula, names = to_spot(request[side], notation)
        except (InvalidFormula, RecursionError):
            result.update(outcome=f"invalid_{side}_formula", message=f"Invalid or over-limit {side} formula.")
            return result
        result[f"normalized_{side}"] = formula
        atoms.update(names)
    result["atomic_propositions"] = sorted(atoms)
    try:
        executable = shutil.which(request["checker_path"])
        if executable is None:
            raise ValueError("Missing executable.")
        code, output, errors = _spot([executable, "--version"])
        version = re.search(rb"(?:^|[/\\])ltlfilt \(spot\) ([0-9][^\s]*)", output)
        if code != 0 or errors or not version:
            raise ValueError("Checker initialization failed.")
        result["checker_version"] = version.group(1).decode("ascii")
        # Strict Spot parsing is checked independently before equivalence. A
        # failure here is a checker/converter failure, not evidence of mismatch.
        for side in ("generated", "reference"):
            code, output, errors = _spot([
                executable, "--ltl", "--count", "--formula=" + result[f"normalized_{side}"],
            ])
            if code != 0 or output.strip() != b"1" or errors:
                raise ValueError("Spot rejected normalized LTL.")
        code, output, errors = _spot([
            executable, "--ltl", "--count",
            "--formula=" + result["normalized_generated"],
            "--equivalent-to=" + result["normalized_reference"],
        ])
        if errors or (code, output.strip()) not in {(0, b"1"), (1, b"0")}:
            raise ValueError("Checker equivalence failed.")
        result["outcome"] = "equivalent" if code == 0 else "semantic_mismatch"
    except Exception:
        result.update(outcome="checker_error", message="Spot is unavailable, failed initialization, or failed checking.")
    return result


def _main():
    import resource

    resource.setrlimit(resource.RLIMIT_FSIZE, (131_072, 131_072))
    request = json.loads(sys.stdin.read(2 * MAX_CHARACTERS * 6 + 8192))
    print(json.dumps(_evaluate_worker(request)))


if __name__ == "__main__":
    _main()
