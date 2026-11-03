"""Run the CTF LTL benchmark against selected free OpenRouter Nemotron models."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
from time import perf_counter, sleep
from unittest.mock import patch
from urllib.parse import urlparse

import requests
from maritime_nsrl.translation import translate

from benchmarks.ctf_scoring import score, equivalence


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks/ctf_instruction_set.json"
RESULTS = ROOT / "benchmarks/results/nemotron"
ENDPOINT = "https://openrouter.ai/api/v1"
MODEL_ALIASES = {
    "nano-9b-v2": "nvidia/nemotron-nano-9b-v2:free",
    "nano-12b-v2-vl": "nvidia/nemotron-nano-12b-v2-vl:free",
    "nano": "nvidia/nemotron-3-nano-30b-a3b:free",
    "nano-omni": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "super": "nvidia/nemotron-3-super-120b-a12b:free",
    "ultra": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "lightning": "nvidia/nemotron-3.5-lightning:free",
    "super-vl": "nvidia/nemotron-3.5-super-vl:free",
}
SETTINGS = {
    "temperature": 0,
    "max_tokens": 4096,
    "timeout_seconds": 120,
    "allow_provider_fallbacks": False,
    "translator_validation_repairs": 2,
    "minimum_request_interval_seconds": 3.1,
}
PAID_FALLBACKS = {alias: model.removesuffix(":free")
                  for alias, model in MODEL_ALIASES.items()}
PAID_PRICES_PER_MILLION = {
    "nvidia/nemotron-3-nano-30b-a3b": (0.05, 0.20),
    "nvidia/nemotron-3-super-120b-a12b": (0.08, 0.45),
    "nvidia/nemotron-3-ultra-550b-a55b": (0.50, 2.20),
    "nvidia/nemotron-3.5-lightning": (0.065, 0.18),
    "nvidia/nemotron-3.5-super-vl": (0.10, 0.40),
}
LAST_REQUEST_AT = 0.0


def load_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    path = ROOT / ".env"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        name, separator, value = line.strip().partition("=")
        if separator and name.strip() == "OPENROUTER_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value:
                return value
    return None


def read_dataset():
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    if data.get("semantics") != "LTLf over nonempty finite traces":
        raise ValueError("Unexpected dataset semantics")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Dataset has no items")
    for item in items:
        if not all(isinstance(item.get(key), str) for key in ("id", "instruction", "expected")):
            raise ValueError("Dataset item is missing required fields")
        if item["expected"] == "formula" and not isinstance(item.get("reference_ltl"), str):
            raise ValueError(f"Formula case lacks reference: {item['id']}")
    return data


def safe_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def call_translator(item, glossary, model_alias, model_id, key, timeout):
    attempts = []
    request_started = perf_counter()
    previous_key = os.environ.get("LTL_API_KEY")
    os.environ["LTL_API_KEY"] = key
    original_post = requests.post

    def captured_post(url, **kwargs):
        global LAST_REQUEST_AT
        delay = SETTINGS["minimum_request_interval_seconds"] - (perf_counter() - LAST_REQUEST_AT)
        if LAST_REQUEST_AT and delay > 0:
            sleep(delay)
        LAST_REQUEST_AT = perf_counter()
        body = dict(kwargs.get("json") or {})
        body.update(temperature=SETTINGS["temperature"], max_tokens=SETTINGS["max_tokens"])
        body["provider"] = {"allow_fallbacks": SETTINGS["allow_provider_fallbacks"]}
        kwargs["json"] = body
        kwargs["timeout"] = timeout
        record = {"url": url, "request": body, "requested_model": body.get("model"),
                  "latency_seconds": None,
                  "http_status": None, "resolved_model": None, "usage": None,
                  "response": None, "provider_error": None, "error_type": None}
        attempts.append(record)
        started = perf_counter()
        try:
            response = original_post(url, **kwargs)
            record["http_status"] = response.status_code
            response_body = response.json()
            record["resolved_model"] = response_body.get("model")
            record["usage"] = response_body.get("usage")
            provider_error = response_body.get("error")
            if isinstance(provider_error, dict):
                record["provider_error"] = {
                    name: provider_error.get(name)
                    for name in ("code", "message") if provider_error.get(name) is not None
                }
            choices = response_body.get("choices") or []
            if choices:
                record["response"] = choices[0].get("message", {}).get("content")
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            record["http_status"] = getattr(getattr(exc, "response", None), "status_code", None)
            record["error_type"] = type(exc).__name__
            raise
        except (ValueError, KeyError, TypeError) as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["latency_seconds"] = perf_counter() - started

    result = {"model_alias": model_alias, "requested_model": model_id,
              "used_model": model_id, "endpoint": ENDPOINT, "attempts": attempts,
              "formula": None, "explicit_rejection": False,
              "infrastructure_error": False, "error_type": None,
              "error_message": None, "latency_seconds": None}
    try:
        with patch("maritime_nsrl.translation.requests.post", side_effect=captured_post):
            context = ("CTF benchmark domain glossary:\n" + glossary +
                       "\nInstruction:\n" + item["instruction"])
            artifact = translate(context, backend="llm", model=model_id,
                                 base_url=ENDPOINT, logic_backend="dfa")
        result["formula"] = artifact["ltl"]
        result["validation"] = artifact["validation"]
        result["repair_attempts"] = max(0, len(attempts) - 1)
    except requests.RequestException as exc:
        result.update(infrastructure_error=True, error_type=type(exc).__name__,
                      error_message="OpenRouter request failed")
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error_message=str(exc))
        if attempts and attempts[-1].get("http_status") in {401, 402, 403, 404, 408, 429, 500, 502, 503, 504}:
            result["infrastructure_error"] = True
    finally:
        if previous_key is None:
            os.environ.pop("LTL_API_KEY", None)
        else:
            os.environ["LTL_API_KEY"] = previous_key
    for attempt in attempts:
        raw = attempt.get("response")
        try:
            data = json.loads(raw) if isinstance(raw, str) else None
            if isinstance(data, dict) and "error" in data:
                result["explicit_rejection"] = True
        except (ValueError, TypeError):
            pass
    result["latency_seconds"] = perf_counter() - request_started
    result["provider_requests"] = len(attempts)
    return result


def redact_secret(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [redact_secret(item, secret) for item in value]
    if isinstance(value, dict):
        return {redact_secret(key, secret): redact_secret(item, secret)
                for key, item in value.items()}
    return value


def append_record(path, record, secret):
    with path.open("a", encoding="utf-8") as file:
        file.write(safe_json(redact_secret(record, secret)) + "\n")
        file.flush()


def load_records(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize(output, records, data, selected_models, trials):
    completed = [row for row in records if row.get("status") == "measured"]
    summaries = []
    pairs = []
    for alias in selected_models:
        rows = [row for row in completed if row["model_alias"] == alias]
        formula_rows = [row for row in rows if row.get("expected") == "formula"]
        measured_formula_rows = [row for row in formula_rows if row.get("assessment")]
        latencies = [row["latency_seconds"] for row in rows
                     if isinstance(row.get("latency_seconds"), (int, float))]
        request_counts = [row.get("provider_requests", 0) for row in rows]
        costs = [estimated_cost(row) for row in rows]
        summary = {
            "model_alias": alias, "model_id": MODEL_ALIASES[alias],
            "model_routes_used": sorted({row.get("used_model", row.get("requested_model"))
                                          for row in rows}),
            "scheduled": len(data["items"]) * trials, "measured": len(rows),
            "not_measured": len(data["items"]) * trials - len(rows),
            "syntax_valid_rate": rate([bool(row["assessment"]["syntax_valid"])
                                        for row in measured_formula_rows]),
            "compilable_rate": rate([bool(row["assessment"]["compilable"])
                                      for row in measured_formula_rows]),
            "semantic_correct_rate": rate([row["assessment"]["score"] == "Correct"
                                            for row in formula_rows if row.get("assessment")]),
            "appropriate_rejection_rate": rate([
                row.get("explicit_rejection") is True for row in rows
                if row.get("expected") == "clarify_or_reject"]),
            "repair_rate": rate([row.get("repair_attempts", 0) > 0 for row in rows]),
            "mean_latency_seconds": mean(latencies),
            "median_latency_seconds": quantile(latencies, .5),
            "p90_latency_seconds": quantile(latencies, .9),
            "mean_provider_requests_per_instruction": mean(request_counts),
            "prompt_tokens": sum(token_count(row, "prompt_tokens") for row in rows),
            "completion_tokens": sum(token_count(row, "completion_tokens") for row in rows),
            "estimated_cost_usd": round(sum(costs), 8) if all(x is not None for x in costs) else None,
            "cost_unknown_rows": sum(x is None for x in costs),
        }
        summaries.append(summary)
        for trial in range(trials):
            trial_rows = [row for row in rows if row["trial"] == trial]
            groups = sorted({row.get("paraphrase_group") for row in trial_rows
                             if row.get("paraphrase_group")})
            for group in groups:
                group_rows = [row for row in trial_rows if row.get("paraphrase_group") == group]
                for first, second in itertools.combinations(group_rows, 2):
                    pair = {"model_alias": alias, "trial": trial, "group": group,
                            "equivalent": None, "error": None}
                    if first.get("formula") and second.get("formula"):
                        try:
                            pair.update(equivalence(first["formula"], second["formula"]))
                        except Exception as exc:
                            pair["error"] = f"{type(exc).__name__}: {exc}"
                    else:
                        pair["error"] = "Missing formula; excluded from agreement rate"
                    pairs.append(pair)
    (output / "summary.json").write_text(safe_json(summaries) + "\n", encoding="utf-8")
    (output / "paraphrases.json").write_text(safe_json(pairs) + "\n", encoding="utf-8")
    if summaries:
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    measured_count = sum(item["measured"] for item in summaries)
    report = ["# Free Nemotron CTF LTL benchmark", "",
              f"Measured {measured_count} of {sum(item['scheduled'] for item in summaries)} scheduled model/case/trial outcomes.",
              "", "## Methods", "",
              "The CTF dataset contains 40 project-authored statements in nine categories and 20 paraphrase pairs. Gold meanings are project-authored and still require independent review. The same translator and finite-trace semantics are used for every model. Each response is parsed and compiled by the repository validator; formulas are compared over all nonempty finite Boolean traces using the reachable DFA product.",
              "", "Explicit rejection is counted as correct only for cases labelled clarify_or_reject. Semantic correctness is reported over formula-expected cases. Failed provider calls are marked not measured. These results measure formal agreement with the authored references, not proof that each reference captures every human interpretation.",
              "", "## Results", "",
              "See summary.csv and summary.json for model-level metrics. Raw request/response attempts are in raw.jsonl. `Not measured` cases remain pending and may be resumed with the same output directory.",
              "", "## Provenance", "",
              "Configuration, dataset hash, and source revision are recorded in config.json. Free model variants can change routing, availability, rate limits, and behavior over time. No cost is expected for these `:free` endpoints; OpenRouter account-level limits still apply.", ""]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def rate(values):
    return statistics.mean(values) if values else None


def mean(values):
    return statistics.mean(values) if values else None


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def token_count(row, field):
    return sum(
        attempt.get("usage", {}).get(field, 0) or 0
        for attempt in row.get("attempts", [])
        if isinstance(attempt.get("usage"), dict)
    )


def estimated_cost(row):
    total = 0.0
    for attempt in row.get("attempts", []):
        request_model = attempt.get("requested_model")
        prices = PAID_PRICES_PER_MILLION.get(request_model, (0.0, 0.0))
        usage = attempt.get("usage") or {}
        reported_cost = usage.get("cost")
        if isinstance(reported_cost, (int, float)):
            total += reported_cost
        elif request_model in PAID_PRICES_PER_MILLION:
            total += (usage.get("prompt_tokens", 0) or 0) * prices[0] / 1_000_000
            total += (usage.get("completion_tokens", 0) or 0) * prices[1] / 1_000_000
        elif request_model in PAID_FALLBACKS.values():
            return None
    return total


def run(models, output, trials, timeout, resume, pause):
    key = load_key()
    if not key:
        raise ValueError("Missing OPENROUTER_API_KEY in the process environment or repository-root .env")
    if urlparse(ENDPOINT).hostname != "openrouter.ai":
        raise ValueError("Unexpected provider endpoint")
    data = read_dataset()
    output = Path(output)
    if resume:
        if not output.is_dir():
            raise ValueError("Resume directory does not exist")
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        existing_models = set(config.get("model_aliases", []))
        if (not existing_models.issubset(models) or config.get("trials") != trials
                or config.get("dataset_sha256") != hashlib.sha256(DATASET.read_bytes()).hexdigest()):
            raise ValueError("Resume models must include the original selection, with the same trials and dataset")
        config["model_aliases"] = models
        config["requested_models"] = [MODEL_ALIASES[alias] for alias in models]
        config["paid_fallbacks_on_404_or_429"] = PAID_FALLBACKS
        config.setdefault("resume_sessions", []).append(datetime.now(timezone.utc).isoformat())
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "dataset.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        config = {
            "benchmark_version": data["version"], "requested_models": [MODEL_ALIASES[x] for x in models],
            "model_aliases": models, "trials": trials, "endpoint": ENDPOINT,
            "settings": SETTINGS, "timeout_seconds": timeout,
            "semantics": data["semantics"], "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
            "source_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                capture_output=True, text=True).stdout.strip(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gold_review": "Project-authored references; independent human approval pending",
            "paid_fallbacks_on_404_or_429": PAID_FALLBACKS,
        }
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    raw_path = output / "raw.jsonl"
    records = load_records(raw_path)
    completed = {(row.get("model_alias"), row.get("item_id"), row.get("trial"))
                 for row in records if row.get("status") == "measured"}
    paid_models = {row["model_alias"] for row in records
                   if row.get("used_model") in PAID_FALLBACKS.values()}
    randomizer = random.Random(20261102)
    stop_all = False
    unavailable_models = set()
    for trial in range(trials):
        items = list(data["items"])
        randomizer.shuffle(items)
        for item_index, item in enumerate(items):
            item_models = models[(trial + item_index) % len(models):] + models[:(trial + item_index) % len(models)]
            for alias in item_models:
                if alias in unavailable_models:
                    continue
                identity = (alias, item["id"], trial)
                if identity in completed:
                    continue
                route = PAID_FALLBACKS[alias] if alias in paid_models else MODEL_ALIASES[alias]
                result = call_translator(item, data["glossary"], alias, route, key, timeout)
                status_code = (result.get("attempts", [{}])[-1].get("http_status")
                               if result.get("attempts") else None)
                if (result["infrastructure_error"] and status_code in {404, 429}
                        and alias in PAID_FALLBACKS and alias not in paid_models):
                    failed_free_result = result
                    result = call_translator(
                        item, data["glossary"], alias, PAID_FALLBACKS[alias], key, timeout
                    )
                    result["requested_model"] = MODEL_ALIASES[alias]
                    result["used_model"] = PAID_FALLBACKS[alias]
                    result["fallback_reason"] = f"free_endpoint_http_{status_code}"
                    result["attempts"] = failed_free_result["attempts"] + result["attempts"]
                    result["provider_requests"] += failed_free_result["provider_requests"]
                    result["latency_seconds"] += failed_free_result["latency_seconds"]
                    paid_models.add(alias)
                assessment = score(item, result.get("formula"), result["explicit_rejection"],
                                   result["infrastructure_error"])
                status = "measured" if not result["infrastructure_error"] else "not_measured"
                record = {
                    "item_id": item["id"], "category": item["category"],
                    "paraphrase_group": item["paraphrase_group"], "expected": item["expected"],
                    "trial": trial, "instruction": item["instruction"],
                    "reference_ltl": item["reference_ltl"], "status": status,
                    **result, "assessment": assessment,
                }
                append_record(raw_path, record, key)
                records.append(record)
                if status == "measured":
                    completed.add(identity)
                print(f"{alias} trial={trial + 1}/{trials} {item['id']}: {assessment['score']}", flush=True)
                if result["infrastructure_error"]:
                    status_code = result.get("attempts", [{}])[-1].get("http_status") if result.get("attempts") else None
                    if status_code == 429:
                        stop_all = True
                        break
                    if status_code in {401, 402, 403}:
                        stop_all = True
                        break
                    unavailable_models.add(alias)
                    continue
                if pause:
                    sleep(pause)
            if stop_all:
                break
        if stop_all:
            break
    summarize(output, records, data, models, trials)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=[*MODEL_ALIASES, "all"],
                        default=["nano", "super", "ultra"],
                        help="Nemotron aliases: " + ", ".join(MODEL_ALIASES))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=SETTINGS["timeout_seconds"])
    parser.add_argument("--pause", type=float, default=0.0,
                        help="Seconds between completed instructions")
    parser.add_argument("--output", type=Path,
                        help="New result directory; defaults to benchmarks/results/nemotron/<UTC timestamp>")
    parser.add_argument("--resume", action="store_true",
                        help="Resume pending cases in an existing output directory")
    parser.add_argument("--list", action="store_true", help="List free Nemotron model shortcuts")
    args = parser.parse_args(argv)
    if args.list:
        for alias, model in MODEL_ALIASES.items():
            print(f"{alias}\t{model}")
        return 0
    if args.trials < 1 or args.timeout <= 0 or args.pause < 0:
        parser.error("trials and timeout must be positive; pause cannot be negative")
    models = list(MODEL_ALIASES) if "all" in args.models else args.models
    if len(set(models)) != len(models):
        parser.error("model aliases must be unique")
    output = args.output or RESULTS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        location = run(models, output, args.trials, args.timeout, args.resume, args.pause)
    except Exception as exc:
        parser.exit(2, f"benchmark setup failed: {type(exc).__name__}: {exc}\n")
    print(location)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
