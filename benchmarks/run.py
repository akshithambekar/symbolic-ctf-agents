"""Record one translation attempt and a separate infinite-trace semantic check."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from dotenv import dotenv_values
from openai import APIConnectionError, APIStatusError, DefaultHttpxClient, OpenAI

from translators.team_translator import IMPLEMENTATION_ID
from translators.team_translator.translator import UnsupportedCode, code_ltl_translator
from benchmarks.semantics import DEFAULT_TIMEOUT_SECONDS, evaluate as evaluate_semantics


DEFAULT_CASES = Path(__file__).with_name("cases.json")
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
RESULTS_DIR = Path(__file__).with_name("results")
SETTINGS = {
    "temperature": 0.0,
    "max_tokens": 1024,
    "timeout_seconds": 60.0,
    "translator_retries": 0,
    "provider_client_retries": 0,
    "provider_allow_fallbacks": False,
    "follow_redirects": False,
    "formula_format": "listed_prefix",
    "semantic_timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
}
OFFLINE_CODE = """def question():
    target = ap(reach, 'object_7')
    return ltl_eventually(target)
"""


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list):
        raise ValueError("The cases file must contain a JSON list.")
    for case in cases:
        if (not isinstance(case, dict)
                or any(not isinstance(case.get(key), str)
                       for key in ("id", "instruction", "reference_formula"))):
            raise ValueError("Every case requires string id, instruction, and reference_formula fields.")
    return cases


class ProviderResponseError(ValueError):
    """The provider did not return a chat completion."""


class OpenRouterModel:
    def __init__(self, model_id: str, api_key: str):
        self.model_id = model_id
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=0,
            timeout=SETTINGS["timeout_seconds"],
            http_client=DefaultHttpxClient(follow_redirects=False),
        )

    def invoke(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=SETTINGS["temperature"],
            max_tokens=SETTINGS["max_tokens"],
            extra_body={"provider": {"allow_fallbacks": False}},
        )
        if getattr(response, "error", None) or not response.choices:
            raise ProviderResponseError("Provider returned no completion.")
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ProviderResponseError("Provider returned no text completion.")
        return content

    def close(self):
        self.client.close()


def failure_details(error: Exception) -> tuple[str, dict]:
    # Do not persist provider exception text, bodies, headers, or request objects.
    # They can contain credentials or echo request data.
    if isinstance(error, APIStatusError):
        status = error.status_code
        outcome = "rate_limit_error" if status == 429 else "provider_error"
        message = "Provider rate limit." if status == 429 else "Provider HTTP failure."
        return outcome, {"type": type(error).__name__, "http_status": status, "message": message}
    if isinstance(error, (APIConnectionError, ProviderResponseError)):
        return "provider_error", {"type": type(error).__name__, "message": "Provider connection or response failure."}
    if isinstance(error, UnsupportedCode):
        return "translation_error", {"type": type(error).__name__, "message": str(error)}
    return "runner_error", {"type": type(error).__name__, "message": "Unexpected local pipeline failure."}


def run_cases(cases, llm, model_id, provider, configuration_error=None, implementation="restricted"):
    results = []
    for case in cases:
        start = perf_counter()
        result = {
            "case": case,
            "implementation_id": IMPLEMENTATION_ID if implementation == "restricted" else "ltlcodegen-container-v1",
            "model_id": model_id,
            "provider": provider,
            "settings": dict(SETTINGS),
            "attempts": 0,
            "model_invocations": 0,
            "provider_requests": 0,
            "generated_code": None,
            "formula": None,
            "outcome": "configuration_error",
            "failure": None,
            "semantic_evaluation": {"outcome": "not_run", "reason": "Translation did not produce a formula."},
        }
        if configuration_error:
            result["failure"] = {"type": "ConfigurationError", "message": configuration_error}
        else:
            class CountedModel:
                def invoke(self, prompt):
                    result["model_invocations"] += 1
                    if provider == "openrouter":
                        result["provider_requests"] += 1
                    return llm.invoke(prompt)

            result["attempts"] = 1
            try:
                if implementation == "restricted":
                    formula, success, code = code_ltl_translator(
                        CountedModel(), case["instruction"], retry_on_error=False, max_retries=0
                    )
                else:
                    from langchain_core.prompts import PromptTemplate
                    from benchmarks.ltlcodegen_runner.runner import execute

                    template = (Path(__file__).resolve().parent.parent / "translators/ltl_codegen/code_ltl_prompt_template.py").read_text()
                    prompt = PromptTemplate(template=template, input_variables=["instruction"]).format(
                        instruction=case["instruction"], previous_answer="", failure_reason=""
                    )
                    code = CountedModel().invoke(prompt)
                    result["generated_code"] = code
                    formula, success = execute(code)
                result["generated_code"] = code
                if success:
                    result.update(formula=formula, outcome="translated")
                else:
                    result.update(outcome="translation_error", failure={
                        "type": "UnsupportedCode" if implementation == "restricted" else "ExecutionError", "message": formula,
                    })
            except Exception as error:
                result["outcome"], result["failure"] = failure_details(error)
        if result["outcome"] == "translated":
            try:
                result["semantic_evaluation"] = evaluate_semantics(
                    result["formula"], case["reference_formula"],
                    generated_notation=SETTINGS["formula_format"],
                    timeout_seconds=SETTINGS["semantic_timeout_seconds"],
                )
            except Exception:
                result["semantic_evaluation"] = {
                    "outcome": "checker_error", "checker": "spot-ltlfilt",
                    "message": "Unexpected semantic evaluator failure.",
                }
        result["elapsed_seconds"] = perf_counter() - start
        results.append(result)
    return results


def redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {redact(key, secret): redact(item, secret) for key, item in value.items()}
    return value


def write_results(path: Path, results: list[dict], secret: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(), "results": results}
    # Redact every string, including generated code, case metadata, and failures.
    path.write_text(json.dumps(redact(payload, secret), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--list", action="store_true", help="List cases without calling a model")
    parser.add_argument("--model", help="Explicit OpenRouter model ID; defaults to OPENROUTER_MODEL")
    parser.add_argument("--offline", action="store_true", help="Use a canned fake response, with no network")
    parser.add_argument("--implementation", choices=["restricted", "ltlcodegen-container"], default="restricted")
    parser.add_argument("--output", type=Path, help="Result JSON path; defaults to benchmarks/results/<unique>.json")
    args = parser.parse_args(argv)
    cases = load_cases(args.cases)
    if args.list:
        for case in cases:
            print(f"{case['id']}: {case['instruction']} -> {case['reference_formula']}")
        return 0

    local_config = dotenv_values(ENV_FILE)
    secret = (os.environ.get("OPENROUTER_API_KEY", local_config.get("OPENROUTER_API_KEY")) or "").strip()
    model_id = (args.model or os.environ.get("OPENROUTER_MODEL", local_config.get("OPENROUTER_MODEL")) or "").strip()
    configuration_error = None
    llm = None
    provider = "fake" if args.offline else "openrouter"
    if args.offline:
        from langchain_core.language_models import FakeListLLM
        response = OFFLINE_CODE if args.implementation == "restricted" else "```python\n" + OFFLINE_CODE.rstrip() + "\n```"
        llm = FakeListLLM(responses=[response])
        model_id = "offline/fake"
    else:
        missing = []
        if not model_id:
            missing.append("an explicit model ID (--model or OPENROUTER_MODEL)")
        if not secret:
            missing.append("OPENROUTER_API_KEY")
        if missing:
            configuration_error = "Missing " + " and ".join(missing) + ". No provider request was made."
        else:
            llm = OpenRouterModel(model_id, secret)
    try:
        if args.implementation == "restricted":
            results = run_cases(cases, llm, model_id or None, provider, configuration_error)
        else:
            results = run_cases(cases, llm, model_id or None, provider, configuration_error, args.implementation)
    finally:
        if isinstance(llm, OpenRouterModel):
            llm.close()
    path = args.output or RESULTS_DIR / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}.json"
    write_results(path, results, secret)
    print(path)
    return 0 if all(result["outcome"] == "translated" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
