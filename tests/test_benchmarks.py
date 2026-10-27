"""Pipeline tests include the real SDK over a fake HTTP transport."""

import json
import subprocess
import sys

import httpx
from langchain_core.language_models import FakeListLLM
from openai import OpenAI
import pytest

from benchmarks import run


CASES = run.load_cases(run.DEFAULT_CASES)


@pytest.fixture(autouse=True)
def isolate_env_file(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "ENV_FILE", tmp_path / ".env")


def test_env_file_configuration_and_environment_precedence(monkeypatch, tmp_path):
    run.ENV_FILE.write_text("OPENROUTER_API_KEY=file-key\nOPENROUTER_MODEL=file/model\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_MODEL", "environment/model")
    captured = []
    class FakeModel:
        def __init__(self, model, key):
            captured.append((model, key))

        def close(self):
            pass

    monkeypatch.setattr(run, "OpenRouterModel", FakeModel)
    monkeypatch.setattr(run, "run_cases", lambda cases, llm, model, provider, error: [])
    output = tmp_path / "result.json"
    assert run.main(["--output", str(output)]) == 0
    assert captured == [("environment/model", "file-key")]


@pytest.mark.parametrize("status, expected", [
    (200, "translated"), (429, "rate_limit_error"),
    (503, "provider_error"), (401, "provider_error"), (307, "provider_error"),
])
def test_one_provider_request_and_no_sdk_retries(monkeypatch, status, expected):
    requests = []
    constructors = []
    secret = 'secret-with-"quotes"'

    def handle(request):
        requests.append(request)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": secret}}, headers={
                "retry-after": "0", "location": "https://example.invalid/redirect",
            })
        return httpx.Response(200, json={
            "id": "fake-id", "object": "chat.completion", "created": 0, "model": "test/model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": run.OFFLINE_CODE,
            }}],
        })

    def client_factory(**kwargs):
        http_client = kwargs.pop("http_client")
        assert not http_client.follow_redirects
        http_client.close()
        constructors.append(kwargs)
        return OpenAI(**kwargs, http_client=httpx.Client(
            transport=httpx.MockTransport(handle), follow_redirects=False,
        ))

    monkeypatch.setattr(run, "OpenAI", client_factory)
    model = run.OpenRouterModel("test/model", secret)
    try:
        results = run.run_cases(CASES, model, "test/model", "openrouter")
    finally:
        model.close()
    result = results[0]
    assert result["implementation_id"] == "restricted-ast-v1"
    assert result["semantic_evaluation"]["outcome"] == "not_run"
    assert result["outcome"] == expected
    assert result["attempts"] == result["provider_requests"] == result["model_invocations"] == len(requests) == 1
    assert constructors[0]["max_retries"] == 0
    assert constructors[0]["base_url"] == "https://openrouter.ai/api/v1"
    body = json.loads(requests[0].content)
    assert body["model"] == "test/model"
    assert body["provider"] == {"allow_fallbacks": False}
    assert body["temperature"] == 0 and body["max_tokens"] == 1024
    # Provider bodies and exception text must never become failure details.
    assert secret not in json.dumps(result)
    if status != 200:
        assert result["failure"]["http_status"] == status
        assert result["generated_code"] is None
    else:
        assert result["formula"] == "['EVENTUALLY', 'reach(object_7)']"


def test_translation_failure_separate_from_provider_failure():
    fake = FakeListLLM(responses=["def question():\n    return open('unsafe')", run.OFFLINE_CODE])
    result = run.run_cases(CASES, fake, "offline/fake", "fake")[0]
    assert result["implementation_id"] == "restricted-ast-v1"
    assert result["semantic_evaluation"]["outcome"] == "not_run"
    assert result["outcome"] == "translation_error"
    assert result["generated_code"] == fake.responses[0]
    assert result["formula"] is None
    assert result["attempts"] == result["model_invocations"] == fake.i == 1
    assert result["provider_requests"] == 0


def test_connection_failure_no_retry():
    class BrokenModel:
        calls = 0

        def invoke(self, prompt):
            self.calls += 1
            from openai import APIConnectionError
            raise APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))

    model = BrokenModel()
    result = run.run_cases(CASES, model, "test/model", "openrouter")[0]
    assert result["outcome"] == "provider_error"
    assert model.calls == result["attempts"] == 1


def test_readable_json_redacts_every_field(tmp_path):
    secret = 'fake-key-"\\\n'
    code = run.OFFLINE_CODE.replace("    target", f"    '''{secret!r}'''\n    target")
    results = run.run_cases(CASES, FakeListLLM(responses=[code]), secret, "fake")
    results[0]["failure"] = {"message": secret, secret: secret}
    output = tmp_path / "result.json"
    run.write_results(output, results, secret)
    text = output.read_text()
    payload = json.loads(text)
    assert payload["results"][0]["model_id"] == "[REDACTED]"
    assert payload["results"][0]["failure"] == {"message": "[REDACTED]", "[REDACTED]": "[REDACTED]"}
    assert "\n  \"schema_version\"" in text
    assert payload["results"][0]["elapsed_seconds"] >= 0


def test_redacts_key_echoed_in_generated_code_and_case(tmp_path):
    secret = "fake-openrouter-secret"
    code = run.OFFLINE_CODE.replace("    target", f"    '''{secret}'''\n    target", 1)
    cases = [{**CASES[0], "instruction": secret}]
    results = run.run_cases(cases, FakeListLLM(responses=[code]), "offline/fake", "fake")
    assert results[0]["outcome"] == "translated"
    output = tmp_path / "result.json"
    run.write_results(output, results, secret)
    assert secret not in output.read_text()
    assert "[REDACTED]" in json.loads(output.read_text())["results"][0]["generated_code"]


@pytest.mark.parametrize("model,key,missing", [
    (None, None, ["explicit model ID", "OPENROUTER_API_KEY"]),
    ("test/model", None, ["OPENROUTER_API_KEY"]),
    (None, "fake-key", ["explicit model ID"]),
])
def test_missing_configuration_writes_clear_error(monkeypatch, tmp_path, capsys, model, key, missing):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    if model:
        monkeypatch.setenv("OPENROUTER_MODEL", model)
    if key:
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
    monkeypatch.setattr(run, "OpenAI", lambda **kwargs: pytest.fail("Must not initialize client"))
    output = tmp_path / "result.json"
    assert run.main(["--output", str(output)]) == 1
    assert capsys.readouterr().out == str(output) + "\n"
    result = json.loads(output.read_text())["results"][0]
    assert result["outcome"] == "configuration_error"
    assert result["implementation_id"] == "restricted-ast-v1"
    assert result["semantic_evaluation"]["outcome"] == "not_run"
    assert result["attempts"] == result["provider_requests"] == 0
    for message in missing:
        assert message in result["failure"]["message"]


def test_offline_cli_and_git_ignore(tmp_path):
    output = tmp_path / "result.json"
    completed = subprocess.run(
        [sys.executable, "-m", "benchmarks.run", "--offline", "--output", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert completed.stdout == str(output) + "\n"
    assert completed.stderr == ""
    result = json.loads(output.read_text())["results"][0]
    assert result["outcome"] == "translated"
    assert result["implementation_id"] == "restricted-ast-v1"
    assert result["formula"] == "['EVENTUALLY', 'reach(object_7)']"
    assert result["semantic_evaluation"]["outcome"] == "not_run"
    assert result["attempts"] == 1 and result["provider_requests"] == 0
    ignored = subprocess.run(
        ["git", "check-ignore", "benchmarks/results/example.json"], capture_output=True, check=True, text=True
    )
    assert ignored.stdout.strip() == "benchmarks/results/example.json"


def test_list_needs_no_configuration(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    assert run.main(["--list"]) == 0
    assert "eventually-reach-object-7" in capsys.readouterr().out
