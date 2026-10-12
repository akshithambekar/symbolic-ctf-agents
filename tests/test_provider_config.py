from maritime_nsrl.provider_config import load_env, api_key


def test_env_loading_and_provider_isolation(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.setenv('OPENAI_API_KEY', 'other-provider-test')
    monkeypatch.setenv('OPENROUTER_MODEL', 'explicit-model')
    path = tmp_path / '.env'
    path.write_text('OPENROUTER_API_KEY="router-test"\nOPENROUTER_MODEL=ignored\n', encoding='utf-8')
    assert api_key('https://openrouter.ai/api/v1') is None
    load_env(path)
    assert api_key('https://openrouter.ai/api/v1') == 'router-test'
    import os
    assert os.environ['OPENROUTER_MODEL'] == 'explicit-model'


def test_missing_model_does_not_create_run(tmp_path):
    import pytest
    from maritime_nsrl.translator_benchmark.runner import run
    with pytest.raises(ValueError, match='OPENROUTER_MODEL'):
        run(tmp_path / 'run')
    assert not (tmp_path / 'run').exists()
