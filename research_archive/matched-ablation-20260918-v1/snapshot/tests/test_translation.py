import json
from unittest.mock import Mock, patch
import pytest
from maritime_nsrl.translation import translate, load_spec


def test_controlled_english_and_unknown_rejection():
    result = translate("Always stay within the arena. Pick up the enemy flag and then capture it.")
    assert result["ltl"] == "G(in_bounds) & F(has_flag & F(captured))"
    assert result["backend"] == "offline"
    with pytest.raises(ValueError): translate("Perform a feint and attack.")


def test_llm_validation_feedback():
    def response(text):
        r = Mock()
        r.json.return_value = {"choices": [{"message": {"content": text}}]}
        return r
    with patch("maritime_nsrl.translation.requests.post", side_effect=[
        response('{"ltl":"F(nonexistent)"}'), response('{"ltl":"F(captured)"}')]) as post:
        result = translate("Eventually capture the enemy flag.", "llm", "test-model", "http://localhost:1234/v1")
        assert post.call_count == 2
        assert not result["attempts"][0]["valid"] and result["attempts"][1]["valid"]


def test_saved_validation_flag_is_not_trusted(tmp_path):
    file = tmp_path / "spec.json"
    file.write_text(json.dumps({"ltl":"F(fake)", "validation":{"syntax_valid": True}}))
    with pytest.raises(ValueError): load_spec(file)

