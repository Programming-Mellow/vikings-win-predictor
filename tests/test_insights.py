"""Haiku "why" bullets, with a fake client (no network, no API key)."""

import json
from types import SimpleNamespace as NS

import anthropic
import httpx

from vikpred.llm import insights

GAME = {
    "game_id": "2026_08_MIN_DET", "team": "MIN", "opponent": "DET", "venue": "away", "week": 8,
    "vegas_prob": None, "win_prob": None, "win_prob_stats": 0.44, "predicted_at": "t",
    "predictions": [
        {"model": "xgb_full", "available": False, "win_prob": None, "reason": "Vegas lines not posted yet",
         "top_factors": []},
        {"model": "xgb_stats", "available": True, "win_prob": 0.44, "reason": None,
         "top_factors": [{"feature": "diff_point_diff_ewm", "label": "Recent point-differential edge",
                          "value": 3.7, "impact": 0.30}]},
    ],
}


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def client_returning(content, stop="end_turn"):
    return NS(messages=FakeMessages(NS(content=content, stop_reason=stop)))


def test_returns_bullets_and_reads_answer_by_block_type():
    payload = json.dumps({"bullets": ["Point differential helps.", " ", "Offense hurts.", "a", "b", "c"]})
    client = client_returning([NS(type="thinking", thinking=""), NS(type="text", text=payload)])
    bullets = insights.explain_game(client, "claude-haiku-5-5", GAME, "beginner")
    assert bullets == ["Point differential helps.", "Offense hurts.", "a", "b"]  # blanks dropped, capped at 4

    req = client.messages.calls[0]
    assert req["model"] == "claude-haiku-5-5"
    assert req["output_config"]["effort"] == "low"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in req  # sampling params are rejected on this model
    user = req["messages"][0]["content"]
    assert "Beginner" in user and "Minnesota Vikings" in user and "Detroit Lions" in user


def test_refusal_and_truncation_give_no_bullets():
    for stop in ("refusal", "max_tokens"):
        client = client_returning([NS(type="text", text='{"bullets": ["x"]}')], stop=stop)
        assert insights.explain_game(client, "m", GAME) == []


def test_api_error_and_bad_json_give_no_bullets():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.APIConnectionError(request=request)
    assert insights.explain_game(NS(messages=FakeMessages(error=err)), "m", GAME) == []
    assert insights.explain_game(client_returning([NS(type="text", text="not json")]), "m", GAME) == []


def test_facts_include_only_dashboard_data():
    facts = insights.game_facts(GAME)
    assert facts["team"] == "Minnesota Vikings" and facts["opponent"] == "Detroit Lions"
    assert facts["models"][0]["available"] is False
    assert facts["models"][1]["top_factors"][0] == {"factor": "Recent point-differential edge",
                                                    "value": 3.7, "impact": 0.30}


def test_model_defaults_and_override(monkeypatch):
    monkeypatch.delenv("LLM_INSIGHTS_MODEL", raising=False)
    assert insights.insights_model("anthropic") == "claude-haiku-5-5"
    assert insights.insights_model("bedrock") == "anthropic.claude-haiku-5-5"
    monkeypatch.setenv("LLM_INSIGHTS_MODEL", "claude-sonnet-5-5")
    assert insights.insights_model("anthropic") == "claude-sonnet-5-5"
