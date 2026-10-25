"""Chat agent loop tests with a scripted fake Claude client (no network, no API key)."""

from types import SimpleNamespace as NS

import pytest

from vikpred.llm.agent import ChatAgent, describe_tool_call
from vikpred.llm.provider import LLMConfig

TOOLS = [{
    "name": "get_next_game",
    "description": "Next game",
    "input_schema": {"type": "object", "properties": {"team": {"type": "string"}, "top_k": {"type": "integer"}}},
}]
ANTHROPIC = LLMConfig("anthropic", "claude-sonnet-5-5", True)


def text(t):
    return NS(type="text", text=t)


def tool_use(id_, name, input_):
    return NS(type="tool_use", id=id_, name=name, input=input_)


def reply(*blocks, stop="end_turn"):
    events = [NS(type="text", text=b.text) for b in blocks if b.type == "text"]
    return FakeStream(events, NS(content=list(blocks), stop_reason=stop))


class FakeStream:
    def __init__(self, events, message):
        self.events, self.message = events, message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.message


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def stream(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_agent(script, config=ANTHROPIC):
    messages = FakeMessages(script)
    calls = []

    def call_tool(name, args):
        calls.append((name, args))
        return '{"game_id": "2026_07_IND_MIN", "win_prob": 0.74}', False

    agent = ChatAgent(NS(beta=NS(messages=messages)), config, TOOLS, call_tool)
    return agent, messages, calls


def run(agent, prompt):
    return list(agent.respond(prompt))


def test_tool_loop_runs_tool_and_returns_answer():
    agent, api, calls = make_agent([
        reply(text("Let me check."), tool_use("t1", "get_next_game", {"team": "MIN"}), stop="tool_use"),
        reply(text("The model gives MIN 74%.")),
    ])
    events = run(agent, "Who wins this week?")

    assert calls == [("get_next_game", {"team": "MIN"})]
    assert [e.kind for e in events] == ["text", "tool_start", "tool_end", "text"]
    assert "".join(e.text for e in events if e.kind == "text") == "Let me check.The model gives MIN 74%."
    roles = [m["role"] for m in agent.history]
    assert roles == ["user", "assistant", "user", "assistant"]
    result = agent.history[2]["content"][0]
    assert result["tool_use_id"] == "t1" and "is_error" not in result


def test_request_shape():
    agent, api, _ = make_agent([reply(text("hi"))])
    run(agent, "hello")
    req = api.requests[0]
    assert req["model"] == "claude-sonnet-5-5"
    assert req["output_config"] == {"effort": "low"}
    assert req["fallbacks"] == "default"
    assert req["betas"] == ["thinking-display-updates-2026-08-18", "server-side-fallback-2026-07-01"]
    assert req["thinking"] == {"type": "adaptive", "display": "updates"}
    assert req["cache_control"] == {"type": "ephemeral"}
    assert all(t["eager_input_streaming"] for t in req["tools"])
    assert "tool_choice" not in req  # forced tool choice is a 400 on this model


def test_bedrock_config_skips_fallbacks():
    agent, api, _ = make_agent([reply(text("hi"))], LLMConfig("bedrock", "anthropic.claude-sonnet-5-5", False))
    run(agent, "hello")
    assert "fallbacks" not in api.requests[0]
    assert api.requests[0]["betas"] == ["thinking-display-updates-2026-08-18"]


def test_invalid_tool_input_is_rejected_without_calling_tool():
    agent, _, calls = make_agent([
        reply(tool_use("t1", "get_next_game", {"top_k": "five"}), stop="tool_use"),
        reply(text("Sorry, retrying differently.")),
    ])
    run(agent, "next game?")
    assert calls == []
    result = agent.history[2]["content"][0]
    assert result["is_error"] and "Invalid input" in result["content"]


def test_unknown_tool_name_returns_error_result():
    agent, _, calls = make_agent([
        reply(tool_use("t1", "get_weather", {}), stop="tool_use"),
        reply(text("ok")),
    ])
    run(agent, "weather?")
    assert calls == []
    assert agent.history[2]["content"][0]["is_error"]


def test_refusal_stops_without_running_tools():
    agent, _, calls = make_agent([reply(tool_use("t1", "get_next_game", {}), stop="refusal")])
    events = run(agent, "something")
    assert calls == []
    assert events[-1].kind == "notice"


def test_truncated_tool_call_is_never_run():
    agent, _, calls = make_agent([reply(tool_use("t1", "get_next_game", {"team": "M"}), stop="max_tokens")])
    events = run(agent, "next game?")
    assert calls == []
    assert agent.history[-1]["content"][0]["is_error"]
    assert events[-1].kind == "notice"


def test_unparseable_tool_json_is_retried():
    agent, api, _ = make_agent([ValueError("bad json"), reply(text("Recovered."))])
    events = run(agent, "hi")
    assert len(api.requests) == 2
    assert [e.text for e in events if e.kind == "text"] == ["Recovered."]


def test_history_is_append_only_across_turns():
    agent, api, _ = make_agent([
        reply(tool_use("t1", "get_next_game", {}), stop="tool_use"),
        reply(text("74%.")),
        reply(text("Because of defense.")),
    ])
    run(agent, "Who wins?")
    first_turn = list(agent.history)
    run(agent, "Why?")
    assert agent.history[: len(first_turn)] == first_turn
    # The second turn's request replayed the first turn unchanged, then added the new question.
    assert api.requests[2]["messages"][: len(first_turn)] == first_turn


@pytest.mark.parametrize("name, args, expected", [
    ("get_next_game", {"team": "DET"}, "Looking up DET's next game"),
    ("get_team_record", {}, "Checking MIN's record"),
])
def test_describe_tool_call(name, args, expected):
    assert describe_tool_call(name, args) == expected
