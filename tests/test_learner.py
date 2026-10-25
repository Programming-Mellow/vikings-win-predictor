"""Learner profile and how the agent feeds it to Claude."""

from types import SimpleNamespace as NS

import pandas as pd
import pytest

from test_agent import TOOLS, make_agent, reply, text, tool_use
from vikpred.llm import learner
from vikpred.llm.learner import LearnerProfile, LearnerStore
from vikpred.llm.provider import LLMConfig
from vikpred.tracker import summary, tracker_frame


# --- profile -----------------------------------------------------------------------

def test_known_concepts_graduate_out_of_learning():
    p = LearnerProfile()
    p.apply({"learning": ["EPA", "success rate"]})
    p.apply({"known": ["epa"]})
    assert p.known == ["epa"] and p.learning == ["success rate"]
    # A known concept is never demoted back to learning.
    p.apply({"learning": ["EPA"]})
    assert p.learning == ["success rate"]


def test_updates_dedupe_normalize_and_cap():
    p = LearnerProfile()
    p.apply({"interests": [" Cover 2 ", "cover 2", ""]})
    assert p.interests == ["cover 2"]
    p.apply({"known": [f"c{i}" for i in range(learner.LIMITS["known"] + 5)]})
    assert len(p.known) == learner.LIMITS["known"] and p.known[-1] == "c64"


def test_context_message_reflects_level_and_lists():
    p = LearnerProfile(level="analyst", known=["epa"], interests=["defense"])
    msg = p.context_message()
    assert "Level: Analyst" in msg and "epa" in msg and "defense" in msg


def test_store_round_trip_and_reset(tmp_path):
    path = tmp_path / "profile.json"
    store = LearnerStore(path)
    store.set_level("beginner")
    assert store.handle_tool({"known": ["epa"], "note": "likes examples"}).startswith("Profile updated")
    reloaded = LearnerStore(path)
    assert reloaded.profile.level == "beginner" and reloaded.profile.known == ["epa"]
    reloaded.reset()
    assert LearnerStore(path).profile == LearnerProfile(level="beginner", updated_at=LearnerStore(path).profile.updated_at)
    with pytest.raises(ValueError):
        store.set_level("expert")


def test_store_survives_corrupt_file(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("{not json")
    assert LearnerStore(path).profile.level == learner.DEFAULT_LEVEL


# --- agent integration ----------------------------------------------------------------

def make_learning_agent(tmp_path, script, config=None):
    agent, api, calls = make_agent(script, config or LLMConfig("anthropic", "claude-sonnet-5-5", True, True))
    store = LearnerStore(tmp_path / "p.json")
    # Rebuild with the learner attached (make_agent doesn't take one).
    from vikpred.llm.agent import ChatAgent
    agent = ChatAgent(agent.client, agent.config, TOOLS, agent.call_tool, learner=store)
    return agent, api, store


def test_profile_tool_is_declared_from_the_first_request(tmp_path):
    agent, api, _ = make_learning_agent(tmp_path, [reply(text("hi"))])
    list(agent.respond("hello"))
    names = [t["name"] for t in api.requests[0]["tools"]]
    assert names == ["get_next_game", "update_learner_profile"]


def test_profile_sent_as_system_message_only_when_it_changes(tmp_path):
    agent, api, store = make_learning_agent(tmp_path, [
        reply(text("one")), reply(text("two")), reply(text("three"))])
    list(agent.respond("first"))
    list(agent.respond("second"))           # profile unchanged: no new system message
    store.set_level("analyst")
    list(agent.respond("third"))            # level changed: profile re-sent

    roles = [m["role"] for m in agent.history]
    assert roles == ["user", "system", "assistant", "user", "assistant", "user", "system", "assistant"]
    assert "Level: Fan" in agent.history[1]["content"]
    assert "Level: Analyst" in agent.history[6]["content"]
    # The top-level system prompt never changes.
    assert len({r["system"] for r in api.requests}) == 1


def test_profile_tool_updates_store_and_history_stays_append_only(tmp_path):
    agent, api, store = make_learning_agent(tmp_path, [
        reply(tool_use("t1", "update_learner_profile", {"known": ["epa"]}), stop="tool_use"),
        reply(text("Answer.")),
        reply(text("Next answer.")),
    ])
    list(agent.respond("I get EPA now"))
    assert store.profile.known == ["epa"]
    first = list(agent.history)
    list(agent.respond("next"))
    assert agent.history[: len(first)] == first
    # The changed profile goes out with the next user turn.
    assert agent.history[len(first) + 1]["role"] == "system"
    assert "epa" in agent.history[len(first) + 1]["content"]


def test_invalid_profile_update_is_rejected(tmp_path):
    agent, _, store = make_learning_agent(tmp_path, [
        reply(tool_use("t1", "update_learner_profile", {"known": "epa"}), stop="tool_use"),
        reply(text("ok")),
    ])
    list(agent.respond("hi"))
    assert store.profile.known == []


def test_provider_without_system_messages_gets_profile_in_user_turn(tmp_path):
    agent, _, _ = make_learning_agent(tmp_path, [reply(text("hi"))],
                                      LLMConfig("bedrock", "anthropic.claude-sonnet-5-5", False, False))
    list(agent.respond("hello"))
    assert [m["role"] for m in agent.history] == ["user", "assistant"]
    blocks = agent.history[0]["content"]
    assert blocks[0]["text"] == "hello" and "<app_context>" in blocks[1]["text"]


def test_progress_notes_are_streamed_as_progress_events(tmp_path):
    from test_agent import FakeStream
    stream = FakeStream([NS(type="thinking", thinking="Checking the schedule"), NS(type="thinking", thinking=""),
                         NS(type="text", text="Done.")],
                        NS(content=[text("Done.")], stop_reason="end_turn"))
    agent, _, _ = make_learning_agent(tmp_path, [stream])
    events = list(agent.respond("hi"))
    assert [(e.kind, e.text) for e in events] == [("progress", "Checking the schedule"), ("text", "Done.")]


# --- tracker -------------------------------------------------------------------------

def game(week, result, model, vegas, backfilled=False):
    return {"week": week, "opponent": "GB", "venue": "home", "result": result, "score": "1-0",
            "win_prob": model, "vegas_prob": vegas, "backfilled": backfilled}


def test_tracker_scores_played_games_only():
    games = [game(1, "W", 0.6, 0.7), game(2, "L", 0.6, 0.4, backfilled=True),
             game(3, None, 0.5, 0.5), game(4, "W", None, 0.6), game(5, "T", 0.5, 0.5)]
    frame = tracker_frame(games)
    assert frame.week.tolist() == [1, 2]
    assert frame.model_correct.tolist() == [True, False]
    assert frame.cum_model_brier.tolist() == pytest.approx([0.16, (0.16 + 0.36) / 2])
    s = summary(frame)
    assert (s["games"], s["model_correct"], s["vegas_correct"]) == (2, 1, 2)


def test_tracker_live_only_and_empty():
    games = [game(1, "W", 0.6, 0.7, backfilled=True)]
    assert tracker_frame(games, live_only=True).empty
    assert tracker_frame([]).empty
    assert summary(pd.DataFrame()) == {"games": 0}
