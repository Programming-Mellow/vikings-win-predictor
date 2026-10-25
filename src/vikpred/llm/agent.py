"""The chat assistant: Claude + the MCP prediction tools, streamed turn by turn.

The conversation history is append-only. Claude Sonnet 5.5 ties its thinking
blocks to the exact history that produced them, so earlier turns are never
edited; each assistant turn is stored with its full content, thinking included.
"""

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import anthropic
import jsonschema

from vikpred.llm import learner as learner_mod
from vikpred.llm.learner import LearnerStore
from vikpred.llm.provider import LLMConfig

SYSTEM_PROMPT = """You are the assistant inside the Vikings Win Predictor, a portfolio project that predicts NFL games with machine learning. The default team is the Minnesota Vikings (MIN), but any team works.

How the predictions work:
- Two XGBoost models trained on every NFL game since 2010. "xgb_full" uses team stats plus the Vegas line and is the most accurate (it statistically ties Vegas on 2024-2025 test games). "xgb_stats" uses team stats only, an independent football view that is less accurate but explains on-field reasons.
- Each prediction comes with SHAP factors. "impact" is in log-odds from the requested team's side: positive raised that team's chances, negative lowered them. Describe these in plain words ("their defense is the biggest boost", "a struggling offense pulls it down"); mention log-odds only if asked.
- Factor values are differences between the two teams (team minus opponent). For defense, lower EPA allowed is better.
- Predictions are frozen at kickoff. "backfilled" predictions were made after the game, so don't present them as live calls.
- The stats + Vegas model can't predict until Vegas lines are posted; then only the stats-only model is available.

Rules:
- Get every probability, factor, result, or record from the tools. Never estimate or make up a model number. Use the tools to check specifics such as dates, opponents, and results even when you feel confident.
- When both models are available, lead with xgb_full and mention xgb_stats as the football-only view, each with its own factors. Never explain one model's number with the other model's factors.
- You may also discuss general NFL topics (players, history, rules, strategy). When an answer comes from your general knowledge rather than a tool, say so briefly, e.g. "From general NFL knowledge, not the model:". Your general knowledge may be out of date for the current season.
- Don't give betting advice, picks against the spread, or wagering strategy. You can say what the model and Vegas think, but note it isn't betting advice.
- Keep answers short and conversational: a few sentences or a compact list. Round probabilities to whole percents.

Growing with the user:
- The app sends a learner profile as a system message: their level (Beginner, Fan, or Analyst), concepts they know, concepts they're still learning, and interests. It may change during the conversation; always follow the latest one.
- Match depth to the level. Explain a concept briefly (one or two sentences, with a football example) the first time it comes up if it isn't in their known list; after that, use it without re-explaining.
- When the user shows they understand a concept, explains it back, or asks a deeper follow-up, record it with update_learner_profile. Also record concepts you're about to explain (as learning) and recurring interests.
- Make any update_learner_profile call before you write your answer: in the same step as your data lookups, or as your first step if you need no data. Your answer must be the last thing you write in a turn, with no tool calls after it. Never mention the profile or that you recorded anything.
- Now and then, when it fits naturally, end with one short suggestion of a next idea to explore that builds on what they know (for example, from EPA to success rate, or from win probability to calibration). Skip it for quick factual answers."""

MAX_TOOL_ROUNDS = 8
MAX_TOKENS = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Sonnet 5.5 returns text written between tool calls as "progress update"
# thinking blocks, empty by default; this beta returns their text so the UI can show it.
UPDATES_BETA = "thinking-display-updates-2026-08-18"


@dataclass
class Event:
    """Something the UI can render while a turn streams."""
    kind: str  # "text" | "progress" | "tool_start" | "tool_end" | "notice" | "error"
    text: str = ""
    tool: str = ""
    data: dict | None = None


class ChatAgent:
    def __init__(self, client, config: LLMConfig, tools: list[dict],
                 call_tool: Callable[[str, dict], tuple[str, bool]], effort: str = "low",
                 learner: LearnerStore | None = None):
        self.client = client
        self.config = config
        self.call_tool = call_tool
        self.effort = effort
        self.learner = learner
        # The tool list must stay identical for the whole conversation, so the
        # local profile tool is included from the first request.
        tools = tools + ([learner_mod.TOOL] if learner else [])
        self._schemas = {t["name"]: t["input_schema"] for t in tools}
        # Tool inputs stream as they're generated; we validate them ourselves below.
        self.tools = [{**t, "eager_input_streaming": True} for t in tools]
        self.history: list[dict] = []
        self._sent_profile: str | None = None

    # -- model call ----------------------------------------------------------------

    def _request_kwargs(self) -> dict:
        kwargs = dict(
            model=self.config.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=self.tools,
            messages=self.history,
            output_config={"effort": self.effort},
            thinking={"type": "adaptive", "display": "updates"},
            cache_control={"type": "ephemeral"},  # system + tools are identical every turn
            betas=[UPDATES_BETA],
        )
        if self.config.supports_fallbacks:
            # On a safety-classifier decline, the API retries on a fallback model.
            kwargs["betas"].append(FALLBACK_BETA)
            kwargs["fallbacks"] = "default"
        return kwargs

    # -- tool execution --------------------------------------------------------------

    def _run_tool(self, block) -> dict:
        schema = self._schemas.get(block.name)
        if schema is None:
            return self._tool_error(block.id, f"Unknown tool '{block.name}'. Available: {sorted(self._schemas)}")
        try:
            # With eager streaming the API no longer validates inputs, so check them here.
            jsonschema.validate(block.input, schema)
        except jsonschema.ValidationError as exc:
            return self._tool_error(block.id, f"Invalid input for {block.name}: {exc.message}")
        if self.learner and block.name == learner_mod.TOOL["name"]:
            text, is_error = self.learner.handle_tool(block.input), False
        else:
            text, is_error = self.call_tool(block.name, block.input)
        result = {"type": "tool_result", "tool_use_id": block.id, "content": text}
        if is_error:
            result["is_error"] = True
        return result

    @staticmethod
    def _tool_error(tool_use_id: str, message: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": message, "is_error": True}

    # -- conversation turn -------------------------------------------------------------

    def _add_user_turn(self, user_text: str) -> None:
        """Append the user's message, plus the learner profile if it changed since it was last sent.

        The profile can't live in the top-level system prompt: changing that
        mid-conversation invalidates Claude's earlier thinking blocks. Instead it
        is appended as a mid-conversation system message right after the user's
        message, which keeps the history append-only.
        """
        profile = self.learner.profile.context_message() if self.learner else None
        if profile is None or profile == self._sent_profile:
            self.history.append({"role": "user", "content": user_text})
            return
        self._sent_profile = profile
        if self.config.supports_system_messages:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "system", "content": profile})
        else:
            # Providers without mid-conversation system messages get it as a
            # separate text block in the user turn instead.
            self.history.append({"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "text", "text": f"<app_context>\n{profile}\n</app_context>"},
            ]})

    def respond(self, user_text: str) -> Iterator[Event]:
        """Run one user turn to completion, yielding events as they happen."""
        self._add_user_turn(user_text)
        json_retries = 0

        for _ in range(MAX_TOOL_ROUNDS):
            try:
                with self.client.beta.messages.stream(**self._request_kwargs()) as stream:
                    for event in stream:
                        if event.type == "text":
                            yield Event("text", event.text)
                        elif event.type == "thinking" and event.thinking:
                            # Under display "updates", non-empty thinking is a progress note.
                            yield Event("progress", event.thinking)
                    message = stream.get_final_message()
            except ValueError:
                # The SDK couldn't parse a streamed tool input at all, so there's no
                # tool_use id to answer; re-issue the request (bounded).
                json_retries += 1
                if json_retries > 2:
                    yield Event("error", "The assistant produced a malformed tool call. Please try again.")
                    return
                continue
            except anthropic.AuthenticationError:
                yield Event("error", "The Anthropic API key was rejected. Check ANTHROPIC_API_KEY in your .env file.")
                return
            except anthropic.RateLimitError:
                yield Event("error", "Rate limited by the Anthropic API. Wait a moment and try again.")
                return
            except anthropic.APIStatusError as exc:
                yield Event("error", f"Anthropic API error ({exc.status_code}): {exc.message}")
                return
            except anthropic.APIConnectionError:
                yield Event("error", "Couldn't reach the Anthropic API. Check your internet connection.")
                return

            json_retries = 0
            self.history.append({"role": "assistant", "content": message.content})

            if message.stop_reason == "refusal":
                yield Event("notice", "The assistant declined to answer that. Try rephrasing the question.")
                return
            tool_uses = [b for b in message.content if b.type == "tool_use"]
            if message.stop_reason == "max_tokens":
                if tool_uses:
                    # A truncated tool input can parse as a valid-looking partial object; never run it.
                    self.history.append({"role": "user", "content": [
                        self._tool_error(b.id, "Tool input was cut off; not run.") for b in tool_uses]})
                yield Event("notice", "The answer hit its length limit and was cut short.")
                return
            if message.stop_reason == "pause_turn":
                continue
            if not tool_uses:
                return

            results = []
            for block in tool_uses:
                yield Event("tool_start", tool=block.name, data=dict(block.input) if isinstance(block.input, dict) else None)
                result = self._run_tool(block)
                yield Event("tool_end", tool=block.name, data={"is_error": result.get("is_error", False)})
                results.append(result)
            # All results go back together in one user message.
            self.history.append({"role": "user", "content": results})

        yield Event("notice", "Stopped after too many tool calls in one answer.")


def describe_tool_call(name: str, args: dict | None) -> str:
    """Short status line for the UI, e.g. 'Looking up MIN's next game'."""
    args = args or {}
    team = args.get("team") or "MIN"
    labels = {
        "get_next_game": f"Looking up {team}'s next game",
        "get_game": f"Opening game {args.get('game_id', '')}",
        "get_team_schedule": f"Pulling {team}'s schedule",
        "get_team_record": f"Checking {team}'s record",
        "get_model_info": "Reading model test results",
        "get_feature_glossary": "Reading the feature glossary",
        "update_learner_profile": "Noting what you know",
    }
    return labels.get(name, f"Calling {name} {json.dumps(args)}")
