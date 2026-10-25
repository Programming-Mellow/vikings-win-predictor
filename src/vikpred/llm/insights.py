"""Short "why" bullets for a prediction, written by the smallest Claude model.

One stateless call per game: the model gets only the facts already shown on the
dashboard (both models' SHAP factors, probabilities, Vegas) and must explain
those, nothing else. Structured output guarantees a list of bullets back.
"""

import json
import os

import anthropic

from vikpred.llm.learner import LEVELS
from vikpred.teams import full_name

MAX_BULLETS = 4

SYSTEM_PROMPT = """You write the short "why" under an NFL win-probability banner on a dashboard.

You get one game's predictions as JSON: win probabilities from two XGBoost models (stats + Vegas line, and stats only), the Vegas probability, and each model's top SHAP factors. Factor "impact" is in log-odds from the named team's side (positive helped that team, negative hurt it); "value" is the team's edge over the opponent on that stat (for defense "allowed" stats, lower is better).

Spreads are from the named team's side: positive means it's favored by that many points.

Write 2 to 4 bullets explaining why the headline probability is what it is:
- Use only the factors and numbers provided. Never add stats, injuries, players, or news that aren't in the data.
- Lead with the biggest factor. Say plainly whether each one helps or hurts the team.
- If both models are available and they disagree by more than about 5 points, one bullet should say so and why. The stats-only model never sees Vegas data, so a gap usually means the betting line knows something the team stats don't, or the other way round.
- Round probabilities to whole percents (0.437 -> 44%) so they match the dashboard.
- Match the reader's level. Mention log-odds or raw impact numbers only for an Analyst; for others, describe size in words ("the biggest factor", "a small drag").
- Each bullet is one sentence, at most about 25 words. No betting advice."""

SCHEMA = {
    "type": "object",
    "properties": {"bullets": {"type": "array", "items": {"type": "string"}}},
    "required": ["bullets"],
    "additionalProperties": False,
}


def insights_model(provider: str) -> str:
    default = "anthropic.claude-haiku-5-5" if provider == "bedrock" else "claude-haiku-5-5"
    return os.environ.get("LLM_INSIGHTS_MODEL", default)


def game_facts(game: dict) -> dict:
    """The parts of a /teams/{team}/next payload the model needs, with full team names."""
    return {
        "team": full_name(game["team"]),
        "opponent": full_name(game["opponent"]),
        "venue": game["venue"],
        "week": game["week"],
        "vegas_win_prob": game["vegas_prob"],
        "models": [
            {"model": "stats + Vegas line" if p["model"] == "xgb_full" else "stats only",
             "available": p["available"], "win_prob": p["win_prob"], "reason_unavailable": p.get("reason"),
             "top_factors": [{"factor": f["label"], "value": f["value"], "impact": f["impact"]}
                             for f in p["top_factors"]]}
            for p in game["predictions"]
        ],
    }


def explain_game(client, model: str, game: dict, level: str = "fan") -> list[str]:
    """Return 2-4 plain-English bullets, or [] if the call fails or is declined."""
    level_name, level_desc = LEVELS.get(level, LEVELS["fan"])
    prompt = (f"Reader's level: {level_name}. {level_desc}\n\n"
              f"Game data:\n{json.dumps(game_facts(game), indent=1)}")
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,  # leaves room for thinking, which counts toward this cap
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        )
    except anthropic.APIError:
        return []
    if response.stop_reason in ("refusal", "max_tokens"):
        return []
    # A thinking block can come first, so find the answer by type.
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        bullets = json.loads(text)["bullets"]
    except (ValueError, KeyError, TypeError):
        return []
    return [b.strip() for b in bullets if isinstance(b, str) and b.strip()][:MAX_BULLETS]
