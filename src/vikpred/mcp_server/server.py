"""MCP server exposing the prediction API as tools.

Any MCP client can use it: the Streamlit chat, Claude Desktop, or Claude Code.
It only talks to the FastAPI service over HTTP, so it stays lightweight and
never loads models or data itself.

Run (stdio):
    python -m vikpred.mcp_server.server

Env:
    VIKPRED_API_URL   base URL of the FastAPI service (default http://localhost:8000)
"""

import logging
import os

import httpx
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

API_URL = os.environ.get("VIKPRED_API_URL", "http://localhost:8000")

server = MCPServer(
    name="vikings-predictor",
    instructions=(
        "Pre-game NFL win probabilities from an XGBoost model, with SHAP explanations. "
        "Two models: xgb_full (team stats + Vegas line, most accurate) and xgb_stats "
        "(team stats only, an independent football view). Probabilities and factors are "
        "always from the requested team's side. Factor 'impact' is in log-odds: positive "
        "helps the team. Not betting advice."
    ),
)


class APIError(ToolError):
    """An anticipated failure: MCP passes this message to the model (other exceptions are masked)."""


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=10.0)


def _get(path: str, params: dict | None = None):
    params = {k: v for k, v in (params or {}).items() if v is not None}
    try:
        with _client() as client:
            resp = client.get(path, params=params)
    except httpx.ConnectError as exc:
        raise APIError(f"Prediction API unreachable at {API_URL}. Is `uvicorn vikpred.api.main:app` running?") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise APIError(f"{resp.status_code}: {detail}")
    return resp.json()


@server.tool()
def get_next_game(team: str = "MIN", top_k: int = 5) -> dict:
    """Get a team's next unplayed game: opponent, date, venue, the Vegas win probability,
    and both models' win probabilities, each with its top SHAP factors.

    Args:
        team: NFL team abbreviation, e.g. MIN, DET, KC. Defaults to the Vikings.
        top_k: How many top factors to return per model (1-20).
    """
    return _get(f"/teams/{team}/next", {"top_k": top_k})


@server.tool()
def get_game(game_id: str, team: str | None = None, top_k: int = 5) -> dict:
    """Get one game's predictions and explanations from a given team's side.
    Includes the result and score if it has been played.

    Args:
        game_id: nflverse game id like 2026_07_IND_MIN (season_week_AWAY_HOME).
            Use get_team_schedule to find ids.
        team: Which team's perspective to use. Defaults to the home team.
        top_k: How many top factors to return per model (1-20).
    """
    return _get(f"/games/{game_id}", {"team": team, "top_k": top_k})


@server.tool()
def get_team_schedule(team: str = "MIN", season: int | None = None) -> list:
    """List every game in a team's season with win probabilities (both models and Vegas),
    results, and scores. Use it to find game ids, past results, or upcoming opponents.

    Args:
        team: NFL team abbreviation. Defaults to the Vikings.
        season: Season year, e.g. 2026. Defaults to the current season.
    """
    return _get(f"/teams/{team}/games", {"season": season})


@server.tool()
def get_team_record(team: str = "MIN", season: int | None = None, live_only: bool = False) -> dict:
    """Get a team's win-loss record and how often the model and Vegas picked its games
    correctly (with Brier scores; lower is better).

    Args:
        team: NFL team abbreviation. Defaults to the Vikings.
        season: Season year. Defaults to the current season.
        live_only: If true, count only predictions made before kickoff (excludes backfilled ones).
    """
    return _get(f"/teams/{team}/record", {"season": season, "live_only": live_only})


@server.tool()
def get_model_info() -> dict:
    """Get model versions and their 2024-2025 test-set metrics (log loss, Brier, accuracy,
    calibration error) next to the Vegas and always-pick-home baselines."""
    return _get("/model")


@server.tool()
def get_feature_glossary() -> dict:
    """Get plain-English definitions of every feature the models use (the names that
    appear in top_factors)."""
    return _get("/features")


def main() -> None:
    # httpx logs every request at INFO, which floods the MCP client's stderr.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    server.run("stdio")


if __name__ == "__main__":
    main()
