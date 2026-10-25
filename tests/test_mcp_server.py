"""MCP tools against the real FastAPI app with an in-memory prediction store."""

import asyncio

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from test_serving import MemoryStore, game_row
from vikpred.api import main as api
from vikpred.mcp_server import server


@pytest.fixture
def wired(monkeypatch):
    store = MemoryStore({2026: pd.DataFrame([
        game_row("2026_01_GB_MIN", 1, "MIN", "GB", home_win=1.0),
        game_row("2026_02_MIN_DET", 2, "DET", "MIN", full=np.nan, stats=0.55, market=np.nan),
    ])})
    api.app.dependency_overrides[api.get_store] = lambda: store
    monkeypatch.setattr(api, "current_season", lambda: 2026)
    monkeypatch.setattr(server, "_client", lambda: TestClient(api.app))
    yield
    api.app.dependency_overrides.clear()


def test_all_tools_registered():
    names = {t.name for t in asyncio.run(server.server.list_tools())}
    assert names == {"get_next_game", "get_game", "get_team_schedule", "get_team_record",
                     "get_team_news", "get_model_info", "get_feature_glossary"}


def test_next_game_tool(wired):
    game = server.get_next_game("MIN")
    assert game["game_id"] == "2026_02_MIN_DET" and game["venue"] == "away"
    assert game["predictions"][0]["available"] is False


def test_schedule_and_record_tools(wired):
    assert [g["week"] for g in server.get_team_schedule("MIN")] == [1, 2]
    assert server.get_team_record("MIN")["wins"] == 1


def test_api_errors_surface_as_tool_errors(wired):
    with pytest.raises(server.APIError, match="did not play"):
        server.get_game("2026_01_GB_MIN", team="DAL")


def test_unreachable_api_gives_actionable_message(monkeypatch):
    def refuse():
        raise httpx.ConnectError("refused")

    class Boom:
        def __enter__(self):
            refuse()

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(server, "_client", lambda: Boom())
    with pytest.raises(server.APIError, match="unreachable"):
        server.get_model_info()
