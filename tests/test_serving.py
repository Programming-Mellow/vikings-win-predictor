"""Weekly job locking rules and API behavior, using an in-memory store."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vikpred.api import main as api
from vikpred.serving import job


class MemoryStore:
    def __init__(self, frames: dict[int, pd.DataFrame] | None = None):
        self.frames = frames or {}

    def load(self, season):
        return self.frames.get(season, pd.DataFrame()).copy()

    def save(self, season, df):
        self.frames[season] = df.copy()

    def seasons(self):
        return sorted(self.frames)


def game_row(game_id, week, home, away, home_win=np.nan, full=0.7, stats=0.6, market=0.65, **extra):
    row = {
        "game_id": game_id, "season": 2026, "week": week, "game_type": "REG",
        "gameday": pd.Timestamp("2026-09-06") + pd.Timedelta(weeks=week),
        "home_team": home, "away_team": away,
        "home_score": np.nan if pd.isna(home_win) else (24.0 if home_win == 1 else 17.0),
        "away_score": np.nan if pd.isna(home_win) else (17.0 if home_win == 1 else 24.0),
        "home_win": home_win, "market_home_prob": market,
        "features": json.dumps({"diff_point_diff_ewm": 5.0, "temp": 70.0, "market_logit": None}),
        "xgb_full_home_prob": full, "xgb_full_base": 0.1, "xgb_full_version": "fullv1",
        "xgb_full_shap": None if pd.isna(full) else json.dumps({"diff_point_diff_ewm": 0.4, "temp": -0.05}),
        "xgb_stats_home_prob": stats, "xgb_stats_base": 0.1, "xgb_stats_version": "statsv1",
        "xgb_stats_shap": json.dumps({"diff_point_diff_ewm": 0.3, "temp": 0.02}),
        "predicted_at": "2026-10-01T00:00:00+00:00", "backfilled": False,
    }
    row.update(extra)
    return row


# --- Locking -----------------------------------------------------------------

def test_first_run_flags_played_games_as_backfilled():
    new = pd.DataFrame([game_row("g1", 1, "MIN", "GB", home_win=1.0), game_row("g2", 2, "MIN", "CHI")])
    out = job.merge_with_existing(new, pd.DataFrame(), "now")
    assert out.backfilled.tolist() == [True, False]


def test_played_game_keeps_its_pre_game_prediction():
    before = pd.DataFrame([game_row("g2", 2, "MIN", "CHI", full=0.70, predicted_at="week1")])
    # Next week: the game was played and the model would now say something else.
    after = pd.DataFrame([game_row("g2", 2, "MIN", "CHI", home_win=0.0, full=0.20)])
    out = job.merge_with_existing(after, before, "week2").iloc[0]
    assert out.xgb_full_home_prob == pytest.approx(0.70)
    assert out.predicted_at == "week1"
    assert out.home_win == 0.0  # the result itself is updated
    assert not out.backfilled


def test_unplayed_game_is_recomputed():
    before = pd.DataFrame([game_row("g3", 3, "MIN", "DET", full=np.nan, stats=0.5, predicted_at="week1")])
    after = pd.DataFrame([game_row("g3", 3, "MIN", "DET", full=0.55, stats=0.52)])
    out = job.merge_with_existing(after, before, "week2").iloc[0]
    assert out.xgb_full_home_prob == pytest.approx(0.55)
    assert out.predicted_at == "week2"


# --- API -----------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    season = pd.DataFrame([
        game_row("2026_01_GB_MIN", 1, "MIN", "GB", home_win=1.0, full=0.6, market=0.55),
        game_row("2026_02_MIN_CHI", 2, "CHI", "MIN", home_win=1.0, full=0.4, market=0.45),
        game_row("2026_03_DET_MIN", 3, "MIN", "DET", full=np.nan, stats=0.58, market=np.nan),
        game_row("2026_04_MIN_GB", 4, "GB", "MIN", full=0.30, stats=0.35, market=0.32),
    ])
    store = MemoryStore({2026: season})
    api.app.dependency_overrides[api.get_store] = lambda: store
    monkeypatch.setattr(api, "current_season", lambda: 2026)
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_next_game_is_first_unplayed_and_reports_missing_market(client):
    r = client.get("/teams/min/next")
    assert r.status_code == 200
    body = r.json()
    assert body["game_id"] == "2026_03_DET_MIN"
    full, stats = body["predictions"]
    assert full["available"] is False and full["reason"] == "Vegas lines not posted yet"
    assert stats["win_prob"] == pytest.approx(0.58)


def test_away_game_flips_probability_and_impacts(client):
    body = client.get("/games/2026_04_MIN_GB", params={"team": "MIN"}).json()
    assert body["venue"] == "away" and body["opponent"] == "GB"
    assert body["win_prob"] == pytest.approx(0.70)
    assert body["vegas_prob"] == pytest.approx(0.68)
    factors = {f["feature"]: f for f in body["predictions"][0]["top_factors"]}
    assert factors["diff_point_diff_ewm"]["impact"] == pytest.approx(-0.4)
    assert factors["diff_point_diff_ewm"]["value"] == pytest.approx(-5.0)


def test_record_counts_results_and_picks(client):
    body = client.get("/teams/MIN/record").json()
    assert (body["wins"], body["losses"]) == (1, 1)
    assert body["games_scored"] == 2
    # Week 1: MIN 60% and won (correct). Week 2: MIN 60% and lost (wrong).
    assert body["model_correct"] == 1
    assert body["model_brier"] == pytest.approx((0.4**2 + 0.6**2) / 2)


def test_errors(client):
    assert client.get("/teams/XYZ/games").status_code == 404
    assert client.get("/games/2026_01_GB_MIN", params={"team": "DAL"}).status_code == 422
    assert client.get("/games/nope").status_code == 422
    assert client.get("/teams/MIN/games", params={"season": 2019}).status_code == 404


def test_health_and_games_list(client):
    assert client.get("/health").json()["seasons_available"] == [2026]
    games = client.get("/teams/MIN/games").json()
    assert [g["week"] for g in games] == [1, 2, 3, 4]
    assert games[0]["result"] == "W" and games[0]["score"] == "24-17"
    assert games[1]["result"] == "L" and games[1]["score"] == "17-24"
