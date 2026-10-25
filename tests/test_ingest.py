"""Offline tests for the ingest transforms (no network calls)."""

import pandas as pd
import polars as pl
import pytest

from vikpred.data import ingest, load


def make_schedule(**overrides) -> pl.DataFrame:
    row = {c: None for c in ingest.SCHEDULE_COLUMNS}
    row.update(game_id="2015_01_SD_MIN", season=2015, game_type="REG", week=1,
               gameday="2015-09-13", away_team="SD", home_team="MIN",
               away_score=10, home_score=20, result=10, spread_line=3.0)
    row.update(overrides)
    return pl.DataFrame([row])


def test_normalize_teams_maps_relocated_franchises():
    df = pl.DataFrame({"home_team": ["OAK", "SD", "STL", "MIN"]})
    out = ingest.normalize_teams(df, ["home_team", "not_a_column"])
    assert out["home_team"].to_list() == ["LV", "LAC", "LA", "MIN"]


@pytest.mark.parametrize("result, expected", [(10, 1.0), (-3, 0.0), (0, 0.5), (None, None)])
def test_clean_schedules_home_win_label(result, expected):
    out = ingest.clean_schedules(make_schedule(result=result), [2015])
    assert out["home_win"][0] == expected
    assert out["away_team"][0] == "LAC"


def test_clean_schedules_filters_seasons():
    raw = pl.concat([make_schedule(), make_schedule(game_id="2009_x", season=2009)])
    assert ingest.clean_schedules(raw, [2015])["game_id"].to_list() == ["2015_01_SD_MIN"]


def test_aggregate_pbp_counts_only_pass_and_rush_plays():
    base = {"game_id": "g1", "season": 2024, "week": 1, "season_type": "REG",
            "posteam": "MIN", "defteam": "GB"}
    pbp = pl.DataFrame([
        {**base, "pass": 1, "rush": 0, "epa": 0.5, "success": 1},
        {**base, "pass": 1, "rush": 0, "epa": -0.5, "success": 0},
        {**base, "pass": 0, "rush": 1, "epa": 0.3, "success": 1},
        {**base, "pass": 0, "rush": 0, "epa": 2.0, "success": 1},   # punt: excluded
        {**base, "pass": 1, "rush": 0, "epa": None, "success": 0},  # no EPA: excluded
    ])
    row = ingest.aggregate_pbp(pbp).row(0, named=True)
    assert row["team"] == "MIN" and row["opponent"] == "GB"
    assert row["plays"] == 3
    assert row["pass_plays"] == 2
    assert row["epa_per_play"] == pytest.approx(0.1)
    assert row["pass_epa_per_play"] == pytest.approx(0.0)
    assert row["rush_epa_per_play"] == pytest.approx(0.3)
    assert row["success_rate"] == pytest.approx(2 / 3)


def test_team_games_flips_perspective_for_away_games(monkeypatch):
    sched = pd.DataFrame([
        {"game_id": "h", "season": 2024, "game_type": "REG", "week": 1, "gameday": None,
         "home_team": "MIN", "away_team": "GB", "home_score": 24, "away_score": 17,
         "home_win": 1.0, "spread_line": 3.0, "total_line": 45, "div_game": 1},
        {"game_id": "a", "season": 2024, "game_type": "REG", "week": 2, "gameday": None,
         "home_team": "DET", "away_team": "MIN", "home_score": 31, "away_score": 20,
         "home_win": 1.0, "spread_line": 6.5, "total_line": 50, "div_game": 1},
    ])
    monkeypatch.setattr(load, "schedules", lambda: sched)
    games = load.team_games("MIN").set_index("game_id")
    assert games.loc["h", ["opponent", "win", "team_spread", "margin"]].tolist() == ["GB", 1.0, 3.0, 7]
    assert games.loc["a", ["opponent", "win", "team_spread", "margin"]].tolist() == ["DET", 0.0, -6.5, -11]
