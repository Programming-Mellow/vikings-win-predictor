"""Read the ingested Parquet files as pandas DataFrames."""

import pandas as pd

from vikpred.config import RAW_DIR


def _read(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m vikpred.data.ingest` first")
    return pd.read_parquet(path)


def schedules() -> pd.DataFrame:
    return _read("schedules")


def team_game_epa() -> pd.DataFrame:
    return _read("team_game_epa")


def injuries() -> pd.DataFrame:
    return _read("injuries")


def team_games(team: str) -> pd.DataFrame:
    """One row per game from `team`'s point of view (home or away)."""
    s = schedules()
    home = s[s["home_team"] == team].assign(
        is_home=True, opponent=lambda d: d["away_team"],
        points_for=lambda d: d["home_score"], points_against=lambda d: d["away_score"],
        win=lambda d: d["home_win"],
        # spread_line is the home team's expected margin, so it's already from our side.
        team_spread=lambda d: d["spread_line"],
    )
    away = s[s["away_team"] == team].assign(
        is_home=False, opponent=lambda d: d["home_team"],
        points_for=lambda d: d["away_score"], points_against=lambda d: d["home_score"],
        win=lambda d: 1 - d["home_win"],
        team_spread=lambda d: -d["spread_line"],
    )
    cols = ["game_id", "season", "game_type", "week", "gameday", "opponent", "is_home",
            "points_for", "points_against", "win", "team_spread", "total_line", "div_game"]
    games = pd.concat([home, away])[cols].sort_values(["season", "week"]).reset_index(drop=True)
    return games.assign(margin=games["points_for"] - games["points_against"])
