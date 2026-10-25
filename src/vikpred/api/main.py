"""Read-only prediction API. All heavy work happens in the weekly job.

Run locally:
    uvicorn vikpred.api.main:app --reload
    open http://localhost:8000/docs
"""

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query

from vikpred.config import REPORTS_DIR, TEAM, current_season
from vikpred.features.build import FEATURE_LABELS
from vikpred.models.xgb import explain_for_team
from vikpred.news.collect import load_team_news
from vikpred.teams import TEAM_NAMES
from vikpred.serving.job import MODEL_NAMES
from vikpred.serving.store import LocalParquetStore, PredictionStore

from vikpred.api.schemas import (  # noqa: I001
    Factor, GameDetail, GameSummary, Health, ModelInfo, ModelPrediction, Record, TeamNews,
)

app = FastAPI(
    title="Vikings Win Predictor API",
    description="Pre-game NFL win probabilities with SHAP explanations. Not betting advice.",
    version="0.4.0",
)


@lru_cache
def _default_store() -> LocalParquetStore:
    return LocalParquetStore()


def get_store() -> PredictionStore:
    return _default_store()


def _season_frame(store: PredictionStore, season: int) -> pd.DataFrame:
    df = store.load(season)
    if df.empty:
        raise HTTPException(404, f"No predictions for {season}. Run `python -m vikpred.serving.job --season {season}`.")
    return df


def _team_games(df: pd.DataFrame, team: str) -> pd.DataFrame:
    games = df[(df.home_team == team) | (df.away_team == team)]
    if games.empty:
        raise HTTPException(404, f"Unknown team '{team}' or no games found")
    return games.sort_values(["gameday", "game_id"])


def _for_team(home_prob: float, is_home: bool) -> float | None:
    if pd.isna(home_prob):
        return None
    return round(float(home_prob if is_home else 1 - home_prob), 4)


def _result(row: pd.Series, team: str) -> tuple[str | None, str | None]:
    if pd.isna(row.home_win):
        return None, None
    is_home = row.home_team == team
    ours, theirs = (row.home_score, row.away_score) if is_home else (row.away_score, row.home_score)
    result = "T" if ours == theirs else ("W" if ours > theirs else "L")
    return result, f"{int(ours)}-{int(theirs)}"


def _summary(row: pd.Series, team: str) -> dict:
    is_home = row.home_team == team
    result, score = _result(row, team)
    return {
        "game_id": row.game_id, "season": int(row.season), "week": int(row.week),
        "gameday": str(row.gameday)[:10], "team": team,
        "opponent": row.away_team if is_home else row.home_team,
        "venue": "home" if is_home else "away",
        "vegas_prob": _for_team(row.market_home_prob, is_home),
        "win_prob": _for_team(row.xgb_full_home_prob, is_home),
        "win_prob_stats": _for_team(row.xgb_stats_home_prob, is_home),
        "result": result, "score": score, "backfilled": bool(row.backfilled),
    }


def _model_prediction(row: pd.Series, name: str, team: str, top_k: int) -> ModelPrediction:
    version = str(row[f"{name}_version"])
    if pd.isna(row[f"{name}_home_prob"]):
        reason = "Vegas lines not posted yet" if name == "xgb_full" else "Not enough team history"
        return ModelPrediction(model=name, win_prob=None, available=False, reason=reason, version=version)

    contribs = json.loads(row[f"{name}_shap"])
    feature_values = {k: (np.nan if v is None else v) for k, v in json.loads(row.features).items()}
    features = list(contribs)
    game = pd.Series({**row.to_dict(), **feature_values})
    explained = explain_for_team(game, np.array([contribs[f] for f in features]), features, team,
                                 float(row[f"{name}_home_prob"]), top_k=top_k)
    return ModelPrediction(model=name, win_prob=explained["win_prob"], available=True,
                           top_factors=[Factor(**f) for f in explained["top_factors"]], version=version)


def _detail(row: pd.Series, team: str, top_k: int) -> GameDetail:
    return GameDetail(**_summary(row, team), predicted_at=str(row.predicted_at),
                      predictions=[_model_prediction(row, m, team, top_k) for m in MODEL_NAMES])


@app.get("/health", response_model=Health)
def health(store: PredictionStore = Depends(get_store)):
    seasons = store.seasons()
    latest = None
    if seasons:
        df = store.load(seasons[-1])
        latest = None if df.empty else str(df.predicted_at.max())
    return Health(status="ok", seasons_available=seasons, latest_prediction_run=latest)


@app.get("/model", response_model=ModelInfo)
def model_info(store: PredictionStore = Depends(get_store)):
    seasons = store.seasons()
    versions = {}
    if seasons:
        df = store.load(seasons[-1])
        versions = {m: str(df[f"{m}_version"].iloc[-1]) for m in MODEL_NAMES if f"{m}_version" in df}
    metrics_path = REPORTS_DIR / "phase3_metrics.json"
    metrics = json.loads(metrics_path.read_text())["test_metrics"] if metrics_path.exists() else {}
    return ModelInfo(models=versions, test_metrics=metrics, seasons_available=seasons)


@app.get("/teams/{team}/games", response_model=list[GameSummary])
def team_games(team: str, season: int | None = None, store: PredictionStore = Depends(get_store)):
    """Every game for a team in a season, with both models' win probabilities."""
    team = team.upper()
    df = _season_frame(store, season or current_season())
    return [_summary(row, team) for _, row in _team_games(df, team).iterrows()]


@app.get("/teams/{team}/next", response_model=GameDetail)
def next_game(team: str = TEAM, top_k: int = Query(5, ge=1, le=20),
              store: PredictionStore = Depends(get_store)):
    """The team's next unplayed game, with explanations from both models."""
    team = team.upper()
    games = _team_games(_season_frame(store, current_season()), team)
    upcoming = games[games.home_win.isna()]
    if upcoming.empty:
        raise HTTPException(404, f"{team} has no remaining games this season")
    return _detail(upcoming.iloc[0], team, top_k)


@app.get("/games/{game_id}", response_model=GameDetail)
def game(game_id: str, team: str | None = None, top_k: int = Query(5, ge=1, le=20),
         store: PredictionStore = Depends(get_store)):
    """One game, explained from `team`'s side (defaults to the home team)."""
    try:
        season = int(game_id[:4])
    except ValueError:
        raise HTTPException(422, "game_id should look like 2026_07_IND_MIN")
    df = _season_frame(store, season)
    match = df[df.game_id == game_id]
    if match.empty:
        raise HTTPException(404, f"Game '{game_id}' not found")
    row = match.iloc[0]
    team = (team or row.home_team).upper()
    if team not in (row.home_team, row.away_team):
        raise HTTPException(422, f"{team} did not play in {game_id}")
    return _detail(row, team, top_k)


@app.get("/teams/{team}/record", response_model=Record)
def record(team: str, season: int | None = None, live_only: bool = False,
           store: PredictionStore = Depends(get_store)):
    """Win-loss record, and how the model and Vegas did picking this team's games."""
    team = team.upper()
    season = season or current_season()
    games = _team_games(_season_frame(store, season), team)
    played = games[games.home_win.notna()]
    results = [_result(r, team)[0] for _, r in played.iterrows()]

    scored = played[played.xgb_full_home_prob.notna() & (played.home_win != 0.5)]
    if live_only:
        scored = scored[~scored.backfilled.astype(bool)]
    is_home = (scored.home_team == team).to_numpy()
    won = np.where(is_home, scored.home_win, 1 - scored.home_win)

    def evaluate(home_probs: pd.Series) -> tuple[int, float | None]:
        if scored.empty:
            return 0, None
        p = np.where(is_home, home_probs, 1 - home_probs)
        return int(((p > 0.5) == (won == 1)).sum()), round(float(np.mean((p - won) ** 2)), 4)

    model_correct, model_brier = evaluate(scored.xgb_full_home_prob)
    vegas_correct, vegas_brier = evaluate(scored.market_home_prob)
    return Record(team=team, season=season, wins=results.count("W"), losses=results.count("L"),
                  ties=results.count("T"), games_scored=len(scored),
                  model_correct=model_correct, model_brier=model_brier,
                  vegas_correct=vegas_correct, vegas_brier=vegas_brier, live_only=live_only)


@app.get("/features", response_model=dict[str, str])
def features():
    """Plain-English meaning of every feature the models use."""
    return {k: label for k, (label, _) in FEATURE_LABELS.items()}


@app.get("/teams/{team}/news", response_model=TeamNews)
def team_news(team: str):
    """This week's facts (nflverse) and headlines (allow-listed feeds) for one team. Refreshed daily."""
    team = team.upper()
    if team not in TEAM_NAMES:
        raise HTTPException(404, f"Unknown team '{team}'")
    news = load_team_news(team)
    if news is None:
        raise HTTPException(404, "No news collected yet. Run `python -m vikpred.news.collect` or the daily job.")
    return news
