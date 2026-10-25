"""Response models. These double as the API's OpenAPI documentation."""

from pydantic import BaseModel, Field


class Factor(BaseModel):
    feature: str = Field(description="Model feature name")
    label: str = Field(description="Plain-English description")
    value: float | None = Field(description="Feature value from the requested team's side")
    impact: float = Field(description="SHAP contribution in log-odds; positive helps the requested team")


class ModelPrediction(BaseModel):
    model: str = Field(description="xgb_full (stats + Vegas line) or xgb_stats (stats only)")
    win_prob: float | None = Field(description="Requested team's win probability; null if inputs are missing")
    available: bool
    reason: str | None = Field(None, description="Why no prediction is available, if applicable")
    top_factors: list[Factor] = []
    version: str


class GameSummary(BaseModel):
    game_id: str
    season: int
    week: int
    gameday: str
    team: str
    opponent: str
    venue: str
    vegas_prob: float | None
    win_prob: float | None = Field(description="xgb_full probability (stats + Vegas)")
    win_prob_stats: float | None = Field(description="xgb_stats probability (stats only)")
    result: str | None = Field(description="W, L, T, or null if not played")
    score: str | None
    backfilled: bool = Field(description="True if the prediction was made after the game was played")


class GameDetail(GameSummary):
    predictions: list[ModelPrediction]
    predicted_at: str


class Record(BaseModel):
    team: str
    season: int
    wins: int
    losses: int
    ties: int
    games_scored: int = Field(description="Played games with a stats + Vegas prediction")
    model_correct: int
    model_brier: float | None
    vegas_correct: int
    vegas_brier: float | None
    live_only: bool = Field(description="If true, backfilled predictions are excluded")


class ModelInfo(BaseModel):
    models: dict[str, str] = Field(description="Model name to version hash")
    test_metrics: dict[str, dict[str, float]] = Field(description="2024-2025 test results from Phase 3")
    seasons_available: list[int]


class Health(BaseModel):
    status: str
    seasons_available: list[int]
    latest_prediction_run: str | None


class NewsFact(BaseModel):
    id: str
    kind: str = Field(description="Injury, Roster move, Trade, or Coaching")
    text: str
    in_model: bool = Field(description="True if the models' features already account for this (key starters ruled Out)")
    date: str | None = None


class NewsHeadline(BaseModel):
    title: str = Field(description="Exactly as published")
    link: str
    source: str
    published: str | None
    category: str | None = Field(description="Keyword category (not AI): Injury, Roster move, Trade, Coaching, Suspension")


class NewsSource(BaseModel):
    name: str
    url: str
    terms: str


class TeamNews(BaseModel):
    team: str
    season: int
    generated_at: str
    facts: list[NewsFact] = Field(description="From nflverse: this week's injury report, roster moves, trades, coach changes")
    headlines: list[NewsHeadline] = Field(description="Last 7 days from the allow-listed feeds, newest first")
    sources: list[NewsSource]
