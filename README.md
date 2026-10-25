# Vikings Win Predictor

Weekly win-probability predictions for the Minnesota Vikings, combining a traditional ML model with a generative-AI assistant that explains each prediction.

- **ML** produces the number: a calibrated win probability trained on every NFL game since 2010.
- **Generative AI** (Claude, via an MCP server) explains *why*, using only the model's actual feature contributions, and answers user questions. It never makes up the prediction.

> **Status:** Phase 1 of 7 complete (data ingest + EDA). See the [roadmap](#roadmap).

## Architecture

```
nflreadpy ──► ingest (polars) ──► data/raw/*.parquet
                                       │
                         features + model (LogReg → XGBoost) + SHAP      ← Phase 2-3
                                       │
                         weekly predictions ──► FastAPI                   ← Phase 4
                                       │
             MCP server (tools) ◄── Claude chat in Streamlit UI           ← Phase 5
```

Everything runs locally; Phase 7 maps each piece to AWS (S3, EventBridge, ECS/App Runner, Bedrock).

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"

python -m vikpred.data.ingest     # downloads 2010-current season (~2 MB on disk)
pytest
jupyter notebook notebooks/01_eda.ipynb
```

Re-run the ingest weekly. Finished seasons are cached; the current season is always refreshed. Use `--refresh` to re-download everything.

## Data

All data comes from [nflverse](https://github.com/nflverse) via [`nflreadpy`](https://github.com/nflverse/nflreadpy).

| File | Grain | Contents |
|---|---|---|
| `schedules.parquet` | game | scores, Vegas spread/total/moneyline, rest days, roof/surface/weather, starting QBs, `home_win` label |
| `team_game_epa.parquet` | team × game | offensive EPA/play and success rate, overall and split by pass/rush |
| `injuries.parquet` | player × week | official injury report status |

Relocated franchises are standardized to current codes (OAK→LV, SD→LAC, STL→LA) so every table joins cleanly.

## EDA highlights ([notebook](notebooks/01_eda.ipynb))

- The 2026 Vikings (5-0) rank **1st in defensive EPA/play** but **27th on offense**.
- Vegas favorites win **66.5%** of regular-season games; "always pick home" wins **55.3%**. These are the baselines to compare against.
- Home-field advantage has shrunk from ~57% to ~52-54%.
- Pre-game EPA predicts outcomes (r = 0.30 with margin), but is largely priced into the spread (r = 0.70).

## Project layout

```
src/vikpred/
  config.py         paths, seasons, team aliases
  data/ingest.py    nflverse → Parquet (CLI: python -m vikpred.data.ingest)
  data/load.py      pandas readers + team-perspective game view
notebooks/01_eda.ipynb
tests/              offline unit tests for ingest transforms
```

## Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Data ingest + EDA | ✅ |
| 2 | Pre-game feature table, logistic-regression baseline vs. Vegas/home baselines | |
| 3 | XGBoost + calibration + SHAP, MLflow tracking, model card | |
| 4 | FastAPI service + weekly prediction job | |
| 5 | MCP server + Claude chat assistant in Streamlit | |
| 6 | Docker, CI, season tracker (predicted vs. actual) | |
| 7 | AWS deployment via CloudFormation (S3, EventBridge, ECS, Bedrock) | |
