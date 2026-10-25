# Vikings Win Predictor

Weekly win-probability predictions for the Minnesota Vikings, combining a traditional ML model with a generative-AI assistant that explains each prediction.

- **ML** produces the number: a calibrated win probability trained on every NFL game since 2010.
- **Generative AI** (Claude, via an MCP server) explains *why*, using only the model's actual feature contributions, and answers user questions. It never makes up the prediction.

> **Status:** Phase 2 of 7 complete (feature table + baseline models). See the [roadmap](#roadmap).

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
python -m vikpred.features.build  # pre-game feature table
python -m vikpred.models.baseline # train + evaluate, writes reports/phase2_*
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

## Phase 2 results ([full report](reports/phase2_baseline.md))

Every feature is computed from information available before kickoff. Rolling team stats are exponentially weighted over previous games and shifted one game; a unit test confirms a game's own stats can never feed its prediction.

Evaluated on the **2024-2025 seasons (569 games)** after training on 2010-2022 and tuning on 2023:

| Approach | Log loss ↓ | Brier ↓ | Accuracy |
|---|---|---|---|
| Always pick home | 0.690 | 0.249 | 54.1% |
| **Vegas (de-vigged moneyline)** | **0.598** | **0.206** | **68.4%** |
| Logistic regression, stats only | 0.628 | 0.219 | 65.2% |
| Logistic regression, stats + Vegas line | 0.598 | 0.206 | 68.4% |

- Stats alone beat the home-field baseline by a wide margin and get within ~3 points of Vegas accuracy without seeing any betting data.
- Adding the market line matches Vegas (bootstrap 95% CI on the log-loss gap: −0.005 to +0.006), but doesn't beat it. That's the expected result against an efficient market, and it sets the bar for Phase 3.
- The stats-only model is underconfident (calibration error 0.064 vs. Vegas 0.028); see the [calibration chart](reports/phase2_calibration.png).

## Project layout

```
src/vikpred/
  config.py         paths, seasons, team aliases
  data/ingest.py    nflverse → Parquet (CLI: python -m vikpred.data.ingest)
  data/load.py      pandas readers + team-perspective game view
  features/build.py pre-game feature table (CLI: python -m vikpred.features.build)
  models/baseline.py baselines + logistic regression, time-split evaluation
  models/metrics.py log loss, Brier, calibration error, bootstrap comparison
notebooks/01_eda.ipynb
reports/            generated Phase 2 report, metrics, calibration chart
tests/              offline unit tests (ingest, leakage, features, metrics)
```

## Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Data ingest + EDA | ✅ |
| 2 | Pre-game feature table, logistic-regression baseline vs. Vegas/home baselines | ✅ |
| 3 | XGBoost + calibration + SHAP, MLflow tracking, model card | |
| 4 | FastAPI service + weekly prediction job | |
| 5 | MCP server + Claude chat assistant in Streamlit | |
| 6 | Docker, CI, season tracker (predicted vs. actual) | |
| 7 | AWS deployment via CloudFormation (S3, EventBridge, ECS, Bedrock) | |
