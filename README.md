# Vikings Win Predictor

Weekly win-probability predictions for the Minnesota Vikings, combining a traditional ML model with a generative-AI assistant that explains each prediction.

- **ML** produces the number: a calibrated win probability trained on every NFL game since 2010.
- **Generative AI** (Claude, via an MCP server) explains *why*, using only the model's actual feature contributions, and answers user questions. It never makes up the prediction.

> **Status:** Phase 6 of 7 complete (adaptive assistant, season tracker, Docker, CI). See the [model card](reports/MODEL_CARD.md) and the [roadmap](#roadmap).

[![CI](https://github.com/Programming-Mellow/vikings-win-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Programming-Mellow/vikings-win-predictor/actions/workflows/ci.yml)

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

**With Docker** (copy `.env.example` to `.env` and add your Anthropic key for the chat):

```bash
docker compose --profile bootstrap run --rm bootstrap   # first time: download data + train (~2 min)
docker compose up                                        # API :8000, app :8501
docker compose --profile job run --rm job               # weekly refresh
```

**Without Docker:**

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"

python -m vikpred.data.ingest     # downloads 2010-current season (~2 MB on disk)
python -m vikpred.features.build  # pre-game feature table
python -m vikpred.models.baseline # Phase 2 baselines, writes reports/phase2_*
python -m vikpred.models.xgb      # Phase 3 XGBoost + SHAP (~1 min), writes reports/phase3_*
python -m vikpred.serving.job       # weekly: refresh data and predict this season
uvicorn vikpred.api.main:app       # API at http://localhost:8000/docs
streamlit run app/streamlit_app.py # dashboard + chat at http://localhost:8501
pytest
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # browse experiment runs
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

## Phase 3 results ([full report](reports/phase3_report.md) · [model card](reports/MODEL_CARD.md))

XGBoost was tuned with **rolling-origin validation** (train on every season before 2019, 2020, … 2023, and predict that season), then tested once on 2024-2025:

| Model | Log loss ↓ | Brier ↓ | Accuracy | Calibration error ↓ |
|---|---|---|---|---|
| Vegas | 0.5981 | 0.2059 | 68.4% | 0.028 |
| LogReg, stats only (Phase 2) | 0.6281 | 0.2192 | 65.2% | 0.064 |
| **XGBoost, stats only** | 0.6258 | 0.2182 | 65.0% | 0.044 |
| **XGBoost, stats + Vegas** | **0.5980** | **0.2059** | 67.7% | 0.033 |

- XGBoost improves calibration error by about a third over logistic regression for the stats-only model. With the Vegas line it statistically ties the market (95% CI on the log-loss gap: −0.007 to +0.007).
- Platt and isotonic calibration were tested on held-out seasons and **didn't help**, so neither is applied.
- **Every prediction comes with SHAP explanations.** For the 2026 Vikings, the stats-only model's top positive factor each week is recent point differential, and the top negative factor is offensive success rate. That's the same "elite defense, struggling offense" pattern found in the EDA.

![SHAP summary](reports/phase3_shap_xgb_stats.png)

All runs are tracked in **MLflow** (parameters, CV and test metrics, charts).

## Phase 4: prediction API

A **weekly job** refreshes the data, rebuilds features, and stores every game's prediction from both models, along with the SHAP contributions. The **FastAPI service** only reads those stored results, so requests are instant and the models never load in the request path.

```bash
python -m vikpred.serving.job          # run weekly (e.g. Tuesday after Monday Night Football)
uvicorn vikpred.api.main:app --reload  # interactive docs at http://localhost:8000/docs
```

| Endpoint | Returns |
|---|---|
| `GET /teams/{team}/next` | Next unplayed game: both models' win probabilities, each with its own top SHAP factors |
| `GET /games/{game_id}?team=MIN` | Any game, explained from either team's side |
| `GET /teams/{team}/games?season=2026` | The season schedule with predictions, results, and scores |
| `GET /teams/{team}/record?live_only=true` | Win-loss record, plus how often the model and Vegas picked the winner, with Brier scores |
| `GET /model`, `/features`, `/health` | Model versions and test metrics, feature glossary, service status |

Example: `GET /teams/MIN/next` today returns IND @ MIN with **74.3%** (stats + Vegas) and **64.4%** (stats only). The stats-only model's top factors are *recent point-differential edge* (+0.57) and *offensive success-rate edge* (−0.14).

**Design choices:**
- **Prediction locking.** Once a game is played, its last pre-game prediction is frozen. Predictions made after kickoff are flagged `backfilled`, and `/record?live_only=true` excludes them, so tracked accuracy can't be inflated after the fact.
- **No guessing.** When Vegas lines aren't posted yet, the stats + Vegas model reports `available: false` with a reason instead of predicting from missing inputs.
- **Swappable storage.** The job and API use a `PredictionStore` interface (local Parquet today), so S3 or DynamoDB in Phase 7 is one new class.

## Phase 5: Claude assistant over MCP

```
Streamlit page ──HTTP──► FastAPI ◄──HTTP── MCP server (6 tools) ◄──stdio── MCP client ◄── Claude Sonnet 5.5
   dashboard                                                                      chat sidebar
```

- **MCP server** (`vikpred.mcp_server.server`) exposes the API as tools: `get_next_game`, `get_game`, `get_team_schedule`, `get_team_record`, `get_model_info`, `get_feature_glossary`. Any MCP client can use it, not just this app.
- **Chat agent** (`vikpred.llm.agent`): Claude Sonnet 5.5 with streamed replies, calling the MCP tools. Every number it quotes comes from a tool. It labels general NFL knowledge as such and declines betting advice. Each model's probability is explained with that model's own SHAP factors.
- **Streamlit app** (`app/streamlit_app.py`): the next-game card (stats + Vegas model, stats-only model, Vegas), a "why" chart per model, the season table with results and live-call flags, the model vs. Vegas record, and the chat in the sidebar.

**Setup:** copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY` (`.env` is git-ignored). Without a key the dashboard still works, and the chat explains how to turn it on.

```bash
uvicorn vikpred.api.main:app          # terminal 1
streamlit run app/streamlit_app.py    # terminal 2
```

**Use the tools from Claude Code or Claude Desktop** (with the API running):

```bash
claude mcp add vikings-predictor -- .venv/Scripts/python -m vikpred.mcp_server.server
```

For Claude Desktop, add to `claude_desktop_config.json`:

```json
{"mcpServers": {"vikings-predictor": {
  "command": "<repo>/.venv/Scripts/python.exe",
  "args": ["-m", "vikpred.mcp_server.server"],
  "env": {"VIKPRED_API_URL": "http://localhost:8000"}}}}
```

**Implementation notes:**
- Effort is `low`, the recommended chat setting, for fast, inexpensive replies. The system prompt and tool definitions are prompt-cached.
- Server-side refusal fallback (`fallbacks: "default"`) retries a wrongly declined harmless question on a fallback model.
- The conversation history is append-only, as Claude Sonnet 5.5 requires, so earlier turns are never edited.
- Tool inputs stream as they're generated and are validated against each tool's JSON schema before they run. A tool call cut off at the length limit is never run.
- `LLM_PROVIDER=bedrock` switches to Amazon Bedrock with no code changes (Phase 7).

## Phase 6: an assistant that grows with you, season tracker, Docker, CI

**Adaptive assistant.** Answers match your football knowledge and get more advanced as it grows:
- **Level selector** (Beginner / Fan / Analyst) in the sidebar sets the baseline depth.
- **Learned profile:** Claude records concepts you've shown you understand, concepts it has introduced, and your interests, through a local `update_learner_profile` tool. It's saved to `data/learner_profile.json` (git-ignored), and you can view or reset it from the sidebar.
- **Teaches as it goes:** a new term (EPA, success rate, SHAP, Brier score…) gets a one- or two-sentence explanation the first time. After that it's used freely, and answers sometimes end with a suggested next idea to explore.
- **Why the profile isn't in the system prompt:** Claude Sonnet 5.5 rejects requests whose system prompt changed mid-conversation, because its thinking blocks are tied to the exact history that produced them. So the profile is appended as a **mid-conversation system message** only when it changes, and the history stays append-only.
- **Progress notes:** Sonnet 5.5 returns text written between tool calls as "progress update" thinking blocks. The agent requests `display: "updates"` and shows them as status lines, and the prompt has Claude finish tool calls before it writes the answer.

**Chat UI.** The input is pinned at the top of the sidebar. The conversation scrolls in its own panel below, newest exchange first.

**Season tracker.** For each played game: the model's and Vegas's pre-game calls against the result, picks correct, and running Brier scores, with a *live predictions only* toggle that excludes backfilled calls.

**Docker.** One image (`Dockerfile`) runs every service through `docker-compose.yml`: `api`, `app`, plus on-demand `bootstrap` (first-time data + training) and `job` (weekly refresh). `data/`, `models/`, `reports/` and `mlruns/` are shared with the host. `VIKPRED_HOME` tells the installed package where the project lives.

**CI.** GitHub Actions (`.github/workflows/ci.yml`) runs the full offline test suite and builds and smoke-tests the Docker image on every PR and push to `main`.

## Project layout

```
src/vikpred/
  config.py         paths, seasons, team aliases
  data/ingest.py    nflverse → Parquet (CLI: python -m vikpred.data.ingest)
  data/load.py      pandas readers + team-perspective game view
  features/build.py pre-game feature table (CLI: python -m vikpred.features.build)
  models/baseline.py baselines + logistic regression, time-split evaluation
  models/metrics.py log loss, Brier, calibration error, bootstrap comparison
  models/xgb.py     XGBoost tuning, calibration, SHAP explanations, MLflow logging
  serving/job.py    weekly prediction job with locking (CLI: python -m vikpred.serving.job)
  serving/store.py  PredictionStore interface + local Parquet implementation
  api/main.py       FastAPI app (uvicorn vikpred.api.main:app)
  api/schemas.py    Pydantic response models
  mcp_server/server.py  MCP tools over the API (python -m vikpred.mcp_server.server)
  llm/agent.py      Claude chat loop with MCP tools
  llm/toolbox.py    sync MCP client for Streamlit
  llm/provider.py   Anthropic / Bedrock client config
  llm/learner.py    learner profile + update_learner_profile tool
  tracker.py        season tracker math (predictions vs. results, running Brier)
app/streamlit_app.py  dashboard + chat UI
notebooks/01_eda.ipynb
reports/            generated Phase 2/3 reports, charts, model card
tests/              offline unit tests (ingest, leakage, features, metrics, explanations, API, locking, MCP, agent, learner, tracker)
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Data ingest + EDA | ✅ |
| 2 | Pre-game feature table, logistic-regression baseline vs. Vegas/home baselines | ✅ |
| 3 | XGBoost + calibration + SHAP, MLflow tracking, model card | ✅ |
| 4 | FastAPI service + weekly prediction job | ✅ |
| 5 | MCP server + Claude chat assistant in Streamlit | ✅ |
| 6 | Adaptive assistant, pinned chat, season tracker, Docker, CI | ✅ |
| 7 | AWS deployment via CloudFormation (S3, EventBridge, ECS, Bedrock) | |
