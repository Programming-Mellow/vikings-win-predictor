# Model Card: Vikings Win Predictor (Phase 3)

## Overview

| | |
|---|---|
| **Models** | `xgb_full` (primary) and `xgb_stats`: gradient-boosted trees (XGBoost 3.2) predicting the probability that the home team wins an NFL game |
| **Output** | Win probability from either team's side, plus SHAP factor contributions per game |
| **Training data** | Every NFL game 2010-2025 (regular season + playoffs) from [nflverse](https://github.com/nflverse); ties excluded |
| **Owner** | Built as a portfolio project; see the [README](../README.md) |

## Intended use

- Show a Vikings fan the model's pre-game win probability for each game and *why*, through the dashboard and chat assistant (Phases 4-5).
- Show end-to-end ML practice: leakage-safe features, time-aware validation, calibration, explainability, and experiment tracking.

**Not intended for betting.** The best model only ties the betting market (see below). It gives no edge, and nothing in it should be read as financial advice.

## Two variants

| Variant | Inputs | Use |
|---|---|---|
| `xgb_full` | Team form + context + **Vegas line** | Most accurate probability. Explanations are dominated by "what Vegas expects." |
| `xgb_stats` | Team form + context only, **no betting data** | An independent "football-only" view. Its explanations name the on-field reasons. |

## Features (all known before kickoff)

Home-minus-away differences in exponentially weighted (span 8 games, shifted one game) offensive/defensive EPA per play, success rate, pass/rush EPA, and point differential. Also: starting-QB change, QB ruled Out, key starters ruled Out, rest-day difference, division game, indoors, temperature, wind. `xgb_full` adds the Vegas spread, total, and moneyline win probability with the bookmaker margin removed. Full list with plain-English labels: `FEATURE_LABELS` in [`build.py`](../src/vikpred/features/build.py).

## Training and evaluation

- **Tuning:** 30 random hyperparameter trials, each scored by rolling-origin validation over 2019-2023 (train on all earlier seasons, predict the next).
- **Calibration:** none, Platt, and isotonic compared with leave-one-season-out on those predictions. **None won for both models**, because XGBoost's log-loss objective was already well calibrated.
- **Test:** 2024-2025 (569 games), untouched until final scoring.

| Model | Log loss | Brier | Accuracy | Calibration error |
|---|---|---|---|---|
| Vegas (reference) | 0.5981 | 0.2059 | 68.4% | 0.028 |
| `xgb_full` | 0.5980 | 0.2059 | 67.7% | 0.033 |
| `xgb_stats` | 0.6258 | 0.2182 | 65.0% | 0.044 |
| Always pick home | 0.6903 | 0.2486 | 54.1% | 0.017 |

`xgb_full` vs. Vegas log loss: −0.0001 (bootstrap 95% CI −0.007 to +0.007), so **statistically tied**. `xgb_stats` is significantly worse than Vegas (+0.028; CI +0.008 to +0.047) but far better than picking the home team. Full numbers are in [`phase3_report.md`](phase3_report.md).

## Explanations

SHAP values come from `shap.TreeExplainer`. They're in log-odds and add up exactly to the model's raw prediction (checked by a unit test). Because no calibration step is applied, they explain the final probability directly. For away games, every value and impact is flipped to the Vikings' side, and those flips are unit-tested.

## Limitations

- **Small, noisy data:** about 3,500 training games. Tuning picked depth-2 trees, so the model learns simple patterns only.
- **Early-season form leans on last year.** Rolling stats carry across seasons, so Week 1-3 numbers partly reflect the prior roster.
- **Coarse injury data:** counts of starters ruled Out don't weigh *who* is out (except the QB).
- **Correlated features.** EPA, success rate, point differential, and the Vegas line overlap, so credit can shift between them. Read SHAP factors as "what this model leaned on," not as cause and effect.
- **Market dependence:** `xgb_full` mostly mirrors the betting line. If lines are missing (more than about a week ahead), only `xgb_stats` can produce a prediction.
- **Changing league:** home-field advantage has shrunk since 2010. The model should be retrained each season.

## Reproduce

```bash
python -m vikpred.data.ingest
python -m vikpred.features.build
python -m vikpred.models.xgb        # ~1 min; logs to MLflow in ./mlruns
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```
