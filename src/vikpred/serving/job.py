"""Weekly prediction job: refresh data, rebuild features, predict the season.

Predictions are stored with their SHAP contributions so the API can explain
any game from either team's side without loading the models.

Locking: once a game has a result, the prediction from the last run *before*
it was played is kept as-is and never recomputed. Games that were already
played the first time the job saw them are predicted anyway but flagged
`backfilled`, since those predictions weren't made live.

Usage:
    python -m vikpred.serving.job                 # current season
    python -m vikpred.serving.job --season 2025   # any season (all backfilled)
    python -m vikpred.serving.job --skip-ingest   # don't re-download; still rebuilds features
"""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from vikpred.config import MODELS_DIR, current_season
from vikpred.data import ingest
from vikpred.features import build
from vikpred.models import baseline
from vikpred.models.xgb import WinModel
from vikpred.serving.store import LocalParquetStore, PredictionStore

log = logging.getLogger(__name__)

MODEL_NAMES = ["xgb_full", "xgb_stats"]
META_COLUMNS = ["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team",
                "home_score", "away_score", "home_win", "market_home_prob"]
FEATURE_COLUMNS = baseline.FULL_FEATURES  # superset of both models' inputs
PREDICTION_COLUMNS = [c for m in MODEL_NAMES for c in (f"{m}_home_prob", f"{m}_shap", f"{m}_base")]


def model_version(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_models() -> dict[str, tuple[WinModel, str]]:
    models = {}
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `python -m vikpred.models.xgb` first")
        models[name] = (joblib.load(path), model_version(path))
    return models


def _json_or_none(values: dict) -> str:
    return json.dumps({k: (None if pd.isna(v) else float(v)) for k, v in values.items()})


def predict_season(features: pd.DataFrame, models: dict[str, tuple[WinModel, str]]) -> pd.DataFrame:
    """One row per game: metadata, feature values, and each model's probability + SHAP."""
    out = features[META_COLUMNS].copy().reset_index(drop=True)
    out["features"] = [_json_or_none(r) for r in features[FEATURE_COLUMNS].to_dict("records")]
    for name, (model, version) in models.items():
        out[f"{name}_home_prob"] = np.nan
        out[f"{name}_shap"] = None
        out[f"{name}_base"] = np.nan
        out[f"{name}_version"] = version
        # XGBoost would silently accept NaNs, but the market model never saw a
        # missing line in training, so only predict when every input is present.
        ready = features[model.features].notna().all(axis=1).to_numpy()
        if not ready.any():
            continue
        rows = features.loc[ready]
        contribs, base = model.shap_values(rows)
        out.loc[ready, f"{name}_home_prob"] = model.predict_home_prob(rows)
        out.loc[ready, f"{name}_shap"] = [json.dumps(dict(zip(model.features, map(float, c)))) for c in contribs]
        out.loc[ready, f"{name}_base"] = base
    return out


def merge_with_existing(new: pd.DataFrame, existing: pd.DataFrame, now: str) -> pd.DataFrame:
    """Apply locking: keep stored pre-game predictions for games that have since been played."""
    new = new.copy()
    played = new["home_win"].notna()
    if existing.empty:
        new["predicted_at"] = now
        new["backfilled"] = played
        return new

    prev = existing.set_index("game_id")
    keep_cols = PREDICTION_COLUMNS + [c for c in prev.columns if c.endswith("_version")] + ["features", "predicted_at", "backfilled"]
    new["predicted_at"] = now
    new["backfilled"] = False
    for i, row in new.iterrows():
        gid = row.game_id
        if gid not in prev.index:
            new.at[i, "backfilled"] = bool(played[i])
            continue
        # Locked: the stored prediction was made before this game had a result.
        if played[i]:
            for c in keep_cols:
                new.at[i, c] = prev.at[gid, c]
    return new


def run(season: int | None = None, store: PredictionStore | None = None, skip_ingest: bool = False) -> pd.DataFrame:
    season = season or current_season()
    store = store or LocalParquetStore()
    if not skip_ingest:
        ingest.run()
    build.run()
    features = baseline.load_features()  # same transform the models were trained on
    features = features[features.season == season].reset_index(drop=True)
    preds = predict_season(features, load_models())
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    merged = merge_with_existing(preds, store.load(season), now)
    store.save(season, merged)
    log.info("season %d: %d games, %d with market predictions, %d locked/backfilled",
             season, len(merged), merged.xgb_full_home_prob.notna().sum(), merged.backfilled.sum())
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.season, skip_ingest=args.skip_ingest)


if __name__ == "__main__":
    main()
