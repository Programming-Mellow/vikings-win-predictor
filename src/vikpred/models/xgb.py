"""Phase 3: XGBoost with time-aware tuning, calibration, and SHAP explanations.

Tuning uses rolling-origin validation: for each season in CV_SEASONS, train on
every earlier season and score that one. The same out-of-fold predictions are
used to pick and fit a probability calibrator. 2024-2025 stay untouched until
the final test, exactly as in Phase 2.

Usage:
    python -m vikpred.models.xgb     # writes reports/phase3_*, models/xgb_*.joblib, MLflow runs
"""

import json
import logging
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import matplotlib  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vikpred.config import MLRUNS_DIR, MODELS_DIR, REPORTS_DIR, TEAM  # noqa: E402
from vikpred.features.build import FEATURE_LABELS  # noqa: E402
from vikpred.models import baseline  # noqa: E402
from vikpred.models.metrics import bootstrap_log_loss_diff, score  # noqa: E402

log = logging.getLogger(__name__)

CV_SEASONS = [2019, 2020, 2021, 2022, 2023]
TEST = baseline.TEST
LIVE = baseline.LIVE
VARIANTS = {"xgb_stats": baseline.STATS_FEATURES, "xgb_full": baseline.FULL_FEATURES}

# Shallow trees and strong regularization: ~3,500 noisy games won't support much depth.
PARAM_SPACE = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.02, 0.05],
    "n_estimators": [150, 300, 600],
    "min_child_weight": [5, 10, 20, 40],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "reg_lambda": [1.0, 5.0, 20.0],
}
N_TRIALS = 30
CALIBRATION_METHODS = ["none", "platt", "isotonic"]


def make_xgb(params: dict) -> XGBClassifier:
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                         n_jobs=-1, random_state=0, **params)


def sample_params(rng: np.random.Generator) -> dict:
    return {k: v[rng.integers(len(v))] for k, v in PARAM_SPACE.items()}


def rolling_oof(data: pd.DataFrame, features: list[str], params: dict, seasons: list[int]) -> pd.DataFrame:
    """Out-of-fold predictions: each season predicted by a model trained only on earlier seasons."""
    folds = []
    for season in seasons:
        train, valid = data[data.season < season], data[data.season == season]
        model = make_xgb(params).fit(train[features], train.home_win)
        folds.append(valid[["game_id", "season", "home_win"]].assign(p_raw=model.predict_proba(valid[features])[:, 1]))
    return pd.concat(folds, ignore_index=True)


def tune(data: pd.DataFrame, features: list[str], seed: int = 0) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    trials = []
    for i in range(N_TRIALS):
        params = sample_params(rng)
        oof = rolling_oof(data, features, params, CV_SEASONS)
        trials.append({**params, "cv_log_loss": score(oof.home_win, oof.p_raw)["log_loss"]})
        log.debug("trial %d: %.4f", i, trials[-1]["cv_log_loss"])
    trials = pd.DataFrame(trials).sort_values("cv_log_loss").reset_index(drop=True)
    best = {k: trials.loc[0, k] for k in PARAM_SPACE}
    best = {k: (int(v) if k in ("max_depth", "n_estimators", "min_child_weight") else float(v)) for k, v in best.items()}
    return best, trials


class Calibrator:
    """Monotone map from raw model probability to calibrated probability."""

    def __init__(self, method: str):
        self.method = method
        self._fn = None

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y) -> "Calibrator":
        if self.method == "platt":
            self._fn = LogisticRegression(C=1e6).fit(self._logit(p), y)
        elif self.method == "isotonic":
            self._fn = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(p, y)
        return self

    def transform(self, p) -> np.ndarray:
        if self.method == "platt":
            return self._fn.predict_proba(self._logit(p))[:, 1]
        if self.method == "isotonic":
            return self._fn.predict(np.asarray(p, dtype=float))
        return np.asarray(p, dtype=float)


def choose_calibration(oof: pd.DataFrame) -> tuple[str, dict[str, float]]:
    """Leave-one-season-out over the OOF predictions: fit on 4 seasons, score the 5th."""
    results = {}
    for method in CALIBRATION_METHODS:
        losses = []
        for season in oof.season.unique():
            fit, held = oof[oof.season != season], oof[oof.season == season]
            cal = Calibrator(method).fit(fit.p_raw, fit.home_win)
            losses.append(score(held.home_win, cal.transform(held.p_raw))["log_loss"] * len(held))
        results[method] = sum(losses) / len(oof)
    return min(results, key=results.get), results


class WinModel:
    """XGBoost + calibrator, with SHAP explanations. This is what gets saved and served."""

    def __init__(self, name: str, features: list[str], params: dict, calibration: str):
        self.name, self.features, self.params = name, features, params
        self.model = make_xgb(params)
        self.calibrator = Calibrator(calibration)

    def fit(self, train: pd.DataFrame, oof: pd.DataFrame) -> "WinModel":
        self.model.fit(train[self.features], train.home_win)
        self.calibrator.fit(oof.p_raw, oof.home_win)
        return self

    def predict_home_prob(self, df: pd.DataFrame) -> np.ndarray:
        return self.calibrator.transform(self.model.predict_proba(df[self.features])[:, 1])

    def shap_values(self, df: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Per-feature contributions to the raw log-odds of a home win, plus the base value.

        Calibration is monotone, so these rank and sign the reasons correctly even
        though they're measured before calibration.
        """
        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_values(df[self.features]), float(explainer.expected_value)


def explain_for_team(row: pd.Series, contribs: np.ndarray, features: list[str], team: str,
                     home_prob: float, top_k: int = 5) -> dict:
    """Turn one game's SHAP values into a team-perspective explanation.

    Positive `impact` means the factor raised `team`'s win chance.
    """
    is_home = row.home_team == team
    sign = 1 if is_home else -1
    factors = []
    for feat, contrib in zip(features, contribs):
        label, flip = FEATURE_LABELS[feat]
        value = float(row[feat])
        if feat == "market_logit":
            value = 1 / (1 + np.exp(-value))  # show as a probability
            value = value if is_home else 1 - value
        elif flip:
            value = sign * value
        factors.append({"feature": feat, "label": label, "value": round(value, 4),
                        "impact": round(float(sign * contrib), 4)})
    factors.sort(key=lambda f: abs(f["impact"]), reverse=True)
    return {
        "game_id": row.game_id,
        "week": int(row.week),
        "opponent": row.away_team if is_home else row.home_team,
        "venue": "home" if is_home else "away",
        "win_prob": round(float(home_prob if is_home else 1 - home_prob), 4),
        "vegas_prob": round(float(row.market_home_prob if is_home else 1 - row.market_home_prob), 4),
        "result": None if pd.isna(row.home_win) else ("W" if (row.home_win == 1) == is_home else "L"),
        "top_factors": factors[:top_k],
    }


def calibration_plot(preds: pd.DataFrame, cols: dict[str, str], path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="perfect")
    bins = np.linspace(0, 1, 11)
    for col, color in cols.items():
        cal = (preds.groupby(pd.cut(preds[col], bins), observed=True)
               .agg(pred=(col, "mean"), actual=("home_win", "mean"), n=(col, "size")))
        cal = cal[cal.n >= 10]
        ax.plot(cal.pred, cal.actual, marker="o", color=color, label=col)
    ax.set(xlabel="predicted home win probability", ylabel="actual home win rate",
           title="Calibration on 2024-2025 test seasons")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def shap_summary_plot(model: WinModel, df: pd.DataFrame, path) -> None:
    values, _ = model.shap_values(df)
    X = df[model.features].rename(columns={f: FEATURE_LABELS[f][0] for f in model.features})
    plt.figure()
    shap.summary_plot(values, X, show=False, max_display=12, plot_size=(10, 6))
    plt.title(f"What drives {model.name} (SHAP, 2024-2025 test games)")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def write_report(result: dict, vikings: dict, path) -> None:
    """Markdown summary generated from the run, so the numbers never drift from the code."""
    m = pd.DataFrame(result["test_metrics"]).T
    lines = [
        "# Phase 3 results: XGBoost, calibration, and SHAP",
        "",
        f"Rolling-origin tuning ({N_TRIALS} random trials, validation seasons "
        f"{CV_SEASONS[0]}-{CV_SEASONS[-1]}), then trained on 2010-2023 and tested on "
        f"{TEST[0]}-{TEST[-1]}. Lower is better for every metric except accuracy.",
        "",
        "| Approach | Log loss | Brier | Accuracy | Calibration error (ECE) |",
        "|---|---|---|---|---|",
    ]
    for name, r in m.iterrows():
        lines.append(f"| {name} | {r.log_loss:.4f} | {r.brier:.4f} | {r.accuracy:.1%} | {r.ece:.4f} |")
    lines += ["", "## Model details", ""]
    for name, d in result["models"].items():
        cal = ", ".join(f"{k} {v:.4f}" for k, v in d["calibration_cv_log_loss"].items())
        diff = d["log_loss_vs_vegas"]
        lines += [
            f"**{name}**: CV log loss {d['cv_log_loss']:.4f}; calibration chosen: `{d['calibration']}` "
            f"(leave-one-season-out log loss: {cal}).  ",
            f"Log loss vs. Vegas on test: {diff['mean']:+.4f} (95% CI {diff['ci_low']:+.4f} to {diff['ci_high']:+.4f}).  ",
            f"Params: `{json.dumps(d['params'])}`",
            "",
        ]
    lines += ["![Calibration](phase3_calibration.png)", "",
              "## What drives the predictions (SHAP)", "",
              "![SHAP stats](phase3_shap_xgb_stats.png)", "", "![SHAP full](phase3_shap_xgb_full.png)", "",
              "## Vikings 2026 (win probability and top 3 factors from MIN's side)", ""]
    for name, games in vikings.items():
        lines += [f"### {name}", "", "| Week | Opp | Venue | Model | Vegas | Result | Top factors (impact in log-odds) |",
                  "|---|---|---|---|---|---|---|"]
        for g in games:
            factors = "; ".join(f"{f['label']} {f['impact']:+.2f}" for f in g["top_factors"][:3])
            lines.append(f"| {g['week']} | {g['opponent']} | {g['venue']} | {g['win_prob']:.1%} | "
                         f"{g['vegas_prob']:.1%} | {g['result'] or '—'} | {factors} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def setup_mlflow() -> None:
    MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(MLRUNS_DIR / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name("vikings-win-predictor") is None:
        mlflow.create_experiment("vikings-win-predictor",
                                 artifact_location=(MLRUNS_DIR / "artifacts").as_uri())
    mlflow.set_experiment("vikings-win-predictor")


def run() -> dict:
    setup_mlflow()
    f = baseline.load_features()
    data = baseline.labeled(f)
    trainval, test = data[data.season < min(TEST)], data[data.season.isin(TEST)]
    history = data[data.season < LIVE]
    live = f[f.season == LIVE].dropna(subset=baseline.FULL_FEATURES)

    # Phase 2 models on the same split, for a like-for-like comparison.
    train, valid = data[data.season.isin(baseline.TRAIN)], data[data.season.isin(baseline.VALID)]
    cs = {n: baseline.tune_c(train, valid, feats) for n, feats in
          [("logreg_stats", baseline.STATS_FEATURES), ("logreg_full", baseline.FULL_FEATURES)]}
    test_preds, _ = baseline.predict_all(trainval, test, cs)

    summary, final_models = {}, {}
    for name, features in VARIANTS.items():
        with mlflow.start_run(run_name=name):
            log.info("%s: tuning %d trials x %d folds", name, N_TRIALS, len(CV_SEASONS))
            params, trials = tune(trainval, features)
            oof = rolling_oof(trainval, features, params, CV_SEASONS)
            method, cal_scores = choose_calibration(oof)
            log.info("%s: best cv log loss %.4f, calibration=%s", name, trials.cv_log_loss[0], method)

            model = WinModel(name, features, params, method).fit(trainval, oof)
            test_preds[name] = model.predict_home_prob(test)
            metrics = score(test_preds.home_win, test_preds[name])
            diff = dict(zip(["mean", "ci_low", "ci_high"],
                            bootstrap_log_loss_diff(test_preds.home_win, test_preds[name], test_preds.vegas)))

            mlflow.log_params({**params, "calibration": method, "n_features": len(features),
                               "cv_seasons": ",".join(map(str, CV_SEASONS)), "n_trials": N_TRIALS})
            mlflow.log_metrics({"cv_log_loss": trials.cv_log_loss[0],
                                **{f"cal_{k}_log_loss": v for k, v in cal_scores.items()},
                                **{f"test_{k}": v for k, v in metrics.items()},
                                "test_log_loss_minus_vegas": diff["mean"]})
            trials_path = REPORTS_DIR / f"phase3_{name}_trials.csv"
            trials.to_csv(trials_path, index=False)
            mlflow.log_artifact(str(trials_path))

            summary[name] = {"params": params, "cv_log_loss": trials.cv_log_loss[0],
                             "calibration": method, "calibration_cv_log_loss": cal_scores,
                             "test_metrics": metrics, "log_loss_vs_vegas": diff}

            # Final model for 2026: all completed seasons, calibrator from a longer OOF window.
            final_oof = rolling_oof(history, features, params, CV_SEASONS + TEST)
            final_models[name] = WinModel(name, features, params, method).fit(history, final_oof)
            joblib.dump(final_models[name], MODELS_DIR / f"{name}.joblib")

    approaches = ["always_home", "vegas", "logreg_stats", "logreg_full", "xgb_stats", "xgb_full"]
    all_metrics = {a: score(test_preds.home_win, test_preds[a]) for a in approaches}
    calibration_plot(test_preds, {"vegas": "#FFC62F", "logreg_stats": "#BDBDBD",
                                  "xgb_stats": "#757575", "xgb_full": "#4F2683"},
                     REPORTS_DIR / "phase3_calibration.png")
    for name in VARIANTS:
        shap_summary_plot(final_models[name], test, REPORTS_DIR / f"phase3_shap_{name}.png")

    vikings = {}
    team_games = live[(live.home_team == TEAM) | (live.away_team == TEAM)]
    for name, model in final_models.items():
        probs = model.predict_home_prob(team_games)
        contribs, _ = model.shap_values(team_games)
        vikings[name] = [explain_for_team(row, contribs[i], model.features, TEAM, probs[i])
                         for i, (_, row) in enumerate(team_games.iterrows())]

    result = {"test_metrics": all_metrics, "models": summary}
    (REPORTS_DIR / "phase3_metrics.json").write_text(json.dumps(result, indent=2, default=float))
    (REPORTS_DIR / "phase3_vikings_2026.json").write_text(json.dumps(vikings, indent=2))
    write_report(result, vikings, REPORTS_DIR / "phase3_report.md")
    with mlflow.start_run(run_name="comparison"):
        mlflow.log_metrics({f"{a}_{k}": v for a, m in all_metrics.items() for k, v in m.items()})
        for p in ["phase3_calibration.png", "phase3_shap_xgb_stats.png", "phase3_shap_xgb_full.png",
                  "phase3_metrics.json", "phase3_vikings_2026.json"]:
            mlflow.log_artifact(str(REPORTS_DIR / p))
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    r = run()
    print(pd.DataFrame(r["test_metrics"]).T.round(4))
    for name, m in r["models"].items():
        print(name, m["calibration"], m["params"], json.dumps(m["log_loss_vs_vegas"]))


if __name__ == "__main__":
    # Run via the importable module so pickled WinModels reference
    # vikpred.models.xgb.WinModel rather than __main__.WinModel.
    from vikpred.models import xgb

    xgb.main()
