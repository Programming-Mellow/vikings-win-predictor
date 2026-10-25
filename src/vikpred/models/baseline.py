"""Phase 2: baselines vs. logistic regression on pre-game features.

Time-based evaluation: train 2010-2022, tune on 2023, test on 2024-2025.
After tuning, models are refit on train+validation before scoring the test
seasons, and refit on everything through 2025 to predict 2026.

Usage:
    python -m vikpred.models.baseline     # writes reports/phase2_* and models/*.joblib
"""

import json
import logging

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vikpred.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, TEAM  # noqa: E402
from vikpred.features.build import DIFF_FEATURES, GAME_FEATURES  # noqa: E402
from vikpred.models.metrics import bootstrap_log_loss_diff, score  # noqa: E402

log = logging.getLogger(__name__)

TRAIN = range(2010, 2023)
VALID = [2023]
TEST = [2024, 2025]
LIVE = 2026

STATS_FEATURES = DIFF_FEATURES + GAME_FEATURES
# The market enters as log-odds so it's on the same scale the model works in.
FULL_FEATURES = STATS_FEATURES + ["market_logit", "spread_line", "total_line"]
C_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def load_features() -> pd.DataFrame:
    f = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    p = f["market_home_prob"].clip(0.01, 0.99)
    return f.assign(market_logit=np.log(p / (1 - p)))


def labeled(f: pd.DataFrame) -> pd.DataFrame:
    """Played, non-tied games with complete features."""
    return f[f["home_win"].isin([0.0, 1.0])].dropna(subset=FULL_FEATURES)


def make_logreg(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))


def tune_c(train: pd.DataFrame, valid: pd.DataFrame, features: list[str]) -> float:
    losses = {}
    for c in C_GRID:
        model = make_logreg(c).fit(train[features], train["home_win"])
        losses[c] = score(valid["home_win"], model.predict_proba(valid[features])[:, 1])["log_loss"]
    best = min(losses, key=losses.get)
    log.info("tuned C=%s for %d features (valid log loss %.4f)", best, len(features), losses[best])
    return best


def predict_all(fit: pd.DataFrame, target: pd.DataFrame, cs: dict[str, float]) -> pd.DataFrame:
    """Probabilities for `target` from every approach, trained on `fit`."""
    out = target[["game_id", "season", "week", "home_team", "away_team", "home_win"]].copy()
    out["always_home"] = fit["home_win"].mean()
    out["vegas"] = target["market_home_prob"]
    models = {}
    for name, features in [("logreg_stats", STATS_FEATURES), ("logreg_full", FULL_FEATURES)]:
        models[name] = make_logreg(cs[name]).fit(fit[features], fit["home_win"])
        out[name] = models[name].predict_proba(target[features])[:, 1]
    return out, models


def coefficients(model, features: list[str]) -> pd.Series:
    coef = model.named_steps["logisticregression"].coef_[0]
    return pd.Series(coef, index=features).sort_values(key=abs, ascending=False)


def team_view(preds: pd.DataFrame, team: str) -> pd.DataFrame:
    """Flip home-perspective probabilities to `team`'s perspective."""
    rows = preds[(preds.home_team == team) | (preds.away_team == team)].copy()
    home = rows.home_team == team
    rows["opponent"] = np.where(home, rows.away_team, rows.home_team)
    rows["venue"] = np.where(home, "home", "away")
    rows["won"] = np.where(home, rows.home_win, 1 - rows.home_win)
    for col in ["vegas", "logreg_stats", "logreg_full"]:
        rows[col] = np.where(home, rows[col], 1 - rows[col])
    return rows[["week", "opponent", "venue", "vegas", "logreg_stats", "logreg_full", "won"]]


def calibration_plot(preds: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="perfect")
    bins = np.linspace(0, 1, 11)
    for col, color in [("vegas", "#FFC62F"), ("logreg_stats", "#9E9E9E"), ("logreg_full", "#4F2683")]:
        b = pd.cut(preds[col], bins)
        cal = preds.groupby(b, observed=True).agg(pred=(col, "mean"), actual=("home_win", "mean"), n=(col, "size"))
        cal = cal[cal.n >= 10]
        ax.plot(cal.pred, cal.actual, marker="o", color=color, label=col)
    ax.set(xlabel="predicted home win probability", ylabel="actual home win rate",
           title="Calibration on 2024-2025 test seasons")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(result: dict, vikings_live: pd.DataFrame, path) -> None:
    """Markdown summary generated from the run, so the numbers never drift from the code."""
    m = pd.DataFrame(result["test_metrics"]).T
    lines = [
        "# Phase 2 results: baselines vs. logistic regression",
        "",
        f"Train {result['splits']['train'][0]}-{result['splits']['train'][1]} "
        f"({result['splits']['n_train']:,} games), validate {result['splits']['valid'][0]} "
        f"({result['splits']['n_valid']}), test {'-'.join(map(str, result['splits']['test']))} "
        f"({result['splits']['n_test']} games). Lower is better for every metric except accuracy.",
        "",
        "| Approach | Log loss | Brier | Accuracy | Calibration error (ECE) |",
        "|---|---|---|---|---|",
    ]
    for name, r in m.iterrows():
        lines.append(f"| {name} | {r.log_loss:.4f} | {r.brier:.4f} | {r.accuracy:.1%} | {r.ece:.4f} |")
    lines += ["", "## Log loss vs. Vegas (bootstrap, 95% CI; negative = better than Vegas)", ""]
    for name, d in result["log_loss_vs_vegas"].items():
        lines.append(f"- **{name}**: {d['mean']:+.4f} ({d['ci_low']:+.4f} to {d['ci_high']:+.4f})")
    lines += ["", "## Largest standardized coefficients (final models)", ""]
    for name, coefs in result["coefficients"].items():
        top = ", ".join(f"`{k}` {v:+.3f}" for k, v in list(coefs.items())[:6])
        lines.append(f"- **{name}** (C={result['C'][name]}): {top}")
    lines += ["", "## Vikings 2026 (win probability from MIN's side)", "",
              "| Week | Opponent | Venue | Vegas | Stats only | Stats + Vegas | Result |",
              "|---|---|---|---|---|---|---|"]
    for _, r in vikings_live.iterrows():
        res = "—" if pd.isna(r.won) else ("W" if r.won == 1 else "L")
        lines.append(f"| {r.week} | {r.opponent} | {r.venue} | {r.vegas:.1%} | "
                     f"{r.logreg_stats:.1%} | {r.logreg_full:.1%} | {res} |")
    lines += ["", "![Calibration](phase2_calibration.png)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    f = load_features()
    data = labeled(f)
    train, valid = data[data.season.isin(TRAIN)], data[data.season.isin(VALID)]
    trainval, test = data[data.season.isin(list(TRAIN) + VALID)], data[data.season.isin(TEST)]
    log.info("train %d / valid %d / test %d games", len(train), len(valid), len(test))

    cs = {"logreg_stats": tune_c(train, valid, STATS_FEATURES),
          "logreg_full": tune_c(train, valid, FULL_FEATURES)}

    test_preds, _ = predict_all(trainval, test, cs)
    approaches = ["always_home", "vegas", "logreg_stats", "logreg_full"]
    metrics = {a: score(test_preds.home_win, test_preds[a]) for a in approaches}
    vs_vegas = {a: dict(zip(["mean", "ci_low", "ci_high"],
                            bootstrap_log_loss_diff(test_preds.home_win, test_preds[a], test_preds.vegas)))
                for a in ["logreg_stats", "logreg_full"]}

    # Final models: everything through the last completed season, applied to 2026.
    history = data[data.season < LIVE]
    live = f[(f.season == LIVE)].dropna(subset=FULL_FEATURES)
    live_preds, final_models = predict_all(history, live, cs)

    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    for name, model in final_models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    calibration_plot(test_preds, REPORTS_DIR / "phase2_calibration.png")

    result = {
        "splits": {"train": [min(TRAIN), max(TRAIN)], "valid": VALID, "test": TEST,
                   "n_train": len(train), "n_valid": len(valid), "n_test": len(test)},
        "C": cs,
        "test_metrics": metrics,
        "log_loss_vs_vegas": vs_vegas,
        "coefficients": {
            "logreg_stats": coefficients(final_models["logreg_stats"], STATS_FEATURES).round(4).to_dict(),
            "logreg_full": coefficients(final_models["logreg_full"], FULL_FEATURES).round(4).to_dict(),
        },
    }
    (REPORTS_DIR / "phase2_metrics.json").write_text(json.dumps(result, indent=2))
    vikings_live = team_view(live_preds, TEAM)
    vikings_live.to_csv(REPORTS_DIR / "phase2_vikings_2026.csv", index=False)
    write_report(result, vikings_live, REPORTS_DIR / "phase2_baseline.md")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    r = run()
    print(pd.DataFrame(r["test_metrics"]).T.round(4))
    print(json.dumps(r["log_loss_vs_vegas"], indent=2))
