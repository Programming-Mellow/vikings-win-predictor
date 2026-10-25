"""Phase 3 tests: calibration, team-perspective explanations, and model persistence."""

import joblib
import numpy as np
import pandas as pd
import pytest

from vikpred.models import xgb

FEATURES = ["diff_off_epa_ewm", "rest_diff", "temp", "market_logit"]


def synthetic_games(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "game_id": [f"g{i}" for i in range(n)],
        "season": np.repeat(np.arange(2016, 2016 + n // 100), 100)[:n],
        "week": 1,
        "home_team": "MIN", "away_team": "GB",
        "diff_off_epa_ewm": rng.normal(0, 0.1, n),
        "rest_diff": rng.integers(-7, 8, n),
        "temp": rng.uniform(20, 90, n),
        "market_logit": rng.normal(0, 1, n),
    })
    signal = 8 * df.diff_off_epa_ewm + 0.8 * df.market_logit
    df["home_win"] = (signal + rng.logistic(size=n) > 0).astype(float)
    df["market_home_prob"] = 1 / (1 + np.exp(-df.market_logit))
    return df


@pytest.mark.parametrize("method", xgb.CALIBRATION_METHODS)
def test_calibrators_are_monotone_and_bounded(method):
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 500)
    y = (rng.uniform(size=500) < p).astype(int)
    out = xgb.Calibrator(method).fit(p, y).transform(np.linspace(0.01, 0.99, 50))
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0 and out.max() <= 1


def test_choose_calibration_prefers_fixing_a_miscalibrated_model():
    rng = np.random.default_rng(2)
    true_p = rng.uniform(0.2, 0.8, 2000)
    y = (rng.uniform(size=2000) < true_p).astype(int)
    # Overconfident raw predictions: pushed toward 0/1.
    raw = 1 / (1 + np.exp(-3 * np.log(true_p / (1 - true_p))))
    oof = pd.DataFrame({"season": np.repeat([2019, 2020, 2021, 2022], 500), "home_win": y, "p_raw": raw})
    method, scores = xgb.choose_calibration(oof)
    assert method != "none"
    assert scores[method] < scores["none"]


def make_row(home_team: str, away_team: str) -> pd.Series:
    return pd.Series({"game_id": "g", "week": 7, "home_team": home_team, "away_team": away_team,
                      "diff_off_epa_ewm": 0.10, "rest_diff": 7, "temp": 70.0,
                      "market_logit": np.log(0.7 / 0.3), "market_home_prob": 0.7, "home_win": np.nan})


def test_explanation_from_home_side_keeps_signs():
    contribs = np.array([0.30, 0.05, -0.01, 0.40])
    e = xgb.explain_for_team(make_row("MIN", "IND"), contribs, FEATURES, "MIN", home_prob=0.73)
    assert e["venue"] == "home" and e["opponent"] == "IND"
    assert e["win_prob"] == pytest.approx(0.73) and e["vegas_prob"] == pytest.approx(0.7)
    factors = {f["feature"]: f for f in e["top_factors"]}
    assert factors["diff_off_epa_ewm"]["value"] == pytest.approx(0.10)
    assert factors["diff_off_epa_ewm"]["impact"] == pytest.approx(0.30)
    assert factors["market_logit"]["value"] == pytest.approx(0.7)
    assert e["result"] is None


def test_explanation_from_away_side_flips_signs():
    contribs = np.array([0.30, 0.05, -0.01, 0.40])
    e = xgb.explain_for_team(make_row("DET", "MIN"), contribs, FEATURES, "MIN", home_prob=0.73)
    assert e["venue"] == "away" and e["opponent"] == "DET"
    assert e["win_prob"] == pytest.approx(0.27) and e["vegas_prob"] == pytest.approx(0.3)
    factors = {f["feature"]: f for f in e["top_factors"]}
    # Home team's EPA edge is MIN's deficit, and it hurt MIN.
    assert factors["diff_off_epa_ewm"]["value"] == pytest.approx(-0.10)
    assert factors["diff_off_epa_ewm"]["impact"] == pytest.approx(-0.30)
    # Temperature is the same for both teams; only its impact flips.
    assert factors["temp"]["value"] == pytest.approx(70.0)
    assert factors["temp"]["impact"] == pytest.approx(0.01)
    assert factors["market_logit"]["value"] == pytest.approx(0.3)


def test_factors_sorted_by_absolute_impact():
    contribs = np.array([0.05, -0.50, 0.01, 0.20])
    e = xgb.explain_for_team(make_row("MIN", "IND"), contribs, FEATURES, "MIN", home_prob=0.5, top_k=3)
    assert [f["feature"] for f in e["top_factors"]] == ["rest_diff", "market_logit", "diff_off_epa_ewm"]


def test_win_model_shap_adds_up_and_survives_pickling(tmp_path):
    df = synthetic_games()
    params = {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 50, "min_child_weight": 5,
              "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0}
    oof = xgb.rolling_oof(df, FEATURES, params, [2020, 2021])
    model = xgb.WinModel("test", FEATURES, params, "platt").fit(df, oof)

    values, base = model.shap_values(df.head(20))
    raw = model.model.predict_proba(df.head(20)[FEATURES])[:, 1]
    np.testing.assert_allclose(1 / (1 + np.exp(-(values.sum(axis=1) + base))), raw, atol=1e-5)

    path = tmp_path / "m.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert type(loaded).__module__ == "vikpred.models.xgb"
    np.testing.assert_allclose(loaded.predict_home_prob(df.head(20)), model.predict_home_prob(df.head(20)))
