import numpy as np
import pytest

from vikpred.models.metrics import bootstrap_log_loss_diff, expected_calibration_error, score


def test_perfectly_calibrated_forecast_has_zero_ece():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert expected_calibration_error(y, p) == pytest.approx(0.0)


def test_overconfident_forecast_has_large_ece():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.95, 0.95, 0.95, 0.95])
    assert expected_calibration_error(y, p) == pytest.approx(0.45)


def test_score_keys_and_coin_flip_values():
    s = score(np.array([1, 0]), np.array([0.5, 0.5]))
    assert s["log_loss"] == pytest.approx(np.log(2))
    assert s["brier"] == pytest.approx(0.25)


def test_bootstrap_diff_is_zero_for_identical_forecasts():
    y = np.array([1, 0, 1, 1, 0])
    p = np.array([0.6, 0.4, 0.7, 0.55, 0.3])
    mean, lo, hi = bootstrap_log_loss_diff(y, p, p, n=50)
    assert mean == lo == hi == 0.0
