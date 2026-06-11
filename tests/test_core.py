import numpy as np
import pandas as pd

from pairsresearch.core import fit_spread, walk_forward


def fixture(rows=700):
    rng = np.random.default_rng(3)
    x = 100 + np.cumsum(rng.normal(0, 0.4, rows))
    residual = np.zeros(rows)
    for i in range(1, rows):
        residual[i] = 0.9 * residual[i - 1] + rng.normal(0, 0.3)
    return pd.DataFrame(
        {"Y": 4 + 1.2 * x + residual, "X": x}, index=pd.date_range("2020-01-01", periods=rows)
    )


def test_fit_recovers_hedge_ratio():
    _, beta, _, _ = fit_spread(fixture().iloc[:500])
    assert abs(beta - 1.2) < 0.05


def test_predictions_begin_after_training_window():
    prices = fixture()
    result = walk_forward(prices, train_days=252)
    assert result.index.min() == prices.index[253]
    assert len(result) == len(prices) - 253


def test_rejects_reverse_time():
    prices = fixture().sort_index(ascending=False)
    try:
        walk_forward(prices)
    except ValueError:
        return
    raise AssertionError("reverse chronology must be rejected")
