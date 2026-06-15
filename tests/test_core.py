import numpy as np
import pandas as pd

from pairsresearch.core import buy_and_hold, cash_baseline, fit_spread, naive_zscore, walk_forward


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
    assert result.signal_date.iloc[0] == prices.index[252]
    assert result.train_end.iloc[0] == prices.index[251]


def test_rejects_reverse_time():
    prices = fixture().sort_index(ascending=False)
    try:
        walk_forward(prices)
    except ValueError:
        return
    raise AssertionError("reverse chronology must be rejected")


def test_turnover_uses_both_legs_without_future_weights():
    prices = fixture()
    result = walk_forward(prices, train_days=252, entry=0.0, cost_bps=0)
    first = result.iloc[0]
    expected_first_turnover = abs(first.weight_y) + abs(first.weight_x)
    assert np.isclose(first.turnover, expected_first_turnover)
    second = result.iloc[1]
    expected_second_turnover = abs(second.weight_y - first.weight_y) + abs(
        second.weight_x - first.weight_x
    )
    assert np.isclose(second.turnover, expected_second_turnover)


def test_baselines_align_to_strategy_dates():
    prices = fixture()
    result = walk_forward(prices, train_days=252)
    hold = buy_and_hold(prices, result.index)
    cash = cash_baseline(result.index)
    naive = naive_zscore(prices, train_days=252)
    assert hold.index.equals(result.index)
    assert cash.index.equals(result.index)
    assert naive.index.min() == result.index.min()
    assert np.isclose(cash.abs().sum(), 0)
