from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


def fit_spread(train: pd.DataFrame) -> tuple[float, float, float, float]:
    x, y = train.iloc[:, 1].to_numpy(), train.iloc[:, 0].to_numpy()
    beta, alpha = np.polyfit(x, y, 1)
    spread = y - (alpha + beta * x)
    return float(alpha), float(beta), float(spread.mean()), float(spread.std(ddof=1))


def diagnostics(train: pd.DataFrame) -> dict[str, float]:
    score, pvalue, _ = coint(train.iloc[:, 0], train.iloc[:, 1])
    alpha, beta, _, _ = fit_spread(train)
    spread = train.iloc[:, 0] - (alpha + beta * train.iloc[:, 1])
    adf = adfuller(spread, autolag="AIC")
    return {
        "cointegration_t": float(score),
        "cointegration_p": float(pvalue),
        "adf_p": float(adf[1]),
    }


def walk_forward(
    prices: pd.DataFrame,
    train_days: int = 252,
    rebalance: int = 21,
    entry: float = 1.5,
    exit_z: float = 0.25,
    stop: float = 3.5,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    if prices.isna().any().any() or not prices.index.is_monotonic_increasing:
        raise ValueError("prices must be complete and chronologically sorted")
    records, position, previous_weight = [], 0, 0.0
    params = None
    for i in range(train_days, len(prices) - 1):
        if params is None or (i - train_days) % rebalance == 0:
            params = fit_spread(prices.iloc[i - train_days : i])
        alpha, beta, mean, std = params
        today = prices.iloc[i]
        tomorrow = prices.iloc[i + 1]
        spread = today.iloc[0] - (alpha + beta * today.iloc[1])
        z = (spread - mean) / std
        if position == 0 and abs(z) >= entry:
            position = -1 if z > 0 else 1
        elif position != 0 and (abs(z) <= exit_z or abs(z) >= stop):
            position = 0
        gross = abs(1.0) + abs(beta)
        w_y, w_x = position / gross, -position * beta / gross
        ret = w_y * (tomorrow.iloc[0] / today.iloc[0] - 1) + w_x * (
            tomorrow.iloc[1] / today.iloc[1] - 1
        )
        turnover = abs(w_y - previous_weight) + abs(w_x + previous_weight * beta)
        net = ret - turnover * cost_bps / 10_000
        records.append(
            {
                "date": prices.index[i + 1],
                "zscore": z,
                "position": position,
                "gross_return": ret,
                "net_return": net,
                "turnover": turnover,
                "hedge_ratio": beta,
            }
        )
        previous_weight = w_y
    return pd.DataFrame(records).set_index("date")


def performance(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    downside = returns[returns < 0].std(ddof=1)
    return {
        "annual_return": float(equity.iloc[-1] ** (252 / len(returns)) - 1),
        "annual_volatility": float(returns.std(ddof=1) * np.sqrt(252)),
        "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)),
        "sortino": float(returns.mean() / downside * np.sqrt(252)) if downside else 0.0,
        "max_drawdown": float(drawdown.min()),
    }


def bootstrap_sharpe_ci(
    returns: pd.Series, samples: int = 1000, seed: int = 7
) -> tuple[float, float]:
    rng, values = np.random.default_rng(seed), returns.to_numpy()
    estimates = []
    for _ in range(samples):
        draw = rng.choice(values, len(values), replace=True)
        estimates.append(np.mean(draw) / np.std(draw, ddof=1) * np.sqrt(252))
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))
