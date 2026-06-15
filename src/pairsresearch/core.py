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
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    if prices.isna().any().any() or not prices.index.is_monotonic_increasing:
        raise ValueError("prices must be complete and chronologically sorted")
    records, position = [], 0
    previous_w_y, previous_w_x = 0.0, 0.0
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
        turnover = abs(w_y - previous_w_y) + abs(w_x - previous_w_x)
        net = ret - turnover * (cost_bps + slippage_bps) / 10_000
        records.append(
            {
                "signal_date": prices.index[i],
                "date": prices.index[i + 1],
                "train_start": prices.index[i - train_days],
                "train_end": prices.index[i - 1],
                "zscore": z,
                "position": position,
                "weight_y": w_y,
                "weight_x": w_x,
                "gross_return": ret,
                "net_return": net,
                "turnover": turnover,
                "hedge_ratio": beta,
                "alpha": alpha,
                "spread_mean": mean,
                "spread_std": std,
            }
        )
        previous_w_y, previous_w_x = w_y, w_x
    return pd.DataFrame(records).set_index("date")


def performance(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    downside = returns[returns < 0].std(ddof=1)
    volatility = returns.std(ddof=1)
    sharpe = returns.mean() / volatility * np.sqrt(252) if np.isfinite(volatility) and volatility > 0 else 0.0
    sortino = (
        returns.mean() / downside * np.sqrt(252)
        if np.isfinite(downside) and downside > 0
        else 0.0
    )
    return {
        "annual_return": float(equity.iloc[-1] ** (252 / len(returns)) - 1),
        "annual_volatility": float(volatility * np.sqrt(252)),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(drawdown.min()),
    }


def bootstrap_sharpe_ci(
    returns: pd.Series, samples: int = 1000, seed: int = 7
) -> tuple[float, float]:
    rng, values = np.random.default_rng(seed), returns.to_numpy()
    estimates = []
    for _ in range(samples):
        draw = rng.choice(values, len(values), replace=True)
        std = np.std(draw, ddof=1)
        estimates.append(np.mean(draw) / std * np.sqrt(252) if std > 0 else 0.0)
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def exposure_metrics(run: pd.DataFrame) -> dict[str, float]:
    gross = run[["weight_y", "weight_x"]].abs().sum(axis=1)
    net = run[["weight_y", "weight_x"]].sum(axis=1)
    return {
        "average_gross_exposure": float(gross.mean()),
        "average_net_exposure": float(net.mean()),
        "average_abs_net_exposure": float(net.abs().mean()),
        "average_turnover": float(run.turnover.mean()),
        "trade_fraction": float((run.position != 0).mean()),
    }


def buy_and_hold(prices: pd.DataFrame, start: pd.Index) -> pd.Series:
    returns = prices.pct_change().dropna()
    equal_weight = returns.mean(axis=1)
    return equal_weight.reindex(start).fillna(0.0)


def cash_baseline(index: pd.Index) -> pd.Series:
    return pd.Series(0.0, index=index, name="cash")


def naive_zscore(
    prices: pd.DataFrame,
    train_days: int = 252,
    entry: float = 1.5,
    exit_z: float = 0.25,
    stop: float = 3.5,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    if prices.isna().any().any() or not prices.index.is_monotonic_increasing:
        raise ValueError("prices must be complete and chronologically sorted")
    log_spread = np.log(prices.iloc[:, 0]) - np.log(prices.iloc[:, 1])
    records, position, previous_w_y, previous_w_x = [], 0, 0.0, 0.0
    for i in range(train_days, len(prices) - 1):
        train = log_spread.iloc[i - train_days : i]
        z = (log_spread.iloc[i] - train.mean()) / train.std(ddof=1)
        if position == 0 and abs(z) >= entry:
            position = -1 if z > 0 else 1
        elif position != 0 and (abs(z) <= exit_z or abs(z) >= stop):
            position = 0
        w_y, w_x = 0.5 * position, -0.5 * position
        today = prices.iloc[i]
        tomorrow = prices.iloc[i + 1]
        gross_return = w_y * (tomorrow.iloc[0] / today.iloc[0] - 1) + w_x * (
            tomorrow.iloc[1] / today.iloc[1] - 1
        )
        turnover = abs(w_y - previous_w_y) + abs(w_x - previous_w_x)
        records.append(
            {
                "signal_date": prices.index[i],
                "date": prices.index[i + 1],
                "zscore": z,
                "position": position,
                "weight_y": w_y,
                "weight_x": w_x,
                "gross_return": gross_return,
                "net_return": gross_return - turnover * cost_bps / 10_000,
                "turnover": turnover,
            }
        )
        previous_w_y, previous_w_x = w_y, w_x
    return pd.DataFrame(records).set_index("date")


def rank_pairs(diagnostic_rows: dict[str, dict[str, float]]) -> list[dict[str, float | str]]:
    rows = [
        {"pair": pair, **values, "selected_at_5pct": values["cointegration_p"] < 0.05}
        for pair, values in diagnostic_rows.items()
    ]
    return sorted(rows, key=lambda row: (row["cointegration_p"], row["adf_p"]))
