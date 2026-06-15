# Statistical Pairs Trading

A leakage-controlled, walk-forward study of equity pairs with rolling hedge
ratios, transaction costs, stationarity diagnostics, bootstrap uncertainty,
and explicit failure analysis. This is research software, not investment advice.

## What is implemented

- Frozen daily close snapshots for AAPL-MSFT, KO-PEP, and XOM-CVX.
- Engle-Granger cointegration and ADF spread diagnostics on formation windows.
- Rolling hedge-ratio estimation with explicit signal-date and traded-return
  alignment.
- Entry, exit, stop-loss, transaction-cost, slippage, exposure, turnover, and
  bootstrap Sharpe analysis.
- Baselines for cash, equal-weight buy-and-hold, and naive log-spread z-score
  trading.
- Transaction-cost sensitivity and machine-readable result artifacts.

```bash
uv sync
make test
make reproduce-smoke
make reproduce-results
```

Full runs freeze Yahoo Finance data under `data/` and record SHA-256 checksums.
