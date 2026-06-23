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

## Headline results

`make reproduce-results` on frozen 2015–2023 Yahoo Finance daily closes (2,264
rows per pair). All strategy metrics include 1 bps transaction cost plus 0.5 bps
slippage, with signals formed before the traded close-to-close return.

| Pair | Cointegration p | Selected (5%) | Strategy Sharpe | 95% bootstrap CI | Max drawdown |
|---|---:|:--:|---:|---:|---:|
| AAPL-MSFT | 0.2923 | No | 0.627 | [-0.072, 1.342] | -20.3% |
| KO-PEP | 0.0294 | Yes | 0.202 | [-0.475, 0.879] | -13.0% |
| XOM-CVX | 0.1068 | No | -0.119 | [-0.802, 0.544] | -30.4% |

Only KO-PEP rejects the no-cointegration null at 5%, so it is the single pair
that passes the repository's selection gate — and even its bootstrap Sharpe
interval still crosses zero. These are exploratory research results, not evidence
of future profitability. See [`RESULTS.md`](RESULTS.md) for the baseline
comparison and transaction-cost sensitivity.

## Reproduce

```bash
uv sync
make test
make reproduce-smoke
make reproduce-results
```

Full runs freeze Yahoo Finance data under `data/` and record SHA-256 checksums.
