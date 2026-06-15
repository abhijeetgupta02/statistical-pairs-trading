# Model Card

This repository is an independently reproducible research implementation of
statistical pairs-trading evaluation. It is not investment advice.

## Intended use

- Demonstrate leakage-controlled backtesting and market-data hygiene.
- Compare a rolling hedge-ratio strategy against simple baselines.
- Record uncertainty, transaction-cost sensitivity, and failure cases.

## Not intended use

- Live trading or capital allocation.
- Claims that a pair remains cointegrated after the sampled period.
- Claims of profitability without additional execution, borrow, tax, and
  capacity analysis.

## Evaluation

The verified run uses lagged walk-forward signals, frozen data snapshots,
bootstrap Sharpe intervals, baseline comparisons, and transaction-cost
sensitivity. The headline result is deliberately cautious: only KO-PEP passes
the 5% cointegration screen, and all strategy Sharpe intervals include zero.
