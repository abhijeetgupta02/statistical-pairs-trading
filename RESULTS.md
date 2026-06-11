# Verified Results

## Reproduction status

`make reproduce-results` completed locally on June 11, 2026 using frozen Yahoo
Finance price snapshots. These results are exploratory and are not evidence of
future profitability.

| Pair | Cointegration p | Sharpe | 95% bootstrap CI | Max drawdown |
|---|---:|---:|---:|---:|
| AAPL-MSFT | 0.2923 | 0.635 | [-0.065, 1.350] | -20.2% |
| KO-PEP | 0.0294 | 0.212 | [-0.465, 0.890] | -12.9% |
| XOM-CVX | 0.1068 | -0.113 | [-0.795, 0.551] | -30.2% |

Only KO–PEP rejects the no-cointegration null at 5%. AAPL–MSFT and XOM–CVX
therefore fail the repository's primary statistical-selection criterion, and
their strategy metrics should not be interpreted as validated opportunities.
All results include the configured transaction costs and use lagged signals.
