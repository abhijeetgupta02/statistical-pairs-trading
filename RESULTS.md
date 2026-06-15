# Verified Results

## Reproduction status

`make reproduce-results` completed locally on June 15, 2026 using frozen Yahoo
Finance price snapshots. These results are exploratory and are not evidence of
future profitability.

| Pair | Cointegration p | Selected? | Strategy Sharpe | 95% bootstrap CI | Max drawdown |
|---|---:|---|---:|---:|---:|
| AAPL-MSFT | 0.2923 | No | 0.627 | [-0.072, 1.342] | -20.3% |
| KO-PEP | 0.0294 | Yes | 0.202 | [-0.475, 0.879] | -13.0% |
| XOM-CVX | 0.1068 | No | -0.119 | [-0.802, 0.544] | -30.4% |

Only KO–PEP rejects the no-cointegration null at 5%. AAPL–MSFT and XOM–CVX
therefore fail the repository's primary statistical-selection criterion, and
their strategy metrics should not be interpreted as validated opportunities.
All strategy results include 1 bps transaction cost plus 0.5 bps slippage and
use signals formed before the traded close-to-close return.

## Baseline comparison

| Pair | Strategy Sharpe | Naive z-score Sharpe | Equal-weight buy-hold Sharpe |
|---|---:|---:|---:|
| AAPL-MSFT | 0.627 | 0.446 | 1.132 |
| KO-PEP | 0.202 | 0.191 | 0.581 |
| XOM-CVX | -0.119 | 0.104 | 0.480 |

The walk-forward strategy improves over the naive z-score baseline for
AAPL-MSFT and KO-PEP in this historical sample, but it does not beat
equal-weight buy-and-hold Sharpe. Because only KO-PEP passes the cointegration
gate and its bootstrap Sharpe interval crosses zero, the result is best read
as a leakage-controlled research workflow, not a trade recommendation.

See `reports/latest/` for pair selection, transaction-cost sensitivity,
predictions, and exact machine-readable metrics.
