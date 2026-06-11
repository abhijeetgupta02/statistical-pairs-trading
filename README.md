# Statistical Pairs Trading

A leakage-controlled, walk-forward study of equity pairs with rolling hedge
ratios, transaction costs, stationarity diagnostics, bootstrap uncertainty,
and explicit failure analysis. This is research software, not investment advice.

```bash
uv sync
make test
make reproduce-results
```

Full runs freeze Yahoo Finance data under `data/` and record SHA-256 checksums.
