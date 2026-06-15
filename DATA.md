# Data

Full experiments use frozen Yahoo Finance daily adjusted-close snapshots for:

- AAPL-MSFT, 2015-01-01 through 2023-12-31
- KO-PEP, 2015-01-01 through 2023-12-31
- XOM-CVX, 2015-01-01 through 2023-12-31

CSV snapshots are stored under `data/` and checksummed in each run's
`data_manifest.json`. If a snapshot is missing, the runner can download it
through `yfinance`, but committed result reports are based on the frozen local
files. Tests and smoke runs use deterministic synthetic fixtures and do not
require network access.
