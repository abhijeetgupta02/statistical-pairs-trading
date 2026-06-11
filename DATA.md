# Data

Data used by full experiments is downloaded from Yahoo Finance through
`yfinance`, normalized to CSV, checksummed, and stored under `data/`.
Tests and smoke runs use deterministic synthetic fixtures and do not
require network access.
