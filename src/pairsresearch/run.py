from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from .artifacts import environment, run_dir, save_json, sha256
from .core import bootstrap_sharpe_ci, diagnostics, performance, walk_forward


PAIRS = [("AAPL", "MSFT"), ("KO", "PEP"), ("XOM", "CVX")]


def synthetic(rows: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    common = np.cumsum(rng.normal(0, 0.01, rows))
    noise = np.zeros(rows)
    for i in range(1, rows):
        noise[i] = 0.92 * noise[i - 1] + rng.normal(0, 0.006)
    return pd.DataFrame({"Y": 100 * np.exp(common + noise), "X": 90 * np.exp(common)})


def load_pair(a: str, b: str) -> tuple[pd.DataFrame, dict]:
    path = Path("data") / f"{a}_{b}_2015_2023.csv"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        raw = yf.download(
            [a, b], start="2015-01-01", end="2024-01-01", auto_adjust=True, progress=False
        )
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        close[[a, b]].dropna().to_csv(path, index_label="Date")
    frame = pd.read_csv(path, parse_dates=["Date"], index_col="Date")[[a, b]].dropna()
    return frame, {"path": str(path), "sha256": sha256(path), "rows": len(frame)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = run_dir("statistical-pairs-trading", args.smoke)
    results, manifests, predictions = {}, [], []
    pairs = [("Y", "X")] if args.smoke else PAIRS
    for a, b in pairs:
        prices, manifest = (
            (synthetic(), {"source": "synthetic", "rows": 800}) if args.smoke else load_pair(a, b)
        )
        run = walk_forward(prices, train_days=252, cost_bps=1)
        metrics = performance(run.net_return)
        metrics["sharpe_ci_95"] = bootstrap_sharpe_ci(run.net_return, 250 if args.smoke else 2000)
        metrics.update(diagnostics(prices.iloc[:252]))
        metrics["average_turnover"] = float(run.turnover.mean())
        results[f"{a}-{b}"] = metrics
        manifests.append({"pair": f"{a}-{b}", **manifest})
        run.assign(pair=f"{a}-{b}").reset_index().pipe(predictions.append)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_parquet(out / "predictions.parquet", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, group in pred.groupby("pair"):
        ax.plot(group.date, (1 + group.net_return).cumprod(), label=name)
    ax.set_ylabel("Growth of $1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "equity_curves.png", dpi=180)
    plt.close(fig)
    save_json(
        out / "config.yaml", {"train_days": 252, "rebalance": 21, "entry": 1.5, "cost_bps": 1}
    )
    save_json(out / "environment.json", environment())
    save_json(out / "data_manifest.json", manifests)
    save_json(out / "metrics.json", results)
    save_json(
        out / "statistical_tests.json",
        {k: {"sharpe_ci_95": v["sharpe_ci_95"]} for k, v in results.items()},
    )
    (out / "run.log").write_text("completed\n")
    print(out)


if __name__ == "__main__":
    main()
