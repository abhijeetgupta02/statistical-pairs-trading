from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from .artifacts import environment, run_dir, save_json, sha256
from .core import (
    bootstrap_sharpe_ci,
    buy_and_hold,
    cash_baseline,
    diagnostics,
    exposure_metrics,
    naive_zscore,
    performance,
    rank_pairs,
    walk_forward,
)


PAIRS = [("AAPL", "MSFT"), ("KO", "PEP"), ("XOM", "CVX")]


def publish_latest(out: Path) -> None:
    latest = Path("reports/latest")
    latest.mkdir(parents=True, exist_ok=True)
    for existing in latest.iterdir():
        if existing.is_file():
            existing.unlink()
    for source in out.iterdir():
        if source.is_file():
            shutil.copy2(source, latest / source.name)
    Path("reports/SOURCE_RUN.txt").write_text(f"{out}\n")


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
    pair_diagnostics = {}
    cost_sensitivity = {}
    baseline_results = {}
    pairs = [("Y", "X")] if args.smoke else PAIRS
    for a, b in pairs:
        prices, manifest = (
            (synthetic(), {"source": "synthetic", "rows": 800}) if args.smoke else load_pair(a, b)
        )
        pair_name = f"{a}-{b}"
        pair_diagnostics[pair_name] = diagnostics(prices.iloc[:252])
        run = walk_forward(prices, train_days=252, cost_bps=1, slippage_bps=0.5)
        metrics = performance(run.net_return)
        metrics["sharpe_ci_95"] = bootstrap_sharpe_ci(run.net_return, 250 if args.smoke else 2000)
        metrics.update(pair_diagnostics[pair_name])
        metrics.update(exposure_metrics(run))
        results[pair_name] = metrics
        costs = [0, 1] if args.smoke else [0, 1, 5, 10]
        cost_sensitivity[pair_name] = {
            f"{cost}_bps": performance(
                walk_forward(prices, train_days=252, cost_bps=cost, slippage_bps=0).net_return
            )
            for cost in costs
        }
        hold = buy_and_hold(prices, run.index)
        naive = naive_zscore(prices, train_days=252, cost_bps=1).net_return.reindex(run.index)
        cash = cash_baseline(run.index)
        baseline_results[pair_name] = {
            "strategy": metrics,
            "naive_zscore": performance(naive),
            "buy_and_hold_equal_weight": performance(hold),
            "cash": performance(cash),
        }
        manifests.append({"pair": pair_name, **manifest})
        run.assign(pair=pair_name, strategy="walk_forward").reset_index().pipe(predictions.append)
        naive_zscore(prices, train_days=252, cost_bps=1).assign(
            pair=pair_name, strategy="naive_zscore"
        ).reset_index().pipe(predictions.append)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_parquet(out / "predictions.parquet", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, group in pred[pred.strategy == "walk_forward"].groupby("pair"):
        ax.plot(group.date, (1 + group.net_return).cumprod(), label=name)
    ax.set_ylabel("Growth of $1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "equity_curves.png", dpi=180)
    plt.close(fig)
    save_json(
        out / "config.yaml",
        {
            "train_days": 252,
            "rebalance": 21,
            "entry": 1.5,
            "exit_z": 0.25,
            "stop": 3.5,
            "cost_bps": 1,
            "slippage_bps": 0.5,
        },
    )
    save_json(out / "environment.json", environment())
    save_json(out / "data_manifest.json", manifests)
    save_json(out / "metrics.json", results)
    save_json(
        out / "statistical_tests.json",
        {
            "pair_selection": rank_pairs(pair_diagnostics),
            "bootstrap_sharpe_ci": {k: {"sharpe_ci_95": v["sharpe_ci_95"]} for k, v in results.items()},
            "baseline_comparison": baseline_results,
            "transaction_cost_sensitivity": cost_sensitivity,
        },
    )
    (out / "run.log").write_text("completed\n")
    if not args.smoke:
        publish_latest(out)
    print(out)


if __name__ == "__main__":
    main()
