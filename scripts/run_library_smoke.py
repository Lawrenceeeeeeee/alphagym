"""Exercise the installed public API using synthetic data, including a worker."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from mlquant import FactorDefinition, Workspace, storage_io
from mlquant.serialization import dumps


def run(root: Path, *, background: bool = False) -> dict:
    workspace = Workspace(root)
    workspace.initialize()
    dates = pd.bdate_range("2022-11-01", "2024-04-05")
    symbols = [f"{number:06d}.SZ" for number in range(1, 21)]
    day = np.repeat(np.arange(len(dates)), len(symbols))
    stock = np.tile(np.arange(1, len(symbols) + 1), len(dates))
    close = 10 + stock + day * (0.005 + stock / 10000) + np.sin(day / 3 + stock) * stock * .04
    daily = pd.DataFrame({
        "trade_date": dates.repeat(len(symbols)), "symbol": symbols * len(dates),
        "open": close * .999, "high": close * 1.01, "low": close * .99,
        "close": close, "volume": 1000000 + stock, "amount": close * (1000000 + stock),
    })
    equity = root / "equity"
    equity.mkdir(parents=True, exist_ok=True)
    storage_io.write_frame(daily, equity / "daily.parquet", index=False)
    storage_io.write_frame(pd.DataFrame({"trade_date": dates[0], "symbol": symbols, "adjust_factor": 1.0}),
        equity / "adjustments.parquet", index=False,
    )
    for factor_id, formula in (("SMOKE_RETURN", "=RETURN(market.adj_close, 5)"),
                               ("SMOKE_VOL", "=VOLATILITY(market.adj_close, 20)")):
        workspace.save_factor(FactorDefinition(
            factor_id=factor_id, name=factor_id, formula=formula,
            hypothesis_id=f"synthetic_{factor_id.lower()}", family="momentum",
            expected_direction="positive",
        ))
    workspace.run_factors(
        ["SMOKE_RETURN", "SMOKE_VOL"], start_date="2023-01-01",
        end_date="2024-03-31", mode="smoke",
    )
    task = workspace.create_report({
        "name": "Synthetic API smoke", "mode": "smoke", "holding_period": "1M",
        "window": {"start": "2023-01-01", "end": "2024-03-31"},
        "splits": {
            "development": ["2023-01-01", "2023-06-30"],
            "validation": ["2023-07-01", "2023-12-31"],
            "test": ["2024-01-01", "2024-03-31"],
        },
        "factors": {"include": ["SMOKE_RETURN", "SMOKE_VOL"]},
        "combine": {"methods": ["equal", "ic_decay"], "correlation_threshold": .99},
    })
    report_id = task["report_id"]
    if background:
        workspace.start_report(report_id)
        row = workspace.wait_report(report_id, timeout=180, poll_interval=.2)
        if row["status"] != "succeeded":
            raise RuntimeError(row["error"])
        output = Path(row["path"])
    else:
        output = workspace.execute_report(report_id)
    required = ("spec.yaml", "manifest.json", "monthly.parquet", "summary.csv",
                "correlation.parquet", "combo.json", "report.md", "report.html")
    assert all(storage_io.exists(output / name) for name in required)
    manifest = workspace.report_manifest(report_id)
    assert manifest["spec"]["mode"] == "smoke"
    assert "echarts" in storage_io.read_text(output / "report.html", encoding="utf-8")
    assert not list(storage_io.glob(output, "*.png"))
    return {"ok": True, "report_id": report_id, "path": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    # Always use an isolated temporary root, never overwrite a user's market data.
    root = Path(tempfile.mkdtemp(prefix="mlquant-library-smoke-"))
    print(dumps(run(root, background=args.background)))


if __name__ == "__main__":
    main()
