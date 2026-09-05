"""Non-neutralized, explicitly non-formal monthly/weekly/daily ML comparison."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mlquant.ml_frequency import (
    add_financial_features,
    financial_events,
    frequency_signal_dates,
    load_frequency_inputs,
    market_features,
)


def build_features(root: Path, output: Path, factors: list[str]) -> list[Path]:
    """One daily feature calculation reused by all three holding horizons."""
    paths = [root / "equity" / f"{name}.parquet" for name in
             ("daily", "adjustments", "fundamentals", "calendar", "securities")]
    signature = hashlib.sha256(json.dumps({
        "version": 1, "factors": factors,
        "files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths],
        "code": hashlib.sha256((Path(__file__).parents[1] / "src/mlquant/ml_frequency.py").read_bytes()).hexdigest(),
    }, sort_keys=True).encode()).hexdigest()[:20]
    cache = output / "features" / signature
    cache.mkdir(parents=True, exist_ok=True)
    calendar = pd.read_parquet(root / "equity/calendar.parquet")
    financial = pd.read_parquet(root / "equity/fundamentals.parquet")
    events_path = cache / "financial_events.parquet"
    if not events_path.exists():
        financial_events(financial).to_parquet(events_path, index=False)
    events = pd.read_parquet(events_path)
    securities = pd.read_parquet(root / "equity/securities.parquet").set_index("symbol")
    opened = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, "trade_date"])).sort_values()
    files = []
    for year in range(2014, 2027):
        path = cache / f"features_{year}.parquet"
        files.append(path)
        if path.exists():
            continue
        started = time.monotonic()
        signals = frequency_signal_dates(calendar, f"{year}-01-01", min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp("2026-08-31")), "daily")
        daily, _financial, _calendar, adjustments = load_frequency_inputs(root, signals[0], signals[-1])
        del _financial, _calendar, adjustments
        raw, observations = market_features(daily, signals, factors)
        raw = add_financial_features(raw, observations, financial, factors, events=events)
        symbols = raw.index.get_level_values("symbol")
        dates = raw.index.get_level_values("signal_date")
        listed = pd.to_datetime(symbols.map(securities.list_date))
        age = opened.searchsorted(dates, side="right") - opened.searchsorted(listed)
        # PIT pool rule, set before seeing returns. No historical ST filter is
        # available: this is explicitly ALL_A including ST, not a tradable pool.
        eligible = (age >= 250) & listed.notna() & observations.close.gt(0).to_numpy() & observations.float_market_cap.gt(0).to_numpy()
        raw = raw.loc[eligible]
        normalized = np.full(raw.shape, np.nan, dtype=np.float32)
        for positions in raw.groupby(level="signal_date", sort=False).indices.values():
            block = raw.iloc[positions].to_numpy(float)
            median = np.nanmedian(block, axis=0)
            mad = np.nanmedian(np.abs(block - median), axis=0)
            block = np.clip(block, np.where(mad > 0, median - 5 * mad, -np.inf), np.where(mad > 0, median + 5 * mad, np.inf))
            mean, std = np.nanmean(block, axis=0), np.nanstd(block, axis=0)
            normalized[positions] = np.divide(block - mean, std, out=np.full_like(block, np.nan), where=std > 0)
        frame = pd.DataFrame(normalized, index=raw.index, columns=factors)
        frame["float_market_cap"] = observations.float_market_cap.reindex(raw.index)
        frame.reset_index().to_parquet(path, index=False)
        print(f"cached {year}: {len(frame):,} rows in {time.monotonic()-started:.1f}s", flush=True)
        del daily, raw, observations, frame, normalized
        gc.collect()
    (output / "feature_manifest.json").write_text(json.dumps({"signature": signature, "files": [str(p) for p in files], "factors": factors}, indent=2), encoding="utf-8")
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["features", "evaluate", "all"], default="all")
    parser.add_argument("--train-row-cap", type=int, default=200_000)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--features-from", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.features_from:
        shutil.copyfile(args.features_from / "feature_manifest.json", args.output / "feature_manifest.json")
    factors = yaml.safe_load((Path(__file__).parents[1] / "config/ml_pool.yaml").read_text(encoding="utf-8"))["pool"]
    if args.stage in ("features", "all"):
        build_features(args.root, args.output, factors)
    if args.stage in ("evaluate", "all"):
        from mlquant.exploratory_evaluation import run_comparison
        run_comparison(args.root, args.output, train_row_cap=args.train_row_cap)
    print(json.dumps({"ok": True, "output": str(args.output), "formal": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
