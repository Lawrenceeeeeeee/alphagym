from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from mlquant.equity_data import EquityDataBundle
from mlquant.factors import REGISTRY
from mlquant.reporting import build_series
from mlquant.research import huatai_industry_layers


def synthetic_bundle() -> EquityDataBundle:
    dates = pd.bdate_range("2024-01-02", periods=260)
    symbols = [f"{index:06d}.SZ" for index in range(1, 31)]
    daily = pd.DataFrame(
        [
            {
                "trade_date": date, "symbol": symbol, "open": 10 + number / 10,
                "high": 10.2 + number / 10, "low": 9.8 + number / 10,
                "close": 10 + number / 10, "adj_close": 10 + number / 10 + day / 1000,
                "volume": 10_000, "amount": 1_000_000, "turnover": .01,
                "float_market_cap": 1e9 * (number + 1),
            }
            for day, date in enumerate(dates)
            for number, symbol in enumerate(symbols)
        ]
    )
    indices = ["000300.SH", "000905.SH", "000852.SH"]
    return EquityDataBundle(
        daily=daily,
        adjustments=pd.DataFrame([{"trade_date": dates[0], "symbol": s, "adjust_factor": 1.0} for s in symbols]),
        calendar=pd.DataFrame({"trade_date": dates, "is_open": True}),
        securities=pd.DataFrame({"symbol": symbols, "list_date": pd.Timestamp("2010-01-01"), "delist_date": pd.NaT}),
        status=pd.DataFrame([{"trade_date": d, "symbol": s, "is_st": False, "is_pt": False,
                             "is_suspended": False, "limit_up": 20.0, "limit_down": 1.0}
                            for d in dates for s in symbols]),
        fundamentals=pd.DataFrame([{"symbol": s, "stat_date": pd.Timestamp("2023-12-31"),
                                    "available_date": pd.Timestamp("2024-04-30"), "roe": 10.0,
                                    "gross_margin": 30.0, "eps": 1.0, "ocfps": 1.2,
                                    "bps": 5.0, "net_profit": 1e8, "revenue": 1e9}
                                   for s in symbols]),
        industries=pd.DataFrame({"symbol": symbols, "industry_code": np.repeat(["A", "B", "C"], 10),
                                 "industry_name": np.repeat(["行业A", "行业B", "行业C"], 10),
                                 "valid_from": pd.Timestamp("2010-01-01"), "valid_to": pd.NaT,
                                 "source": "synthetic", "version": "smoke"}),
        index_members=pd.DataFrame([{"index_code": index, "symbol": symbol,
                                     "valid_from": pd.Timestamp("2010-01-01"), "valid_to": pd.NaT,
                                     "benchmark_weight": 1 / len(symbols)}
                                    for index in indices for symbol in symbols]),
        metadata={"industry_snapshot_only": True},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix="mlquant-equity-smoke-"))
    bundle = synthetic_bundle()
    for index in ("000300.SH", "000905.SH", "000852.SH"):
        bundle.audit(formal=False, index_code=index).require_ok()
    assert len(REGISTRY) == 169
    layer_input = pd.DataFrame({"symbol": bundle.securities["symbol"],
                                "industry_code": bundle.industries["industry_code"],
                                "factor_value": np.arange(len(bundle.securities)),
                                "benchmark_weight": 1 / len(bundle.securities)})
    layers = huatai_industry_layers(layer_input)
    assert np.allclose(layers.groupby("layer")["target_weight"].sum(), 1)
    manifest = build_series(output, smoke=True, metadata={"indices": 3})
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
