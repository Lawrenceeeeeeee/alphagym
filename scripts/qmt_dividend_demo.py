"""Smoke demo: extract QMT backward-adjustment factors and show raw-vs-adjusted continuity.

Usage:
    python scripts/qmt_dividend_demo.py <QMT_datadir> [--min-factor 1.5] [--window 4]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alphagym.equity_data import read_qmt_daily_dat
from alphagym.qmt_dividend import build_adjustments, read_dividend_events


def daily_frame(datadir: Path, symbol: str) -> pd.DataFrame:
    number, exchange = symbol.split(".")
    path = datadir / exchange.upper() / "86400" / f"{number}.DAT"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_qmt_daily_dat(path, symbol)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datadir", type=Path)
    ap.add_argument("--min-factor", type=float, default=1.5)
    ap.add_argument("--window", type=int, default=4)
    args = ap.parse_args()

    events = read_dividend_events(args.datadir / "DividData")
    # plain A-share equity codes only, avoid the 2026 exclusion window, skip fund restructures
    a_share = events[
        events["symbol"].str.match(r"^(000|001|002|003|300|301)\d{3}\.SZ$|^(600|601|603|605|688)\d{3}\.SH$")
        & (events["ex_date"] < "2026-01-01")
        & (events["factor"].between(args.min_factor, 5.0))
    ].sort_values(["factor", "ex_date"], ascending=[False, False])
    if a_share.empty:
        print("no qualifying event")
        return 1
    pick = None
    for _, row in a_share.iterrows():
        number, exchange = row["symbol"].split(".")
        if (args.datadir / exchange.upper() / "86400" / f"{number}.DAT").exists():
            pick = row
            break
    if pick is None:
        print("no qualifying event with downloaded daily data")
        return 1
    symbol, ex_date = pick["symbol"], pick["ex_date"]
    print(f"picked {symbol} event factor={pick['factor']:.4f} ex_date={ex_date.date()}")

    daily = daily_frame(args.datadir, symbol)
    adj = build_adjustments(events)
    stock_adj = adj[adj["symbol"] == symbol]
    # cumulative factor valid on/after each ex_date, forward-filled across the calendar
    merged = daily.merge(
        stock_adj, left_on="trade_date", right_on="trade_date", how="left"
    ).sort_values("trade_date")
    merged["adjust_factor"] = merged["adjust_factor"].ffill().fillna(1.0)
    merged["adj_close"] = merged["close"] * merged["adjust_factor"]

    window = merged[
        (merged["trade_date"] >= ex_date - pd.Timedelta(days=args.window * 2))
        & (merged["trade_date"] <= ex_date + pd.Timedelta(days=args.window * 2))
    ]
    cols = ["trade_date", "close", "adjust_factor", "adj_close"]
    print(window[cols].to_string(index=False))
    # returns across the ex-date (raw should jump, adjusted should be smooth)
    merged["raw_ret"] = merged["close"].pct_change()
    merged["adj_ret"] = merged["adj_close"].pct_change()
    print("\nreturn on ex_date: raw={:.4%} adjusted={:.4%}".format(
        merged.loc[merged["trade_date"] == ex_date, "raw_ret"].iloc[0],
        merged.loc[merged["trade_date"] == ex_date, "adj_ret"].iloc[0],
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
