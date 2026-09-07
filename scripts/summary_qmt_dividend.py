"""Diagnostic: summarize a QMT DividData LevelDB (record count, factor range)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alphagym.qmt_dividend import read_dividend_events


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datadir")
    args = ap.parse_args()
    events = read_dividend_events(Path(args.datadir))
    print(f"events={len(events)} symbols={events['symbol'].nunique()}", file=sys.stderr)
    print(f"ex_date range: {events['ex_date'].min().date()} .. {events['ex_date'].max().date()}")
    print(f"factor range: {events['factor'].min():.6f} .. {events['factor'].max():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
