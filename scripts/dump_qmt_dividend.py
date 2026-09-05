"""Diagnostic: raw-dump key/value pairs from a QMT DividData LevelDB directory."""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

from mlquant.qmt_dividend import iter_leveldb_records, parse_dividend_key


def _printable(b: bytes, limit: int = 160) -> str:
    return "".join(chr(c) if 32 <= c < 127 else "." for c in b[:limit])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datadir")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    for index, (key, value) in enumerate(iter_leveldb_records(Path(args.datadir))):
        if index >= args.limit:
            break
        parsed = parse_dividend_key(key)
        label = f"{parsed[0]}|{parsed[1]}|{parsed[2]}|{parsed[3]}" if parsed else "?"
        print(f"KEY   {label}  tail={key[-8:].hex()}")
        print(f"VALUE ({len(value)}B) {_printable(value)}")
        if len(value) % 8 == 0:
            n = len(value) // 8
            f64 = [round(struct.unpack_from("<d", value, i * 8)[0], 6) for i in range(n)]
            print(f"      f64 {f64}")
        print("-" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
