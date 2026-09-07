from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alphagym.equity_data import SCHEMAS, DataContractError, EquityDataBundle, read_qmt_daily_dat
from alphagym.qmt_dividend import build_adjustments, read_dividend_events


def _is_a_share(number: str, exchange: str) -> bool:
    prefixes = {
        "SH": ("600", "601", "603", "605", "688", "689"),
        "SZ": ("000", "001", "002", "003", "300", "301"),
        "BJ": ("4", "8"),
    }
    return len(number) == 6 and number.isdigit() and number.startswith(prefixes[exchange])


@dataclass(frozen=True, slots=True)
class QmtDailyAdapter:
    """Cross-platform reader for already-downloaded QMT fixed records; no xtquant dependency."""

    datadir: Path

    def symbols(self) -> list[str]:
        """Enumerate every downloaded daily series as ``code.SH``/``code.SZ`` style symbols."""
        result: list[str] = []
        for exchange in ("SH", "SZ", "BJ"):
            folder = self.datadir / exchange / "86400"
            if folder.is_dir():
                for path in sorted(folder.glob("*.DAT")):
                    if _is_a_share(path.stem, exchange):
                        result.append(f"{path.stem}.{exchange}")
        return result

    def read(self, symbols: list[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            number, exchange = symbol.split(".")
            path = self.datadir / exchange.upper() / "86400" / f"{number}.DAT"
            if not path.exists():
                continue
            try:
                frames.append(read_qmt_daily_dat(path, symbol))
            except DataContractError as error:
                print(f"skip {symbol}: {error}", file=sys.stderr)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SCHEMAS["daily"])


@dataclass(frozen=True, slots=True)
class QmtDividendAdapter:
    """Read QMT's DividData LevelDB into cumulative backward-adjustment factors."""

    datadir: Path

    def read(self) -> pd.DataFrame:
        """Return ``trade_date``, ``symbol``, ``adjust_factor`` (adj_close = close * factor)."""
        frame = build_adjustments(read_dividend_events(self.datadir / "DividData"))
        valid = frame["symbol"].map(
            lambda symbol: _is_a_share(str(symbol)[:6], str(symbol)[-2:])
        )
        return frame.loc[valid].reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class ParquetBundleAdapter:
    """Normalize vendor-independent Parquet inputs into the canonical equity contract."""

    paths: dict[str, Path]

    def load(self, *, metadata: dict[str, object] | None = None) -> EquityDataBundle:
        tables: dict[str, pd.DataFrame] = {}
        for table, required in SCHEMAS.items():
            if table not in self.paths:
                raise DataContractError(f"missing parquet mapping for {table}")
            frame = pd.read_parquet(self.paths[table]).rename(
                columns={"date": "trade_date", "code": "symbol", "avail_date": "available_date"}
            )
            missing = sorted(set(required) - set(frame.columns))
            if missing:
                raise DataContractError(f"{table}: missing columns after import {missing}")
            tables[table] = frame
        return EquityDataBundle(**tables, metadata=metadata or {})
