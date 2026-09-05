from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd


class DataContractError(ValueError):
    """Raised when point-in-time research inputs are incomplete or invalid."""


SCHEMAS: dict[str, tuple[str, ...]] = {
    "daily": (
        "trade_date", "symbol", "open", "high", "low", "close", "volume", "amount",
    ),
    "adjustments": ("trade_date", "symbol", "adjust_factor"),
    "calendar": ("trade_date", "is_open"),
    "securities": ("symbol", "list_date", "delist_date"),
    "status": ("trade_date", "symbol", "is_st", "is_pt", "is_suspended", "limit_up", "limit_down"),
    "fundamentals": ("symbol", "stat_date", "available_date"),
    "industries": (
        "symbol", "industry_code", "industry_name", "valid_from", "valid_to", "source", "version",
    ),
    "index_members": ("index_code", "symbol", "valid_from", "valid_to", "benchmark_weight"),
}

DATE_COLUMNS = {
    "trade_date", "list_date", "delist_date", "stat_date", "available_date", "valid_from", "valid_to",
}


@dataclass(slots=True)
class AuditResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def require_ok(self) -> None:
        if not self.ok:
            raise DataContractError("; ".join(self.errors))


@dataclass(slots=True)
class EquityDataBundle:
    daily: pd.DataFrame
    adjustments: pd.DataFrame
    calendar: pd.DataFrame
    securities: pd.DataFrame
    status: pd.DataFrame
    fundamentals: pd.DataFrame
    industries: pd.DataFrame
    index_members: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)

    TABLES: ClassVar[tuple[str, ...]] = tuple(SCHEMAS)

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> EquityDataBundle:
        data_root = Path(root or os.environ.get("MLQUANT_DATA_ROOT", "")).expanduser()
        if not str(data_root):
            raise DataContractError("data root is required via --root or MLQUANT_DATA_ROOT")
        if not data_root.exists():
            raise DataContractError(f"data root does not exist: {data_root}")

        tables: dict[str, pd.DataFrame] = {}
        for name in cls.TABLES:
            path = data_root / "equity" / f"{name}.parquet"
            if not path.exists():
                tables[name] = pd.DataFrame(columns=SCHEMAS[name])
                continue
            frame = pd.read_parquet(path)
            for column in DATE_COLUMNS.intersection(frame.columns):
                frame[column] = pd.to_datetime(frame[column]).dt.normalize()
            tables[name] = frame
        meta_path = data_root / "equity" / "metadata.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(**tables, metadata=metadata)

    def audit(self, *, formal: bool = True, index_code: str | None = None) -> AuditResult:
        result = AuditResult()
        for name, required in SCHEMAS.items():
            frame = getattr(self, name)
            result.rows[name] = len(frame)
            missing = sorted(set(required) - set(frame.columns))
            if missing:
                result.errors.append(f"{name}: missing columns {missing}")
            elif frame.empty:
                result.errors.append(f"{name}: empty")

        if result.errors:
            return result
        if (self.daily[["open", "high", "low", "close"]].le(0)).any().any():
            result.errors.append("daily: non-positive price")
        invalid_ohlc = (self.daily["high"] < self.daily[["open", "close", "low"]].max(axis=1)) | (
            self.daily["low"] > self.daily[["open", "close", "high"]].min(axis=1)
        )
        if invalid_ohlc.any():
            result.errors.append("daily: invalid OHLC bounds")
        if (self.daily[["volume", "amount"]].lt(0)).any().any():
            result.errors.append("daily: negative volume or amount")
        for name, keys in {
            "daily": ["trade_date", "symbol"],
            "adjustments": ["trade_date", "symbol"],
            "status": ["trade_date", "symbol"],
        }.items():
            if getattr(self, name).duplicated(keys).any():
                result.errors.append(f"{name}: duplicate key {keys}")
        if (self.adjustments["adjust_factor"] <= 0).any():
            result.errors.append("adjustments: adjust_factor must be positive")
        self._audit_intervals(self.industries, "industries", result)
        self._audit_intervals(self.index_members, "index_members", result, key="index_code")
        if self.fundamentals["available_date"].isna().any():
            result.errors.append("fundamentals: available_date is required")
        if index_code is not None:
            selected = self.index_members[self.index_members["index_code"] == index_code]
            if selected.empty:
                result.errors.append(f"index_members: missing {index_code}")
            elif selected["benchmark_weight"].isna().any():
                result.errors.append(f"index_members: {index_code} has missing benchmark weights")
            elif (selected["benchmark_weight"] < 0).any():
                result.errors.append(f"index_members: {index_code} has negative benchmark weights")
            if formal and not selected.empty:
                self._audit_formal_history(index_code, result)
        snapshot_only = bool(self.metadata.get("industry_snapshot_only", False))
        imputed_weights = bool(self.metadata.get("benchmark_weights_imputed_equal", False))
        if formal and snapshot_only:
            result.errors.append("industries: current snapshot cannot be used for formal research")
        elif snapshot_only:
            result.warnings.append("NON-FORMAL: current industry snapshot smoke test")
        if formal and imputed_weights:
            result.errors.append("index_members: equal-imputed weights cannot be used for formal research")
        elif imputed_weights:
            result.warnings.append("NON-FORMAL: benchmark weights are equal-imputed")
        return result

    def _audit_formal_history(self, index_code: str, result: AuditResult) -> None:
        calendar = self.calendar[self.calendar["is_open"].astype(bool)].copy()
        calendar = calendar[calendar["trade_date"].between("2014-01-01", "2025-12-31")]
        if calendar.empty or calendar["trade_date"].min() > pd.Timestamp("2014-01-31") or calendar[
            "trade_date"
        ].max() < pd.Timestamp("2025-12-01"):
            result.errors.append(f"calendar: incomplete 2014-2025 formal history for {index_code}")
            return
        signals = calendar.groupby(calendar["trade_date"].dt.to_period("M"))["trade_date"].max()
        members = self.index_members[self.index_members["index_code"] == index_code]
        if not members.empty:
            # Start the PIT check at the index's first available membership date;
            # e.g. CSI 1000 (000852.SH) only has constituents from 2014-10.
            first = pd.Timestamp(members["valid_from"].min()).normalize()
            signals = signals[signals >= first]
        for date in signals:
            active = members[(members["valid_from"] <= date) & (members["valid_to"].isna() | (members["valid_to"] >= date))]
            if active.empty or active["benchmark_weight"].notna().sum() != len(active):
                result.errors.append(f"index_members: no complete PIT weights for {index_code} at {date.date()}")
                break
            industries = self.point_in_time("industries", date)
            if not set(active["symbol"]).issubset(set(industries["symbol"])):
                result.errors.append(f"industries: incomplete PIT mapping for {index_code} at {date.date()}")
                break

    @staticmethod
    def _audit_intervals(
        frame: pd.DataFrame, name: str, result: AuditResult, *, key: str | None = None
    ) -> None:
        if frame.empty or not {"valid_from", "valid_to", "symbol"}.issubset(frame.columns):
            return
        if (frame["valid_to"].notna() & (frame["valid_to"] < frame["valid_from"])).any():
            result.errors.append(f"{name}: valid_to precedes valid_from")
        keys = ["symbol"] + ([key] if key else [])
        for _, group in frame.sort_values("valid_from").groupby(keys, dropna=False):
            starts = group["valid_from"].iloc[1:].reset_index(drop=True)
            ends = group["valid_to"].iloc[:-1].reset_index(drop=True)
            if (ends.isna() | (starts <= ends)).any():
                result.errors.append(f"{name}: overlapping validity intervals")
                break

    def point_in_time(self, table: str, date: str | pd.Timestamp) -> pd.DataFrame:
        frame = getattr(self, table)
        when = pd.Timestamp(date).normalize()
        if "available_date" in frame.columns:
            visible = frame[frame["available_date"] <= when]
            if {"symbol", "stat_date"}.issubset(visible.columns):
                visible = visible.sort_values(["symbol", "stat_date", "available_date"])
                visible = visible.drop_duplicates(["symbol", "stat_date"], keep="last")
            return visible.copy()
        if {"valid_from", "valid_to"}.issubset(frame.columns):
            mask = (frame["valid_from"] <= when) & (
                frame["valid_to"].isna() | (frame["valid_to"] >= when)
            )
            return frame.loc[mask].copy()
        raise DataContractError(f"{table} is not a point-in-time table")

    def build_snapshot(self, destination: str | Path, *, formal: bool = True) -> Path:
        self.audit(formal=formal).require_ok()
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, object] = {"formal": formal, "tables": {}}
        for name in self.TABLES:
            path = target / f"{name}.parquet"
            getattr(self, name).to_parquet(path, index=False)
            manifest["tables"][name] = {
                "rows": len(getattr(self, name)),
                "sha256": sha256_file(path),
            }
        (target / "metadata.json").write_text(
            json.dumps({**self.metadata, "formal": formal}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest_path = target / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path


_QMT_DAILY_DTYPE = np.dtype(
    [
        ("date", "<u4"), ("open", "<u4"), ("high", "<u4"), ("low", "<u4"),
        ("close", "<u4"), ("pad0", "<u4"), ("volume", "<u4"), ("pad1", "<u4"),
        ("amount", "<i8"), ("rest", "V24"),
    ]
)


def read_qmt_daily_dat(path: str | Path, symbol: str) -> pd.DataFrame:
    """Read QMT's 8-byte-header, 64-byte daily format without importing xtquant."""
    source = Path(path)
    if source.stat().st_size < 8 or (source.stat().st_size - 8) % _QMT_DAILY_DTYPE.itemsize:
        raise DataContractError(f"invalid QMT DAT record size: {source}")
    records = np.fromfile(source, dtype=_QMT_DAILY_DTYPE, offset=8)
    dates = pd.to_datetime(records["date"], unit="s", utc=True).tz_convert("Asia/Shanghai").normalize()
    frame = pd.DataFrame(
        {
            "trade_date": dates.tz_localize(None),
            "symbol": symbol,
            "open": records["open"] / 1000.0,
            "high": records["high"] / 1000.0,
            "low": records["low"] / 1000.0,
            "close": records["close"] / 1000.0,
            "volume": records["volume"].astype(float),
            "amount": records["amount"].astype(float),
        }
    )
    valid = (
        frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & frame["high"].ge(frame[["open", "close", "low"]].max(axis=1))
        & frame["low"].le(frame[["open", "close", "high"]].min(axis=1))
        & frame["volume"].ge(0)
        & frame["amount"].ge(0)
    )
    return frame.loc[valid].reset_index(drop=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
