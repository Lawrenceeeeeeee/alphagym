"""Bounded imports into ClickHouse; only complete batches become visible."""
from __future__ import annotations

from pathlib import Path

from alphagym.adapters import QmtDailyAdapter, QmtDividendAdapter
from alphagym.config import resolve_root
from alphagym.equity_data import SCHEMAS, DataContractError
from alphagym.storage import KEYS
from alphagym.storage_io import import_file, register_root, store_for


def import_qmt(datadir: str | Path, root=None, *, batch_size=500):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source = Path(datadir).expanduser().resolve()
    if not source.is_dir():
        raise DataContractError(f"QMT datadir not found: {source}")
    root = register_root(resolve_root(root))
    store = store_for(root, initialize=True)
    count = 0
    with store.batch() as batch:
        adjustments = QmtDividendAdapter(source).read()
        adapter = QmtDailyAdapter(source)
        symbols = adapter.symbols()
        for start in range(0, len(symbols), batch_size):
            frame = adapter.read(symbols[start:start + batch_size])
            if frame.empty:
                continue
            missing = set(SCHEMAS["daily"]) - set(frame)
            if missing:
                raise DataContractError(f"QMT daily missing fields: {sorted(missing)}")
            batch.frame("equity/daily.parquet", frame, mode="upsert", keys=KEYS["daily"])
            count += len(frame)
        if not count:
            raise DataContractError("QMT source contains no valid daily records")
        batch.frame("equity/adjustments.parquet", adjustments,
                    mode="upsert", keys=KEYS["adjustments"])
    return {"backend": "clickhouse", "path": "equity", "version": batch.version,
            "daily_rows": count, "adjustment_rows": len(adjustments)}


def import_parquet(path, table, root=None, *, mode="upsert", batch_size=65536):
    import pyarrow.parquet as pq

    if table not in SCHEMAS:
        raise DataContractError(f"Unknown equity table: {table}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source = Path(path).expanduser().resolve()
    schema = pq.ParquetFile(source).schema_arrow
    missing = set(SCHEMAS[table]) - set(schema.names)
    if missing:
        raise DataContractError(f"{table}: missing fields: {sorted(missing)}")
    target = register_root(resolve_root(root)) / "equity" / f"{table}.parquet"
    return import_file(source, target, mode=mode, keys=KEYS[table], batch_size=batch_size)
