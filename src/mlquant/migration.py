"""Explicit, resumable migration. Source files and SQLite catalogs remain untouched."""
from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd

from mlquant.factor_store import FactorStore
from mlquant.storage import KEYS, StorageError
from mlquant.storage_io import import_file, register_root, store_for


def _rebase(value, source, root):
    if isinstance(value, dict):
        return {key: json.dumps(_rebase(json.loads(item), source, root), ensure_ascii=False)
                if key.endswith("_json") and isinstance(item, str)
                else _rebase(item, source, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebase(item, source, root) for item in value]
    if isinstance(value, str):
        for prefix in (str(source), source.as_posix()):
            if value.startswith((prefix + "/", prefix + "\\")):
                return str(root / value[len(prefix) + 1:])
    return value


def migrate_workspace(source, root):
    source = Path(source).expanduser().resolve()
    root = register_root(root)
    if not source.is_dir():
        raise FileNotFoundError(source)
    store = store_for(root, initialize=True)
    manifests = {item["path"]: item for item in store.list()}
    metadata_path = source / "equity" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    units = metadata.get("units") or {}
    scale = {
        "volume": 100.0 if units.get("volume") == "手" else 1.0,
        "amount": 1000.0 if units.get("amount") == "千元" else 1.0,
        "turnover": 0.01 if units.get("turnover") == "%" else 1.0,
        "float_market_cap": 10000.0 if units.get("float_market_cap") == "万元" else 1.0,
    }

    def normalize_equity(frame):
        for column, multiplier in scale.items():
            if column in frame and multiplier != 1.0:
                frame[column] = pd.to_numeric(frame[column], errors="raise") * multiplier
        return frame
    files = []
    for directory in (
        "equity", "factor_library", "artifacts", "backtests", "experiments", "normalized",
        "snapshots", "qmt_signals", "audits", "models",
    ):
        parent = source / directory
        if parent.is_dir():
            files.extend(p for p in parent.rglob("*") if p.is_file() and p.suffix.lower() in {
                ".parquet", ".csv", ".json", ".yaml", ".html", ".md", ".png", ".pdf", ".bin",
                ".pkl", ".joblib", ".js", ".npz", ".npy", ".pt", ".safetensors",
            })
    migrated = skipped = 0
    for path in sorted(files):
        relative = path.relative_to(source).as_posix()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        checkpoint = "migration/files/" + hashlib.sha256(relative.encode()).hexdigest() + ".json"
        if checkpoint in manifests:
            prior = json.loads(base64.b64decode(manifests[checkpoint]["payload"]))
            if prior["sha256"] == digest and relative in manifests:
                skipped += 1
                continue
        if relative in manifests:
            raise StorageError(f"Migration would replace existing data: {relative}; use a new workspace")
        receipt = json.dumps({"path": relative, "sha256": digest}).encode()
        if path.suffix == ".parquet":
            import_file(path, root / relative,
                        keys=KEYS.get(path.stem) if relative.startswith("equity/") else None,
                        checkpoint=(checkpoint, receipt),
                        transform=normalize_equity if relative == "equity/daily.parquet" else None)
        else:
            with store.batch() as batch:
                if path.suffix == ".csv":
                    try:
                        frame = pd.read_csv(path)
                    except pd.errors.EmptyDataError:
                        frame = pd.DataFrame()
                    batch.frame(relative, frame)
                else:
                    payload = path.read_bytes()
                    if relative == "equity/metadata.json" and any(v != 1.0 for v in scale.values()):
                        normalized = {**metadata, "legacy_units": units, "units": {
                            "volume": "股", "amount": "元", "turnover": "ratio",
                            "float_market_cap": "元",
                        }, "storage_unit_normalization": "clickhouse_v1"}
                        payload = json.dumps(normalized, ensure_ascii=False, indent=2).encode()
                    batch.blob(relative, payload)
                batch.blob(checkpoint, receipt)
        migrated += 1
    old_catalog = source / "factor_library" / "catalog.sqlite"
    catalog_rows = 0
    catalog_receipt = manifests.get("migration/catalog.json")
    if old_catalog.is_file() and (not catalog_receipt or catalog_receipt["kind"] == "deleted"):
        with old_catalog.open("rb") as stream:
            catalog_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        with FactorStore.from_root(root) as catalog:
            receipt = catalog.connection.execute(
                "SELECT rows_imported FROM catalog_migration WHERE source_sha256=?", (catalog_digest,),
            ).fetchone()
            if receipt:
                store.write_blob("migration/catalog.json", json.dumps({"rows": receipt[0]}).encode())
                return {"backend": "clickhouse", "files_imported": migrated, "files_skipped": skipped,
                        "catalog_rows": receipt[0], "source_preserved": True}
            if catalog.list_factors() or catalog.list_runs():
                raise StorageError("Target catalog is not empty; migrate into a new workspace")
            legacy = sqlite3.connect(old_catalog.as_uri() + "?mode=ro", uri=True)
            legacy.row_factory = sqlite3.Row
            try:
                with catalog.connection:
                    catalog.connection.execute("PRAGMA defer_foreign_keys=ON")
                    for name in catalog.connection.tables:
                        if name == "catalog_migration":
                            continue
                        if not legacy.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone():
                            continue
                        for row in legacy.execute(f'SELECT * FROM "{name}"'):
                            data = dict(row)
                            for key, value in data.items():
                                if key == "path" and value:
                                    old = Path(value)
                                    if old.is_absolute():
                                        if not old.is_relative_to(source):
                                            raise StorageError(f"Catalog artifact outside source root: {old.name}")
                                        data[key] = str(root / old.relative_to(source))
                            data = _rebase(data, source, root)
                            columns = list(data)
                            catalog.connection.execute(
                                f'INSERT OR REPLACE INTO "{name}" ({",".join(columns)}) '
                                f'VALUES ({",".join("?" for _ in columns)})', list(data.values()),
                            )
                            catalog_rows += 1
                    catalog.connection.execute(
                        "INSERT INTO catalog_migration VALUES (?, ?)", (catalog_digest, catalog_rows),
                    )
            finally:
                legacy.close()
        store.write_blob("migration/catalog.json", json.dumps({"rows": catalog_rows}).encode())
    return {"backend": "clickhouse", "files_imported": migrated, "files_skipped": skipped,
            "catalog_rows": catalog_rows, "source_preserved": True}
