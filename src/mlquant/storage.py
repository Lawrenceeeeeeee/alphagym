"""ClickHouse persistence for versioned datasets and precomputed artifacts.

Writes stage immutable rows, then publish one small manifest block. Failed writes
are invisible. Historical manifests and rows are retained for reproducible reads.
The filesystem root identifies deployment configuration, never a data backend.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml


class StorageError(ValueError):
    """A safe, user-facing database error (connection secrets are never echoed)."""


def ident(value: str) -> str:
    return "`" + value.replace("\\", "\\\\").replace("`", "\\`") + "`"


@dataclass(frozen=True)
class ClickHouseConfig:
    host: str = "localhost"
    port: int = 8123
    username: str = "default"
    password: str = field(default="", repr=False)
    database: str = "mlquant"
    secure: bool = False
    workspace: str = "default"

    @classmethod
    def load(cls, root: Path) -> ClickHouseConfig:
        path = root / "storage.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
        values = dict((data or {}).get("clickhouse", {}))
        values.setdefault("workspace", "ws_" + hashlib.sha256(str(root).encode()).hexdigest()[:16])
        if "password" in values:
            raise StorageError("Use MLQUANT_CLICKHOUSE_PASSWORD, not a password in storage.yaml")
        for name in ("host", "port", "username", "password", "database", "secure", "workspace"):
            value = os.environ.get("MLQUANT_CLICKHOUSE_" + name.upper())
            if value is not None:
                values[name] = value
        # Keep compatibility with the existing Docker deployment variable.
        if "MLQUANT_CLICKHOUSE_USERNAME" not in os.environ and os.environ.get("MLQUANT_CLICKHOUSE_USER"):
            values["username"] = os.environ["MLQUANT_CLICKHOUSE_USER"]
        if "port" in values:
            values["port"] = int(values["port"])
        if "secure" in values:
            values["secure"] = str(values["secure"]).lower() in {"true", "1", "yes"}
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", values.get("database", "mlquant")):
            raise StorageError("Invalid ClickHouse database name")
        return cls(**values)


KEYS = {
    "daily": ("symbol", "trade_date"),
    "adjustments": ("symbol", "trade_date"),
    "calendar": ("trade_date",),
    "securities": ("symbol",),
    "status": ("symbol", "trade_date"),
    "fundamentals": ("symbol", "stat_date", "available_date"),
    "industries": ("symbol", "valid_from"),
    "index_members": ("index_code", "symbol", "valid_from"),
}


def _ch_type(dtype: pa.DataType) -> str:
    if pa.types.is_dictionary(dtype):
        return _ch_type(dtype.value_type)
    if pa.types.is_boolean(dtype):
        return "Bool"
    if pa.types.is_integer(dtype):
        return ("UInt" if pa.types.is_unsigned_integer(dtype) else "Int") + str(dtype.bit_width)
    if pa.types.is_floating(dtype):
        return "Float64" if dtype.bit_width == 64 else "Float32"
    if pa.types.is_timestamp(dtype):
        return "DateTime64(9, 'UTC')"
    if pa.types.is_date(dtype):
        return "Date32"
    if pa.types.is_decimal(dtype):
        return f"Decimal({dtype.precision}, {dtype.scale})"
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype) or pa.types.is_null(dtype):
        return "String"
    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        return "String"
    raise StorageError(f"Unsupported dataset field type: {dtype}")


class ClickHouseStore:
    def __init__(self, root: str | Path, *, client=None, initialize: bool = False):
        self.root = Path(root).expanduser().resolve()
        self.config = ClickHouseConfig.load(self.root)
        self.client = client
        if client is None:
            import clickhouse_connect

            try:
                self.client = clickhouse_connect.get_client(
                    host=self.config.host, port=self.config.port,
                    username=self.config.username, password=self.config.password,
                    secure=self.config.secure, database="default",
                    connect_timeout=5, send_receive_timeout=120,
                )
            except Exception:  # noqa: BLE001 -- redact transport messages containing credentials
                raise StorageError(
                    "Cannot connect to ClickHouse; check storage.yaml, service and credentials"
                ) from None
        if initialize:
            self.initialize()

    def table(self, name: str) -> str:
        return f"{ident(self.config.database)}.{ident(name)}"

    def initialize(self):
        self.client.command(f"CREATE DATABASE IF NOT EXISTS {ident(self.config.database)}")
        self.client.command(f"""CREATE TABLE IF NOT EXISTS {self.table('resources')} (
            workspace String, path String, version UInt64, batch String,
            kind String, physical String, schema String, mode String,
            rows UInt64, sha256 String, payload String
        ) ENGINE=MergeTree ORDER BY (workspace, path, version, batch)""")
        config_path = self.root / "storage.yaml"
        if not config_path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump({"clickhouse": {
                "host": self.config.host, "port": self.config.port,
                "username": self.config.username, "database": self.config.database,
                "workspace": self.config.workspace, "secure": self.config.secure,
            }}, sort_keys=False), encoding="utf-8")

    def available(self) -> bool:
        return bool(self.client.command(f"EXISTS TABLE {self.table('resources')}"))

    def manifest(self, path: str, *, as_of: int | None = None) -> dict | None:
        if not self.available():
            return None
        rows = self.client.query(
            f"SELECT * FROM {self.table('resources')} WHERE workspace=%(w)s AND path=%(p)s "
            "AND version<=%(v)s ORDER BY version DESC, batch DESC LIMIT 1",
            parameters={"w": self.config.workspace, "p": path, "v": as_of if as_of is not None else 2**64 - 1},
        ).named_results()
        return next(iter(rows), None)

    def list(self, prefix: str = "") -> list[dict]:
        if not self.available():
            return []
        return list(self.client.query(
            f"SELECT * FROM {self.table('resources')} WHERE workspace=%(w)s "
            "AND startsWith(path, %(p)s) ORDER BY version DESC, batch DESC LIMIT 1 BY path",
            parameters={"w": self.config.workspace, "p": prefix},
        ).named_results())

    @contextmanager
    def batch(self):
        batch = WriteBatch(self)
        yield batch
        batch.commit()

    @contextmanager
    def publication_lock(self):
        lock = self.table("data_publish_lock")
        deadline = time.monotonic() + 15
        while True:
            try:
                self.client.command(f"CREATE TABLE {lock} (owner String) ENGINE=MergeTree ORDER BY tuple()")
                break
            except Exception as error:  # noqa: BLE001 -- redact driver errors
                if "TABLE_ALREADY_EXISTS" not in str(error):
                    raise StorageError("Cannot acquire data publication lock") from None
                if time.monotonic() >= deadline:
                    raise StorageError("Data publisher busy; check active writers before recovering a stale lock") from None
                time.sleep(.05)
        try:
            yield
        finally:
            self.client.command(f"DROP TABLE {lock} SYNC")

    def watermark(self, *, prefix=""):
        if not self.available():
            return 0
        return int(self.client.command(
            f"SELECT max(version) FROM {self.table('resources')} WHERE workspace=%(w)s AND startsWith(path, %(prefix)s)",
            parameters={"w": self.config.workspace, "prefix": prefix},
        ))

    def write_frame(self, path: str, frame: pd.DataFrame, *, mode="replace", keys=None,
                    index: bool | None = False) -> dict:
        with self.batch() as batch:
            result = batch.frame(path, frame, mode=mode, keys=keys, index=index)
        return result

    def _frame_query(self, path: str, *, columns=None, filters=None, as_of=None):
        meta = self.manifest(path, as_of=as_of)
        if meta is None or meta["kind"] != "frame":
            raise FileNotFoundError(f"ClickHouse dataset missing: {path}")
        schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(meta["schema"])))
        select = list(schema.names) if columns is None else list(columns)
        if not schema.names:
            if select or filters:
                raise StorageError(f"Empty dataset {path} has no selectable columns")
            return None, {}, schema
        index_columns = []
        if schema.metadata and b"pandas" in schema.metadata:
            index_columns = [c for c in json.loads(schema.metadata[b"pandas"])["index_columns"]
                             if isinstance(c, str)]
        for col in index_columns:
            if col not in select:
                select.append(col)
        if set(select) - set(schema.names):
            raise StorageError(f"Missing columns in {path}: {sorted(set(select) - set(schema.names))}")
        params = {"w": self.config.workspace, "p": path, "v": meta["version"]}
        # A replace starts a new dataset generation; later upserts overlay by business key.
        base = self.client.command(
            f"SELECT max(version) FROM {self.table('resources')} WHERE workspace=%(w)s "
            "AND path=%(p)s AND mode='replace' AND version<=%(v)s", parameters=params,
        )
        params["base"] = base
        where = ""
        if filters:
            groups = filters if isinstance(filters[0], list) else [filters]
            clauses = []
            for group in groups:
                parts = []
                for col, op, value in group:
                    if col not in schema.names or op not in {"=", "==", "!=", ">", ">=", "<", "<=", "in", "not in"}:
                        raise StorageError("Unsupported dataset filter")
                    key = f"f{len(params)}"
                    if isinstance(value, pd.Timestamp):
                        value = (value.tz_localize("UTC") if value.tzinfo is None else value).to_pydatetime()
                    params[key] = tuple(value) if op in {"in", "not in"} else value
                    parts.append(f"{ident(col)} {'=' if op == '==' else op} %({key})s")
                clauses.append("(" + " AND ".join(parts) + ")")
            where = " WHERE " + " OR ".join(clauses)
        generations = list(self.client.query(
            f"SELECT physical, schema FROM {self.table('resources')} WHERE workspace=%(w)s "
            "AND path=%(p)s AND kind='frame' AND version BETWEEN %(base)s AND %(v)s "
            "ORDER BY version DESC LIMIT 1 BY physical", parameters=params,
        ).named_results())
        sources = []
        for generation in generations:
            old_schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(generation["schema"])))
            expressions = [f"CAST({ident(c)} AS Nullable({_ch_type(schema.field(c).type)})) AS {ident(c)}" if c in old_schema.names else
                           f"CAST(NULL AS Nullable({_ch_type(schema.field(c).type)})) AS {ident(c)}"
                           for c in schema.names]
            sources.append(f"SELECT {', '.join(expressions)}, _batch, _row, _key "
                           f"FROM {self.table(generation['physical'])}")
        query = f"""SELECT {', '.join(map(ident, select))} FROM (
            SELECT d.*, r.committed_version AS _commit FROM ({' UNION ALL '.join(sources)}) d
            INNER JOIN (SELECT batch, max(version) AS committed_version FROM {self.table('resources')}
                WHERE workspace=%(w)s AND path=%(p)s AND version BETWEEN %(base)s AND %(v)s
                GROUP BY batch) r ON d._batch=r.batch
            ORDER BY _commit DESC, _row DESC LIMIT 1 BY _key
        ){where} ORDER BY _commit, _row"""
        target = pa.schema([schema.field(c) for c in select], metadata=schema.metadata)
        return query, params, target

    def read_frame(self, path: str, *, columns=None, filters=None, as_of=None) -> pd.DataFrame:
        query, params, target = self._frame_query(path, columns=columns, filters=filters, as_of=as_of)
        if query is None:
            return target.empty_table().to_pandas()
        arrow = self.client.query_arrow(query, parameters=params)
        # Preserve pandas index/dtypes rather than accepting driver's inference.
        try:
            return arrow.cast(target).to_pandas()
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            return arrow.replace_schema_metadata(target.metadata).to_pandas()

    def iter_batches(self, path, *, columns=None, as_of=None, batch_size=65536):
        query, params, schema = self._frame_query(path, columns=columns, as_of=as_of)
        if query is None:
            return
        with self.client.query_arrow_stream(query, parameters=params,
                                            settings={"max_block_size": batch_size}) as stream:
            for block in stream:
                yield block.cast(schema)

    def count_frame(self, path, *, as_of=None):
        query, params, _ = self._frame_query(path, as_of=as_of)
        if query is None:
            return 0
        return int(self.client.command(f"SELECT count() FROM ({query})", parameters=params))

    def read_blob(self, path: str, *, as_of=None) -> bytes:
        meta = self.manifest(path, as_of=as_of)
        if meta is None or meta["kind"] != "blob":
            raise FileNotFoundError(path)
        return base64.b64decode(meta["payload"])

    def write_blob(self, path: str, data: bytes):
        with self.batch() as batch:
            batch.blob(path, data)

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class WriteBatch:
    def __init__(self, store: ClickHouseStore):
        self.store = store
        self.version = time.time_ns()
        self.batch_id = uuid.uuid4().hex
        self.entries: dict[str, dict[str, Any]] = {}
        self.offsets: dict[str, int] = {}

    def frame(self, path, frame, *, mode="replace", keys=None, index=False, schema=None):
        frame = frame.copy()
        for column in frame.columns:
            if pd.api.types.is_datetime64_dtype(frame[column].dtype):
                frame[column] = frame[column].astype("datetime64[ns]")
            elif isinstance(frame[column].dtype, pd.DatetimeTZDtype):
                frame[column] = frame[column].dt.tz_convert("UTC").astype("datetime64[ns, UTC]")
            elif frame[column].dtype == object:
                values = frame[column].dropna()
                if len(values) and isinstance(values.iloc[0], (dict, list, tuple, np.ndarray)):
                    frame[column] = frame[column].map(
                        lambda value: json.dumps(
                            value.tolist() if isinstance(value, np.ndarray) else value,
                            ensure_ascii=False, sort_keys=True,
                        ) if isinstance(value, (dict, list, tuple, np.ndarray)) else value
                    )
        if path.startswith("equity/") and path.endswith(".parquet"):
            from mlquant.equity_data import DATE_COLUMNS, SCHEMAS

            name = path.split("/")[-1].removesuffix(".parquet")
            if name in SCHEMAS:
                keys = keys or KEYS[name]
                for column in set(frame.columns) & DATE_COLUMNS:
                    frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
                for column in frame:
                    if column in DATE_COLUMNS or column in {"symbol", "index_code", "industry_code", "industry_name", "source", "version"}:
                        continue
                    if column not in {"is_open", "is_st", "is_pt", "is_suspended", "limit_up", "limit_down"}:
                        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
                if name == "daily":
                    for column in ("turnover", "float_market_cap"):
                        if column not in frame:
                            frame[column] = float("nan")
                frame = frame.reindex(columns=[c for c in SCHEMAS[name] if c in frame] +
                                      sorted(set(frame) - set(SCHEMAS[name])))
        if mode not in {"replace", "upsert"}:
            raise StorageError("Write mode must be replace or upsert")
        if not frame.columns.is_unique or any(str(c).startswith("_") for c in frame.columns):
            raise StorageError("Dataset columns must be unique and cannot start with underscore")
        if keys and (set(keys) - set(frame.columns) or frame[list(keys)].isna().any().any()):
            raise StorageError("Dataset key columns are missing or contain nulls")
        arrow = pa.Table.from_pandas(frame, preserve_index=index, schema=schema)
        # Null fields have a stable nullable string representation until a typed value arrives.
        previous = self.store.manifest(path)
        if mode == "upsert" and previous and previous["kind"] == "frame":
            old = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(previous["schema"])))
            try:
                union = pa.unify_schemas([old, arrow.schema], promote_options="permissive")
            except pa.ArrowInvalid as error:
                raise StorageError(f"Incompatible schema for {path}: {error}") from error
            arrow = pa.Table.from_arrays([
                arrow[f.name].cast(f.type) if f.name in arrow.schema.names else pa.nulls(len(frame), f.type)
                for f in union
            ], schema=union.with_metadata(arrow.schema.metadata))
        schema = arrow.schema
        schema_bytes = schema.serialize().to_pybytes()
        signature = hashlib.sha256(schema.remove_metadata().serialize().to_pybytes()).hexdigest()[:16]
        physical = "data_" + hashlib.sha256(
            (self.store.config.workspace + "/" + path).encode()
        ).hexdigest()[:20] + "_" + signature
        entry = self.entries.get(path)
        if entry and entry["physical"] != physical:
            raise StorageError(f"Inconsistent schema within batch: {path}")
        size = len(frame)
        if not schema.names and size:
            raise StorageError("A dataset with rows must contain at least one column")
        if schema.names:
            cols = ", ".join(f"{ident(f.name)} Nullable({_ch_type(f.type)})" for f in schema)
            self.store.client.command(f"""CREATE TABLE IF NOT EXISTS {self.store.table(physical)} (
                {cols}, _batch String, _version UInt64, _row UInt64, _key String
            ) ENGINE=MergeTree ORDER BY (_key, _version, _row)""")
        else:
            physical = ""
        offset = self.offsets.get(path, 0)
        row_numbers = list(range(offset, offset + size))
        if keys:
            key_values = frame[keys[0]].astype(str)
            for key_column in keys[1:]:
                key_values = key_values.str.cat(frame[key_column].astype(str), sep="\x1f")
            key_values = key_values.tolist()
        else:
            key_values = [str(n) for n in row_numbers]
        for name, values, dtype in (
            ("_batch", [self.batch_id] * size, pa.string()),
            ("_version", [self.version] * size, pa.uint64()),
            ("_row", row_numbers, pa.uint64()), ("_key", key_values, pa.string()),
        ):
            arrow = arrow.append_column(name, pa.array(values, type=dtype))
        if size:
            self.store.client.insert_arrow(self.store.table(physical), arrow)
        self.offsets[path] = offset + size
        entry = {
            "workspace": self.store.config.workspace, "path": path, "version": self.version,
            "batch": self.batch_id, "kind": "frame", "physical": physical,
            "schema": base64.b64encode(schema_bytes).decode(), "mode": mode,
            "rows": offset + size, "sha256": "", "payload": "",
        }
        self.entries[path] = entry
        return entry

    def blob(self, path: str, data: bytes):
        self.entries[path] = {
            "workspace": self.store.config.workspace, "path": path, "version": self.version,
            "batch": self.batch_id, "kind": "blob", "physical": "", "schema": "",
            "mode": "replace", "rows": 0, "sha256": hashlib.sha256(data).hexdigest(),
            "payload": base64.b64encode(data).decode(),
        }

    def commit(self):
        if self.entries:
            with self.store.publication_lock():
                self.version = max(time.time_ns(), self.store.watermark() + 1)
                for row in self.entries.values():
                    row["version"] = self.version
                    current = self.store.manifest(row["path"])
                    if row["mode"] == "upsert" and current and current["kind"] == "frame":
                        current_schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(current["schema"])))
                        new_schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(row["schema"])))
                        if set(current_schema.names) - set(new_schema.names):
                            raise StorageError("Concurrent schema change; retry after schema migration")
                columns = list(next(iter(self.entries.values())))
                self.store.client.insert(self.store.table("resources"),
                                         [[row[c] for c in columns] for row in self.entries.values()],
                                         column_names=columns)
