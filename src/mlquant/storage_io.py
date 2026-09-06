"""Logical artifact references used by legacy research code.

Paths inside a registered workspace are portable ClickHouse resource keys. Their
historical suffixes remain for API compatibility; no Parquet/CSV data is written
there. Explicit imports/exports are the only filesystem table interchange.
"""
from __future__ import annotations

import fnmatch
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from mlquant.storage import ClickHouseStore

_roots: set[Path] = set()
_local = threading.local()
_snapshots = ContextVar("mlquant_data_snapshots", default=MappingProxyType({}))


@contextmanager
def snapshot(root, version=None):
    root = register_root(root)
    existing = _snapshots.get()
    if version is None:
        version = existing[root] if root in existing else store_for(root).watermark(prefix="equity/")
    token = _snapshots.set({**existing, root: int(version)})
    try:
        yield version
    finally:
        _snapshots.reset(token)


def current_version(root):
    root = register_root(root)
    versions = _snapshots.get()
    return versions[root] if root in versions else store_for(root).watermark(prefix="equity/")


def freeze_market(function):
    @wraps(function)
    def wrapped(self, identifier, *args, **kwargs):
        root = self.store.path.parent.parent
        version = None
        if function.__name__ == "execute_auto_run":
            version = self.store.run_detail(identifier)["config"].get("data_version")
        with snapshot(root, version):
            return function(self, identifier, *args, **kwargs)
    return wrapped


def freeze_root(function):
    @wraps(function)
    def wrapped(root, *args, **kwargs):
        with snapshot(root):
            return function(root, *args, **kwargs)
    return wrapped


def register_root(root) -> Path:
    root = Path(root).expanduser().resolve()
    _roots.add(root)
    return root


def locate(path) -> tuple[Path, str]:
    path = Path(path).expanduser().resolve()
    env = os.environ.get("MLQUANT_DATA_ROOT")
    if env:
        register_root(env)
    for parent in path.parents:
        if (parent / "storage.yaml").is_file():
            register_root(parent)
    candidates = [r for r in _roots if path.is_relative_to(r)]
    if candidates:
        root = max(candidates, key=lambda p: len(p.parts))
    else:
        root = next((p.parent for p in path.parents
                     if p.name in {"equity", "factor_library", "artifacts"}), path.parent)
        register_root(root)
    return root, path.relative_to(root).as_posix()


def store_for(root, *, initialize=False):
    root = register_root(root)
    stores = getattr(_local, "stores", None)
    if stores is None:
        stores = _local.stores = {}
    # Configuration can be changed between explicit operations, including tests.
    from mlquant.storage import ClickHouseConfig

    config = ClickHouseConfig.load(root)
    key = (root, config)
    if key not in stores:
        stores[key] = ClickHouseStore(root)
    store = stores[key]
    if initialize and (not store.available() or not (root / "storage.yaml").is_file()):
        store.initialize()
    return store


def read_frame(path, columns=None, filters=None, **kwargs):
    root, key = locate(path)
    if key.startswith("equity/") and root in _snapshots.get():
        kwargs.setdefault("as_of", _snapshots.get()[root])
    return store_for(root).read_frame(key, columns=columns, filters=filters, **kwargs)


def write_frame(frame, path, *, index=None, **kwargs):
    root, key = locate(path)
    return store_for(root, initialize=True).write_frame(key, frame, index=index)


def read_csv(path, **kwargs):
    frame = read_frame(path)
    if "index_col" in kwargs and kwargs["index_col"] is not None:
        column = kwargs["index_col"]
        frame = frame.set_index(frame.columns[column] if isinstance(column, int) else column)
    return frame


def write_csv(frame, path=None, *, index=True, **kwargs):
    if path is None:
        return frame.to_csv(index=index, **kwargs)
    return write_frame(frame, path, index=index)


def exists(path):
    path = Path(path)
    if path.name == "catalog.sqlite":
        from mlquant.catalog_connection import catalog_exists

        root = path.parent.parent if path.parent.name == "factor_library" else path.parent
        return catalog_exists(store_for(root))
    if path.is_dir():
        return True
    if path.is_file() and path.suffix not in {".parquet", ".csv", ".sqlite"}:
        return True
    root, key = locate(path)
    version = _snapshots.get().get(root) if key.startswith("equity/") else None
    meta = store_for(root).manifest(key, as_of=version)
    return meta is not None and meta["kind"] != "deleted"


def read_bytes(path):
    path = Path(path)
    if path.is_file() and not any(path.resolve().is_relative_to(r) for r in _roots):
        return path.read_bytes()
    root, key = locate(path)
    store = store_for(root)
    version = _snapshots.get().get(root) if key.startswith("equity/") else None
    meta = store.manifest(key, as_of=version)
    if meta is None:
        if path.is_file() and path.suffix not in {".parquet", ".csv", ".sqlite"}:
            return path.read_bytes()
        raise FileNotFoundError(path)
    if meta["kind"] == "frame":
        frame = store.read_frame(key, as_of=version)
        if path.suffix == ".csv":
            return frame.to_csv(index=False).encode("utf-8")
        return frame.to_parquet()
    return store.read_blob(key, as_of=version)


def write_bytes(path, data):
    root, key = locate(path)
    store_for(root, initialize=True).write_blob(key, data)
    return len(data)


def read_text(path, encoding="utf-8", **kwargs):
    return read_bytes(path).decode(encoding)


def write_text(path, text, encoding="utf-8", **kwargs):
    return write_bytes(path, text.encode(encoding))


def stat(path):
    root, key = locate(path)
    version = _snapshots.get().get(root) if key.startswith("equity/") else None
    meta = store_for(root).manifest(key, as_of=version)
    if meta is None:
        return Path(path).stat()
    return SimpleNamespace(st_size=meta["rows"], st_mtime_ns=meta["version"],
                           st_mtime=meta["version"] / 1e9)


def iterdir(path):
    root, key = locate(Path(path) / "__directory__")
    prefix = key.removesuffix("__directory__")
    entries = store_for(root).list(prefix)
    children = {root / prefix / e["path"][len(prefix):].split("/")[0]
                for e in entries if e["kind"] != "deleted"}
    if Path(path).is_dir():
        children.update(Path(path).iterdir())
    return iter(sorted(children))


def glob(path, pattern):
    return (p for p in iterdir(path) if fnmatch.fnmatch(p.name, pattern))


def unlink(path, missing_ok=False):
    if not exists(path) and not missing_ok:
        raise FileNotFoundError(path)
    root, key = locate(path)
    store = store_for(root, initialize=True)
    with store.batch() as batch:
        batch.blob(key, b"")
        batch.entries[key]["kind"] = "deleted"


def replace(source, destination):
    copy(source, destination)
    unlink(source, missing_ok=True)
    return Path(destination)


def copy(source, destination):
    root, key = locate(source)
    meta = store_for(root).manifest(key)
    if meta and meta["kind"] == "frame":
        write_frame(read_frame(source), destination)
    else:
        write_bytes(destination, read_bytes(source))
    return Path(destination)


def export_resource(source, destination):
    """Explicit export; Parquet is an interchange format, never the authority."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if Path(source).suffix == ".parquet":
        read_frame(source).to_parquet(destination)
    elif Path(source).suffix == ".csv":
        read_frame(source).to_csv(destination, index=False)
    else:
        destination.write_bytes(read_bytes(source))
    return destination


def frame_info(path):
    root, key = locate(path)
    version = _snapshots.get().get(root) if key.startswith("equity/") else None
    meta = store_for(root).manifest(key, as_of=version)
    if not meta or meta["kind"] != "frame":
        raise FileNotFoundError(path)
    import base64

    schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(meta["schema"])))
    return schema, meta


class TableReader:
    """Read-only compatibility surface for batch consumers; uses database projections."""
    def __init__(self, path):
        self.path = path
        self.schema_arrow, meta = frame_info(path)
        root, key = locate(path)
        self.metadata = SimpleNamespace(num_rows=store_for(root).count_frame(key, as_of=meta["version"]))

    def iter_batches(self, batch_size=65536, columns=None):
        root, key = locate(self.path)
        version = _snapshots.get().get(root) if key.startswith("equity/") else None
        yield from store_for(root).iter_batches(key, columns=columns, as_of=version, batch_size=batch_size)


def import_file(path, destination, *, mode="replace", keys=None, batch_size=524288,
                checkpoint=None, transform=None):
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    root, key = locate(destination)
    store = store_for(root, initialize=True)
    count = 0
    with store.batch() as batch:
        reader = pq.ParquetFile(source)
        import json

        metadata = reader.schema_arrow.metadata or {}
        index_columns = json.loads(metadata.get(b"pandas", b"{}" )).get("index_columns", [])
        preserve_index = any(isinstance(c, str) for c in index_columns)
        fields = []
        for field in reader.schema_arrow:
            dtype = field.type
            if pa.types.is_timestamp(dtype):
                dtype = pa.timestamp("ns", tz=dtype.tz)
            elif (pa.types.is_list(dtype) or pa.types.is_large_list(dtype)
                  or pa.types.is_struct(dtype) or pa.types.is_map(dtype) or pa.types.is_null(dtype)):
                dtype = pa.string()
            fields.append(pa.field(field.name, dtype, nullable=True))
        schema = pa.schema(fields, metadata=reader.schema_arrow.metadata)
        for chunk in reader.iter_batches(batch_size=batch_size):
            frame = chunk.to_pandas()
            if transform is not None:
                frame = transform(frame)
            batch.frame(key, frame, mode=mode, keys=keys, index=preserve_index, schema=schema)
            count += len(frame)
        if not count:
            if mode == "upsert":
                return {"rows": 0, "path": key, "changed": False}
            empty = reader.schema_arrow.empty_table().to_pandas()
            batch.frame(key, transform(empty) if transform is not None else empty,
                        index=False, schema=schema)
        if checkpoint:
            batch.blob(*checkpoint)
    return {"rows": count, "path": key, "version": batch.version, "changed": True}
