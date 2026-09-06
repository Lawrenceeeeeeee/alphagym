"""Transactional catalog compatibility layer with ClickHouse-only persistence.

The existing catalog SQL runs against a transient in-memory relational working
set for constraints and rollback. Committed row changes live in ClickHouse, not
a SQLite file. A database DDL lock serializes catalog writers across processes
and hosts. A crashed writer leaves a lock requiring explicit operator recovery;
we deliberately never guess that another writer is dead.
"""
from __future__ import annotations

import json
import sqlite3
import time

from mlquant.storage import StorageError


class CatalogConnection:
    def __init__(self, memory, store, *, readonly=False):
        self.memory = memory
        self.store = store
        self.readonly = readonly
        self.depth = 0
        self.version = -1
        self.before = {}
        self.tables = [r[0] for r in memory.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )]
        self.columns = {t: [r[1] for r in memory.execute(f'PRAGMA table_info("{t}")')]
                        for t in self.tables}
        self.keys = {t: [r[1] for r in sorted(memory.execute(f'PRAGMA table_info("{t}")'),
                                             key=lambda r: r[5]) if r[5]] for t in self.tables}
        self.events = store.table("catalog_events")
        self.lock = store.table("catalog_write_lock")
        if not readonly:
            store.client.command(f"""CREATE TABLE IF NOT EXISTS {self.events} (
                workspace String, table_name String, row_key String,
                version UInt64, deleted Bool, payload String
            ) ENGINE=MergeTree ORDER BY (workspace, table_name, row_key, version)""")
        if not store.client.command(f"EXISTS TABLE {self.events}"):
            raise FileNotFoundError("Catalog missing; run factor sync")
        self._refresh(force=True)
        if readonly:
            self.memory.execute("PRAGMA query_only=ON")

    def _refresh(self, *, force=False):
        version = self.store.client.command(
            f"SELECT max(version) FROM {self.events} WHERE workspace=%(w)s",
            parameters={"w": self.store.config.workspace},
        )
        if version == self.version and not force:
            return
        rows = self.store.client.query(
            f"SELECT table_name, row_key, deleted, payload FROM {self.events} "
            "WHERE workspace=%(w)s AND version<=%(v)s ORDER BY version DESC "
            "LIMIT 1 BY table_name, row_key",
            parameters={"w": self.store.config.workspace, "v": version},
        ).result_rows
        self.memory.execute("PRAGMA query_only=OFF")
        self.memory.execute("PRAGMA foreign_keys=OFF")
        for table in self.tables:
            self.memory.execute(f'DELETE FROM "{table}"')
        for table, key, deleted, payload in rows:
            if deleted:
                continue
            if table not in self.columns:
                raise StorageError(f"Catalog schema needs upgrade: {table}")
            data = self._paths(json.loads(payload), encode=False)
            columns = self.columns[table]
            self.memory.execute(
                f'INSERT INTO "{table}" VALUES ({",".join("?" for _ in columns)})',
                [data.get(c) for c in columns],
            )
        self.memory.commit()
        self.memory.execute("PRAGMA foreign_keys=ON")
        if self.readonly:
            self.memory.execute("PRAGMA query_only=ON")
        self.version = version

    def _snapshot(self):
        result = {}
        for table in self.tables:
            for row in self.memory.execute(f'SELECT * FROM "{table}"'):
                data = dict(row)
                key = json.dumps([data[c] for c in self.keys[table]], ensure_ascii=False)
                result[table, key] = json.dumps(self._paths(data, encode=True), ensure_ascii=False, sort_keys=True,
                                               allow_nan=False)
        return result

    def _paths(self, value, *, encode):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key.endswith("_json") and isinstance(item, str):
                    result[key] = json.dumps(self._paths(json.loads(item), encode=encode),
                                             ensure_ascii=False, sort_keys=True)
                else:
                    result[key] = self._paths(item, encode=encode)
            return result
        if isinstance(value, list):
            return [self._paths(item, encode=encode) for item in value]
        if isinstance(value, str):
            if not encode and value.startswith("@workspace/"):
                return str(self.store.root / value.removeprefix("@workspace/"))
            if encode:
                for prefix in (str(self.store.root), self.store.root.as_posix()):
                    if value.startswith((prefix + "/", prefix + "\\")):
                        return "@workspace/" + value[len(prefix) + 1:].replace("\\", "/")
        return value

    def __enter__(self):
        if self.readonly:
            raise sqlite3.OperationalError("readonly catalog")
        if self.depth:
            self.depth += 1
            return self
        deadline = time.monotonic() + 15
        while True:
            try:
                self.store.client.command(
                    f"CREATE TABLE {self.lock} (owner String) ENGINE=MergeTree ORDER BY tuple()"
                )
                break
            except Exception as error:  # noqa: BLE001 -- distinguish DDL lock contention
                if "TABLE_ALREADY_EXISTS" not in str(error):
                    raise StorageError("Cannot acquire ClickHouse catalog lock") from None
                if time.monotonic() >= deadline:
                    raise StorageError(
                        "Catalog writer is busy or a crashed writer left a lock; "
                        "verify all writers stopped before recovering the catalog lock"
                    ) from None
                time.sleep(.1)
        try:
            self._refresh(force=True)
            self.before = self._snapshot()
            self.depth = 1
            self.memory.execute("BEGIN")
        except BaseException:
            self.store.client.command(f"DROP TABLE {self.lock} SYNC")
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        self.depth -= 1
        if self.depth:
            return False
        try:
            if exc_type:
                self.memory.rollback()
                return False
            after = self._snapshot()
            version = max(time.time_ns(), self.version + 1)
            changes = []
            for table, key in self.before.keys() | after.keys():
                if self.before.get((table, key)) != after.get((table, key)):
                    changes.append([self.store.config.workspace, table, key, version,
                                    (table, key) not in after, after.get((table, key), "{}")])
            if changes:
                # One insert block is the catalog transaction publication boundary.
                self.store.client.insert(self.events, changes, column_names=[
                    "workspace", "table_name", "row_key", "version", "deleted", "payload",
                ])
            self.memory.commit()
            self.version = version if changes else self.version
        except BaseException:
            self.memory.rollback()
            self.version = -1
            raise
        finally:
            self.store.client.command(f"DROP TABLE {self.lock} SYNC")
        return False

    def execute(self, sql, parameters=()):
        reading = sql.lstrip().upper().startswith(("SELECT", "WITH", "PRAGMA"))
        if not self.depth:
            if reading:
                self._refresh()
            else:
                with self:
                    return self.memory.execute(sql, parameters)
        return self.memory.execute(sql, parameters)

    def executemany(self, sql, parameters):
        if not self.depth:
            with self:
                return self.memory.executemany(sql, parameters)
        return self.memory.executemany(sql, parameters)

    def commit(self):
        if not self.depth:
            self.memory.commit()

    def close(self):
        if self.depth:
            self.__exit__(RuntimeError, RuntimeError("connection closed"), None)
        self.memory.close()


def catalog_exists(store):
    table = store.table("catalog_events")
    return bool(store.client.command(f"EXISTS TABLE {table}")) and bool(store.client.command(
        f"SELECT count() FROM {table} WHERE workspace=%(w)s",
        parameters={"w": store.config.workspace},
    ))
