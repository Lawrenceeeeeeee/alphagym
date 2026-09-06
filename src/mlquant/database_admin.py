"""Native whole-database backups; restore only into an empty destination."""
from __future__ import annotations

import re

from mlquant.factor_store import FactorStore
from mlquant.storage import StorageError, ident
from mlquant.storage_io import store_for


def _name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", value):
        raise StorageError("Backup name must contain only letters, digits, dot, dash or underscore")
    return value


def backup_database(root, name, *, base=None):
    store = store_for(root, initialize=True)
    name = _name(name)
    query = f"BACKUP DATABASE {ident(store.config.database)}"
    with FactorStore.from_root(root) as catalog, catalog.connection, store.publication_lock():
        pub_lock = store.table("data_publish_lock")
        query += f" EXCEPT TABLES {catalog.connection.lock}, {pub_lock} TO Disk('backups', %(name)s)"
        params = {"name": name}
        if base:
            query += " SETTINGS base_backup=Disk('backups', %(base)s)"
            params["base"] = _name(base)
        result = store.client.command(query, parameters=params)
    return {"database": store.config.database, "backup": name, "base": base,
            "scope": "whole_database", "result": str(result)}


def restore_database(root, name, *, source_database):
    store = store_for(root)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", source_database):
        raise StorageError("Invalid source database name")
    count = store.client.command("SELECT count() FROM system.tables WHERE database=%(db)s",
                                 parameters={"db": store.config.database})
    if count:
        raise StorageError("Restore requires an empty destination database; choose a new database in storage.yaml")
    result = store.client.command(
        f"RESTORE DATABASE {ident(source_database)} AS {ident(store.config.database)} "
        "FROM Disk('backups', %(name)s)", parameters={"name": _name(name)},
    )
    store.initialize()
    return {"database": store.config.database, "backup": name, "result": str(result)}
