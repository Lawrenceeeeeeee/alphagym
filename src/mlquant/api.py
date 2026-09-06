"""Public workspace API. No shell invocation is required for synchronous work."""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from mlquant import storage_io
from mlquant.config import WorkspaceConfig, resolve_root
from mlquant.equity_data import EquityDataBundle
from mlquant.factor_store import FactorStore
from mlquant.factors.base import FactorDefinition
from mlquant.report_spec import ReportSpec, load_spec, parse_spec


class Workspace:
    """A data root with short-lived connections; safe to use in notebooks and agents.

    Construction and inspection never bootstrap a catalog. Call ``initialize``
    explicitly before registering factors or creating research tasks.
    """

    def __init__(self, root: str | Path | WorkspaceConfig | None = None) -> None:
        self.config = root if isinstance(root, WorkspaceConfig) else WorkspaceConfig(resolve_root(root))

    @property
    def root(self) -> Path:
        return self.config.root

    def initialize(self) -> Path:
        """Create/update the catalog and register built-ins (idempotent)."""
        from mlquant.factors.library import seed_definitions

        with FactorStore.from_root(self.root) as store:
            store.bootstrap(seed_definitions())
        return self.config.catalog

    def _store(self, *, readonly: bool = True) -> FactorStore:
        if not storage_io.exists(self.config.catalog):
            raise FileNotFoundError("Catalog missing; call Workspace.initialize() or factor sync")
        return FactorStore(self.config.catalog, readonly=readonly)

    def data(self) -> EquityDataBundle:
        return EquityDataBundle.from_root(self.root)

    def import_parquet(self, path: str | Path, table: str, *, mode="upsert") -> dict:
        """Import an external Parquet file into the canonical database contract."""
        from mlquant.ingest import import_parquet

        return import_parquet(path, table, self.root, mode=mode)

    def sync_tushare(self, *, token=None, start=None, end=None, dataset="market", **options) -> dict:
        """Synchronize completed dates; token defaults to TUSHARE_TOKEN and is never stored."""
        from mlquant.tushare_import import sync_tushare

        return sync_tushare(self.root, token=token, start=start, end=end, dataset=dataset, **options)

    def read_table(self, table: str, *, columns=None, filters=None, as_of=None) -> pd.DataFrame:
        """Projected, filtered database query, optionally pinned to a data version."""
        from mlquant.storage_io import store_for

        if table not in EquityDataBundle.TABLES:
            raise ValueError(f"Unknown equity table: {table}")
        return store_for(self.root).read_frame(f"equity/{table}.parquet", columns=columns,
                                               filters=filters, as_of=as_of)

    def write_table(self, table: str, frame, *, mode="upsert") -> dict:
        """Adapter entry point for custom data sources with canonical columns."""
        from mlquant.equity_data import SCHEMAS
        from mlquant.storage import KEYS
        from mlquant.storage_io import store_for

        if table not in SCHEMAS:
            raise ValueError(f"Unknown equity table: {table}")
        missing = set(SCHEMAS[table]) - set(frame)
        if missing:
            raise ValueError(f"{table}: missing fields: {sorted(missing)}")
        return store_for(self.root, initialize=True).write_frame(
            f"equity/{table}.parquet", frame, mode=mode, keys=KEYS[table],
        )

    def list_factors(self, family: str | None = None) -> list[dict[str, Any]]:
        with self._store() as store:
            return store.list_factors(family)

    def factor(self, factor_id: str) -> dict[str, Any]:
        with self._store() as store:
            return store.factor_detail(factor_id)

    def save_factor(self, definition: FactorDefinition) -> str:
        """Persist a revision; cache refresh remains an explicit operation."""
        with self._store(readonly=False) as store:
            return store.save_definition(definition)

    def create_report(
        self, spec: ReportSpec | dict[str, Any] | str | Path, *, ensure_runs: bool = False,
    ) -> dict[str, Any]:
        from mlquant.services import create_report

        if isinstance(spec, (str, Path)):
            spec = load_spec(spec)
        elif isinstance(spec, dict):
            spec = parse_spec(spec)
        with self._store(readonly=False) as store:
            return create_report(store, spec, ensure_runs=ensure_runs)

    def execute_report(self, report_id: str) -> Path:
        """Compute a queued report synchronously; reads never call this method."""
        from mlquant.report_engine import ReportEngine

        with self._store(readonly=False) as store:
            return ReportEngine(store).execute_report(report_id)

    def report(self, report_id: str) -> dict[str, Any]:
        with self._store() as store:
            return store.report_detail(report_id)

    def list_reports(self) -> list[dict[str, Any]]:
        if not storage_io.exists(self.config.catalog):
            return []
        with self._store() as store:
            return store.list_reports()

    def run(self, run_id: str) -> dict[str, Any]:
        with self._store() as store:
            return store.run_detail(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        if not storage_io.exists(self.config.catalog):
            return []
        with self._store() as store:
            return store.list_runs()

    def run_factors(
        self, factor_ids: list[str], *, start_date: str, end_date: str,
        index_code: str = "ALL_A", mode: str = "formal",
    ) -> dict[str, Any]:
        """Validate PIT inputs and execute a raw quintile factor backtest."""
        from mlquant.factor_research_service import FactorResearchService

        if mode not in {"formal", "smoke"}:
            raise ValueError("mode must be formal or smoke")
        with self._store(readonly=False) as store:
            service = FactorResearchService(store)
            service.validate_auto_request(
                factor_ids, mode=mode, index_code=index_code,
                start_date=start_date, end_date=end_date,
            )
            run_id = store.create_run(factor_ids, mode=mode, config={
                "source": "automatic", "index_code": index_code,
                "start_date": start_date, "end_date": end_date,
                "point_in_time_audit_passed": mode == "formal",
            })
            output = service.execute_auto_run(run_id)
        return {"run_id": run_id, "status": "succeeded", "path": str(output)}

    def run_panel(
        self, factor_ids: list[str], panel: str | Path, *, mode: str = "smoke",
        point_in_time_audit_passed: bool = False,
    ) -> dict[str, Any]:
        """Evaluate a supplied panel through the same registered-factor service."""
        from mlquant.factor_research_service import FactorResearchService

        if mode not in {"formal", "smoke"}:
            raise ValueError("mode must be formal or smoke")
        panel = Path(panel).expanduser().resolve()
        if not storage_io.exists(panel):
            raise FileNotFoundError(panel)
        with self._store(readonly=False) as store:
            run_id = store.create_run(factor_ids, mode=mode, config={
                "panel": str(panel), "point_in_time_audit_passed": point_in_time_audit_passed,
            })
            output = FactorResearchService(store).execute_panel_run(run_id, panel)
        return {"run_id": run_id, "status": "succeeded", "path": str(output)}

    def start_report(self, report_id: str) -> dict[str, Any]:
        """Dispatch a local background worker. Observe it with wait_report."""
        from mlquant.jobs import start_report

        return start_report(self.root, report_id)

    def wait_report(
        self, report_id: str, *, timeout: float = 3600, poll_interval: float = 2,
    ) -> dict[str, Any]:
        """Observe a task started by another worker; never starts queued work."""
        return self._wait(self.report, report_id, timeout=timeout, poll_interval=poll_interval)

    def wait_run(
        self, run_id: str, *, timeout: float = 3600, poll_interval: float = 2,
    ) -> dict[str, Any]:
        """Observe a factor run; timeout never cancels the producing worker."""
        return self._wait(self.run, run_id, timeout=timeout, poll_interval=poll_interval)

    @staticmethod
    def _wait(
        read: Callable[[str], dict[str, Any]], task_id: str, *, timeout: float, poll_interval: float,
    ) -> dict[str, Any]:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")
        deadline = time.monotonic() + timeout
        while True:
            row = read(task_id)
            if row["status"] in {"succeeded", "failed", "cancelled"}:
                return row
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            time.sleep(min(poll_interval, remaining))

    def report_manifest(self, report_id: str) -> dict[str, Any]:
        row = self.report(report_id)
        if row["status"] != "succeeded" or not row.get("path"):
            raise FileNotFoundError(f"Report artifacts unavailable: {report_id}")
        return json.loads(storage_io.read_text(Path(row["path"]) / "manifest.json", encoding="utf-8"))

    def status(self) -> dict[str, Any]:
        """Read metadata only, including on a nonexistent root."""

        equity: dict[str, Any] = {}
        for name in EquityDataBundle.TABLES:
            path = self.root / "equity" / f"{name}.parquet"
            if storage_io.exists(path):
                equity[name] = {"size": None, "data_version": storage_io.stat(path).st_mtime_ns}
                try:
                    parquet = storage_io.TableReader(path)
                    equity[name].update(columns=sorted(parquet.schema_arrow.names),
                                        rows=parquet.metadata.num_rows)
                except (OSError, ValueError) as error:
                    equity[name]["schema_error"] = str(error)
        library: dict[str, Any] = {
            "factors": 0, "runs_succeeded": 0, "runs_active": [],
            "reports": 0, "reports_active": [], "latest_run": None,
        }
        if storage_io.exists(self.config.catalog):
            with self._store() as store:
                db = store.connection
                library["factors"] = db.execute("SELECT COUNT(*) FROM factor").fetchone()[0]
                library["reports"] = db.execute("SELECT COUNT(*) FROM report").fetchone()[0]
                library["runs_succeeded"] = db.execute(
                    "SELECT COUNT(*) FROM research_run WHERE status='succeeded'"
                ).fetchone()[0]
                for table, key, identifier in (
                    ("research_run", "runs_active", "run_id"),
                    ("report", "reports_active", "report_id"),
                ):
                    library[key] = [dict(row) for row in db.execute(
                        f"SELECT {identifier}, status, progress FROM {table} "
                        "WHERE status IN ('queued','running') ORDER BY created_at"
                    )]
                row = db.execute(
                    "SELECT run_id, status, completed_at FROM research_run "
                    "WHERE status='succeeded' ORDER BY completed_at DESC LIMIT 1"
                ).fetchone()
                library["latest_run"] = dict(row) if row else None
        return {"root": str(self.root), "backend": "clickhouse", "equity": equity,
                "factor_library": library}
