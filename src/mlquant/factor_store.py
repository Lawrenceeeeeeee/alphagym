from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import pandas as pd

from mlquant import storage_io
from mlquant.factor_dsl import FormulaCompiler, FormulaEngine
from mlquant.factor_operators import build_field_registry, build_operator_registry
from mlquant.factors.base import FactorContext, FactorDefinition, FactorRegistry, FactorSpec


def _now() -> str:
    return datetime.now(UTC).isoformat()


class FactorStore:
    """ClickHouse authority for definitions and lineage; memory-only mode for pure tests."""

    def __init__(self, path: str | Path = ":memory:", *, readonly: bool = False) -> None:
        self.path = Path(path).resolve() if str(path) != ":memory:" else None
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.fields = build_field_registry()
        self.operators = build_operator_registry()
        self._create_schema()
        fresh = True
        if self.path is not None:
            from mlquant.catalog_connection import CatalogConnection, catalog_exists
            from mlquant.storage_io import store_for

            root = self.path.parent.parent if self.path.parent.name == "factor_library" else self.path.parent
            database = store_for(root, initialize=not readonly)
            fresh = not catalog_exists(database)
            if readonly and fresh:
                self.connection.close()
                raise FileNotFoundError("Catalog missing; run factor sync")
            self.connection = CatalogConnection(self.connection, database, readonly=readonly)
        if not readonly and fresh:
            self.sync_capabilities()

    @classmethod
    def from_root(cls, root: str | Path) -> FactorStore:
        return cls(Path(root).expanduser().resolve() / "factor_library" / "catalog.sqlite")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS catalog_migration (
                source_sha256 TEXT PRIMARY KEY,
                rows_imported INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS factor (
                factor_id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '',
                family TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL CHECK(status IN ('active','blocked','deprecated')),
                hypothesis_id TEXT NOT NULL,
                expected_direction TEXT NOT NULL,
                current_revision_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS factor_revision (
                revision_id TEXT PRIMARY KEY,
                factor_id TEXT NOT NULL REFERENCES factor(factor_id),
                revision_number INTEGER NOT NULL,
                formula_version TEXT NOT NULL,
                formula_source TEXT NOT NULL,
                canonical_ast TEXT NOT NULL,
                definition_hash TEXT NOT NULL,
                fields_json TEXT NOT NULL,
                operators_json TEXT NOT NULL,
                dependencies_json TEXT NOT NULL,
                models_json TEXT NOT NULL,
                lookback_days INTEGER NOT NULL,
                min_observations INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(factor_id, revision_number)
            );
            CREATE TABLE IF NOT EXISTS factor_dependency (
                revision_id TEXT NOT NULL REFERENCES factor_revision(revision_id),
                dependency_factor_id TEXT NOT NULL REFERENCES factor(factor_id),
                dependency_revision_id TEXT NOT NULL REFERENCES factor_revision(revision_id),
                PRIMARY KEY(revision_id, dependency_factor_id)
            );
            CREATE TABLE IF NOT EXISTS operator_definition (
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                description TEXT NOT NULL,
                deterministic INTEGER NOT NULL,
                synced_at TEXT NOT NULL,
                PRIMARY KEY(name, version, code_hash)
            );
            CREATE TABLE IF NOT EXISTS field_definition (
                name TEXT PRIMARY KEY,
                dataset TEXT NOT NULL,
                column_name TEXT NOT NULL,
                dtype TEXT NOT NULL,
                frequency TEXT NOT NULL,
                entity_key TEXT NOT NULL,
                event_time TEXT NOT NULL,
                available_time TEXT,
                formal INTEGER NOT NULL,
                unit TEXT,
                description TEXT NOT NULL,
                synced_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_artifact (
                model_version_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                version TEXT NOT NULL,
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                training_snapshot_hash TEXT NOT NULL,
                training_start TEXT,
                training_end TEXT NOT NULL,
                code_version TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                UNIQUE(model_id, version)
            );
            CREATE TABLE IF NOT EXISTS research_run (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL CHECK(mode IN ('smoke','formal')),
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                config_json TEXT NOT NULL,
                data_snapshot_hash TEXT,
                code_version TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS report (
                report_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                spec_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed','cancelled')),
                progress REAL NOT NULL DEFAULT 0,
                error TEXT,
                run_id TEXT REFERENCES research_run(run_id),
                path TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS run_factor (
                run_id TEXT NOT NULL REFERENCES research_run(run_id),
                factor_id TEXT NOT NULL REFERENCES factor(factor_id),
                revision_id TEXT NOT NULL REFERENCES factor_revision(revision_id),
                PRIMARY KEY(run_id, factor_id)
            );
            CREATE TABLE IF NOT EXISTS factor_metric (
                run_id TEXT NOT NULL REFERENCES research_run(run_id),
                factor_id TEXT NOT NULL REFERENCES factor(factor_id),
                index_code TEXT NOT NULL DEFAULT '',
                period TEXT NOT NULL DEFAULT 'all',
                value_type TEXT NOT NULL DEFAULT 'neutralized',
                orientation TEXT NOT NULL DEFAULT 'original',
                cost_scenario TEXT NOT NULL DEFAULT 'base_5bps',
                metric_name TEXT NOT NULL,
                metric_value REAL,
                PRIMARY KEY(
                    run_id, factor_id, index_code, period, value_type,
                    orientation, cost_scenario, metric_name
                )
            );
            CREATE TABLE IF NOT EXISTS artifact (
                artifact_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES research_run(run_id),
                factor_id TEXT,
                kind TEXT NOT NULL,
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS official_baseline (
                factor_id TEXT PRIMARY KEY REFERENCES factor(factor_id),
                run_id TEXT NOT NULL REFERENCES research_run(run_id),
                revision_id TEXT NOT NULL REFERENCES factor_revision(revision_id),
                note TEXT NOT NULL DEFAULT '',
                promoted_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_event (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                factor_id TEXT,
                revision_id TEXT,
                run_id TEXT,
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_factor_family ON factor(family, status);
            CREATE INDEX IF NOT EXISTS idx_revision_factor ON factor_revision(factor_id, revision_number);
            CREATE INDEX IF NOT EXISTS idx_run_status ON research_run(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_report_status ON report(status, created_at);
            """
        )
        self.connection.commit()

    def sync_capabilities(self) -> None:
        now = _now()
        with self.connection:
            for operator in self.operators.list():
                self.connection.execute(
                    """INSERT OR IGNORE INTO operator_definition
                    (name, version, code_hash, description, deterministic, synced_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (operator.name, operator.version, operator.code_hash,
                     operator.description, int(operator.deterministic), now),
                )
            for field in self.fields.list():
                self.connection.execute(
                    """INSERT INTO field_definition
                    (name, dataset, column_name, dtype, frequency, entity_key, event_time,
                     available_time, formal, unit, description, synced_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                      dataset=excluded.dataset, column_name=excluded.column_name,
                      dtype=excluded.dtype, frequency=excluded.frequency,
                      entity_key=excluded.entity_key, event_time=excluded.event_time,
                      available_time=excluded.available_time, formal=excluded.formal,
                      unit=excluded.unit, description=excluded.description,
                      synced_at=excluded.synced_at""",
                    (field.name, field.dataset, field.column, field.dtype, field.frequency,
                     field.entity_key, field.event_time, field.available_time,
                     int(field.formal), field.unit, field.description, now),
                )

    def _current_revision(self, factor_id: str) -> str:
        row = self.connection.execute(
            "SELECT current_revision_id FROM factor WHERE factor_id = ?", (factor_id,)
        ).fetchone()
        if row is None or not row["current_revision_id"]:
            raise KeyError(f"unknown factor dependency: {factor_id}")
        return str(row["current_revision_id"])

    def _assert_acyclic(self, factor_id: str, dependencies: Iterable[str]) -> None:
        rows = self.connection.execute(
            """SELECT f.factor_id, d.dependency_factor_id
            FROM factor f JOIN factor_dependency d ON d.revision_id=f.current_revision_id"""
        ).fetchall()
        graph: dict[str, set[str]] = {}
        for row in rows:
            graph.setdefault(str(row["factor_id"]), set()).add(
                str(row["dependency_factor_id"])
            )
        graph[factor_id] = set(dependencies)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError(f"cyclic factor dependency involving {node}")
            if node in visited:
                return
            visiting.add(node)
            for child in graph.get(node, set()):
                visit(child)
            visiting.remove(node)
            visited.add(node)

        visit(factor_id)

    def save_definition(self, definition: FactorDefinition) -> str:
        if definition.status not in {"active", "blocked", "deprecated"}:
            raise ValueError(f"invalid factor status: {definition.status}")
        compiler = FormulaCompiler(
            self.fields, self.operators, factor_resolver=self._current_revision,
            model_resolver=self._resolve_model,
        )
        compiled = compiler.compile(definition.formula)
        self._assert_acyclic(
            definition.factor_id, (item[0] for item in compiled.factor_dependencies)
        )
        existing = self.connection.execute(
            "SELECT * FROM factor WHERE factor_id = ?", (definition.factor_id,)
        ).fetchone()
        if existing is not None and existing["name"] != definition.name:
            raise ValueError("factor_id is immutable and already belongs to another name")
        current = None
        if existing is not None and existing["current_revision_id"]:
            current = self.connection.execute(
                "SELECT * FROM factor_revision WHERE revision_id = ?",
                (existing["current_revision_id"],),
            ).fetchone()
        now = _now()
        with self.connection:
            if existing is None:
                self.connection.execute(
                    """INSERT INTO factor
                    (factor_id, name, description, family, tags_json, status, hypothesis_id,
                     expected_direction, current_revision_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)""",
                    (definition.factor_id, definition.name, definition.description,
                     definition.family, json.dumps(definition.tags, ensure_ascii=False),
                     definition.status, definition.hypothesis_id,
                     definition.expected_direction, now, now),
                )
            else:
                self.connection.execute(
                    """UPDATE factor SET description=?, family=?, tags_json=?, status=?,
                    hypothesis_id=?, expected_direction=?, updated_at=? WHERE factor_id=?""",
                    (definition.description, definition.family,
                     json.dumps(definition.tags, ensure_ascii=False), definition.status,
                     definition.hypothesis_id, definition.expected_direction, now,
                     definition.factor_id),
                )
            if current is not None and current["definition_hash"] == compiled.definition_hash:
                return str(current["revision_id"])
            revision_number = int(self.connection.execute(
                "SELECT COALESCE(MAX(revision_number), 0) + 1 FROM factor_revision WHERE factor_id=?",
                (definition.factor_id,),
            ).fetchone()[0])
            revision_id = str(uuid.uuid4())
            self.connection.execute(
                """INSERT INTO factor_revision
                (revision_id, factor_id, revision_number, formula_version, formula_source,
                 canonical_ast, definition_hash, fields_json, operators_json,
                 dependencies_json, models_json, lookback_days, min_observations, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (revision_id, definition.factor_id, revision_number,
                 definition.formula_version, compiled.source, compiled.canonical_ast,
                 compiled.definition_hash, json.dumps(compiled.fields),
                 json.dumps(compiled.operators), json.dumps(compiled.factor_dependencies),
                 json.dumps(compiled.model_dependencies), compiled.lookback_days,
                 compiled.min_observations, now),
            )
            for dependency_id, dependency_revision in compiled.factor_dependencies:
                self.connection.execute(
                    """INSERT INTO factor_dependency
                    (revision_id, dependency_factor_id, dependency_revision_id)
                    VALUES (?, ?, ?)""",
                    (revision_id, dependency_id, dependency_revision),
                )
            self.connection.execute(
                "UPDATE factor SET current_revision_id=?, updated_at=? WHERE factor_id=?",
                (revision_id, now, definition.factor_id),
            )
            self._audit("factor_saved", factor_id=definition.factor_id,
                        revision_id=revision_id,
                        details={"revision_number": revision_number})
        return revision_id

    def validate_formula(self, formula: str) -> dict[str, Any]:
        compiled = FormulaCompiler(
            self.fields, self.operators, factor_resolver=self._current_revision,
            model_resolver=self._resolve_model,
        ).compile(formula)
        return {
            "canonical_ast": json.loads(compiled.canonical_ast),
            "definition_hash": compiled.definition_hash,
            "fields": compiled.fields,
            "operators": compiled.operators,
            "factor_dependencies": compiled.factor_dependencies,
            "model_dependencies": compiled.model_dependencies,
            "lookback_days": compiled.lookback_days,
            "min_observations": compiled.min_observations,
        }

    def _resolve_model(self, model_id: str) -> str:
        row = self.connection.execute(
            """SELECT model_version_id FROM model_artifact
            WHERE model_id=? AND status='active' ORDER BY training_end DESC LIMIT 1""",
            (model_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown active model: {model_id}")
        return str(row["model_version_id"])

    def register_model_artifact(
        self, *, model_id: str, version: str, path: str | Path,
        training_snapshot_hash: str, training_end: str,
        code_version: str, training_start: str | None = None,
        status: str = "active", metadata: dict[str, Any] | None = None,
    ) -> str:
        artifact_path = Path(path).expanduser().resolve()
        if not storage_io.exists(artifact_path):
            raise FileNotFoundError(artifact_path)
        model_version_id = str(uuid.uuid4())
        digest = hashlib.sha256(storage_io.read_bytes(artifact_path)).hexdigest()
        with self.connection:
            self.connection.execute(
                """INSERT INTO model_artifact
                (model_version_id, model_id, version, path, sha256,
                 training_snapshot_hash, training_start, training_end,
                 code_version, status, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (model_version_id, model_id, version, str(artifact_path), digest,
                 training_snapshot_hash, training_start, training_end,
                 code_version, status,
                 json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True), _now()),
            )
            self._audit("model_registered", details={
                "model_id": model_id, "version": version,
                "model_version_id": model_version_id,
            })
        return model_version_id

    def bootstrap(self, definitions: Iterable[FactorDefinition]) -> None:
        with self.connection:
            self._bootstrap(definitions)

    def _bootstrap(self, definitions: Iterable[FactorDefinition]) -> None:
        pending = list(definitions)
        while pending:
            deferred: list[FactorDefinition] = []
            progress = 0
            for definition in pending:
                try:
                    self.save_definition(definition)
                    progress += 1
                except KeyError:
                    deferred.append(definition)
            if not progress:
                names = [item.factor_id for item in deferred]
                raise ValueError(f"unresolved factor dependencies: {names}")
            pending = deferred

    def list_factors(self, family: str | None = None) -> list[dict[str, Any]]:
        query = """SELECT f.*, r.formula_source, r.revision_number, r.definition_hash,
                   r.lookback_days, r.min_observations
                   FROM factor f JOIN factor_revision r ON r.revision_id=f.current_revision_id"""
        params: tuple[object, ...] = ()
        if family:
            query += " WHERE f.family=?"
            params = (family,)
        query += " ORDER BY f.family, f.name"
        return [dict(row) for row in self.connection.execute(query, params).fetchall()]

    def factor_detail(self, factor_id: str) -> dict[str, Any]:
        factor = self.connection.execute(
            "SELECT * FROM factor WHERE factor_id=?", (factor_id,)
        ).fetchone()
        if factor is None:
            raise KeyError(factor_id)
        revisions = [dict(row) for row in self.connection.execute(
            """SELECT * FROM factor_revision WHERE factor_id=?
            ORDER BY revision_number DESC""", (factor_id,)
        ).fetchall()]
        result = dict(factor)
        result["revisions"] = revisions
        result["tags"] = json.loads(result.pop("tags_json"))
        result["dependencies"] = [dict(row) for row in self.connection.execute(
            """SELECT d.*, f.current_revision_id,
            CASE WHEN d.dependency_revision_id=f.current_revision_id THEN 0 ELSE 1 END AS stale
            FROM factor_dependency d JOIN factor f ON f.factor_id=d.dependency_factor_id
            WHERE d.revision_id=? ORDER BY d.dependency_factor_id""",
            (result["current_revision_id"],),
        ).fetchall()]
        result["metrics"] = self.metrics_for_factor(factor_id)
        baseline = self.connection.execute(
            "SELECT * FROM official_baseline WHERE factor_id=?", (factor_id,)
        ).fetchone()
        result["official_baseline"] = dict(baseline) if baseline else None
        return result

    def export_catalog(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for row in self.list_factors():
            records.append({
                "factor_id": row["factor_id"], "name": row["name"],
                "description": row["description"], "family": row["family"],
                "tags": json.loads(row["tags_json"]), "status": row["status"],
                "hypothesis_id": row["hypothesis_id"],
                "expected_direction": row["expected_direction"],
                "formula": row["formula_source"],
                "formula_version": self.connection.execute(
                    "SELECT formula_version FROM factor_revision WHERE revision_id=?",
                    (row["current_revision_id"],),
                ).fetchone()[0],
            })
        output.write_text(
            json.dumps({"schema_version": 1, "factors": records}, ensure_ascii=False,
                       indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output

    def import_catalog(self, path: str | Path) -> int:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        definitions = [FactorDefinition(
            factor_id=item["factor_id"], name=item["name"], formula=item["formula"],
            hypothesis_id=item["hypothesis_id"], family=item["family"],
            formula_version=item.get("formula_version", "1.0"),
            description=item.get("description", ""),
            expected_direction=item.get("expected_direction", "unknown"),
            tags=tuple(item.get("tags", ())), status=item.get("status", "active"),
        ) for item in payload["factors"]]
        self.bootstrap(definitions)
        return len(definitions)

    def load_registry(self) -> FactorRegistry:
        engine = FormulaEngine(self.fields, self.operators)
        registry = FactorRegistry(engine)
        factors = {row["factor_id"]: dict(row) for row in self.connection.execute(
            "SELECT * FROM factor"
        ).fetchall()}
        rows = self.connection.execute(
            "SELECT * FROM factor_revision ORDER BY created_at, revision_number"
        ).fetchall()
        for row in rows:
            locked = dict(json.loads(row["dependencies_json"]))
            locked_models = dict(json.loads(row["models_json"]))
            compiler = FormulaCompiler(
                self.fields, self.operators,
                factor_resolver=lambda factor_id, deps=locked: deps[factor_id],
                model_resolver=lambda model_id, models=locked_models: models[model_id],
            )
            compiled = compiler.compile(str(row["formula_source"]))

            def calculate(
                context: FactorContext, *, item: object = compiled,
                dependencies: dict[str, str] = locked,
            ) -> pd.Series:
                return engine.evaluate(
                    item, context,
                    factor_resolver=lambda factor_id, child_context: registry.compute_revision(
                        dependencies[factor_id], child_context
                    ),
                )

            factor = factors[str(row["factor_id"])]
            spec = FactorSpec(
                name=factor["name"], factor_id=factor["factor_id"],
                revision_id=row["revision_id"], hypothesis_id=factor["hypothesis_id"],
                family=factor["family"], formula_version=row["formula_version"],
                input_fields=tuple(json.loads(row["fields_json"])),
                lookback_days=row["lookback_days"], min_observations=row["min_observations"],
                availability_rule="all inputs available_at/event_time <= signal_date",
                expected_direction=factor["expected_direction"], calculator=calculate,
                formula=row["formula_source"], description=factor["description"],
                tags=tuple(json.loads(factor["tags_json"])), status=factor["status"],
                definition_hash=row["definition_hash"],
            )
            registry.add_revision(
                spec, current=row["revision_id"] == factor["current_revision_id"]
            )
        return registry

    def create_run(
        self, factor_ids: Iterable[str], *, mode: str, config: dict[str, Any],
        data_snapshot_hash: str | None = None, code_version: str | None = None,
    ) -> str:
        if mode not in {"smoke", "formal"}:
            raise ValueError("run mode must be smoke or formal")
        if self.path is not None:
            config = {**config, "data_version": config.get("data_version", storage_io.current_version(self.path.parent.parent))}
        run_id = str(uuid.uuid4())
        now = _now()
        with self.connection:
            self.connection.execute(
                """INSERT INTO research_run
                (run_id, mode, status, progress, config_json, data_snapshot_hash,
                 code_version, created_at) VALUES (?, ?, 'queued', 0, ?, ?, ?, ?)""",
                (run_id, mode, json.dumps(config, ensure_ascii=False, sort_keys=True),
                 data_snapshot_hash, code_version, now),
            )
            for factor_id in factor_ids:
                revision_id = self._current_revision(factor_id)
                self.connection.execute(
                    "INSERT INTO run_factor(run_id, factor_id, revision_id) VALUES (?, ?, ?)",
                    (run_id, factor_id, revision_id),
                )
            self._audit("run_created", run_id=run_id, details={"mode": mode})
        return run_id

    def set_run_status(
        self, run_id: str, status: str, *, progress: float | None = None,
        error: str | None = None,
    ) -> None:
        fields = ["status=?", "error=?"]
        values: list[object] = [status, error]
        if progress is not None:
            fields.append("progress=?")
            values.append(max(0.0, min(1.0, progress)))
        if status == "running":
            fields.append("started_at=COALESCE(started_at, ?)")
            values.append(_now())
        if status in {"succeeded", "failed", "cancelled"}:
            fields.append("completed_at=?")
            values.append(_now())
        values.append(run_id)
        with self.connection:
            self.connection.execute(
                f"UPDATE research_run SET {', '.join(fields)} WHERE run_id=?", values
            )
            self._audit("run_status", run_id=run_id,
                        details={"status": status, "error": error})

    def list_runs(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute(
            "SELECT * FROM research_run ORDER BY created_at DESC"
        ).fetchall()]

    def run_detail(self, run_id: str) -> dict[str, Any]:
        run = self.connection.execute(
            "SELECT * FROM research_run WHERE run_id=?", (run_id,)
        ).fetchone()
        if run is None:
            raise KeyError(run_id)
        result = dict(run)
        result["config"] = json.loads(result.pop("config_json"))
        result["factors"] = [dict(row) for row in self.connection.execute(
            """SELECT rf.factor_id, rf.revision_id, f.name
            FROM run_factor rf JOIN factor f ON f.factor_id=rf.factor_id
            WHERE rf.run_id=? ORDER BY f.name""", (run_id,)
        ).fetchall()]
        result["artifacts"] = [dict(row) for row in self.connection.execute(
            "SELECT * FROM artifact WHERE run_id=? ORDER BY kind, path", (run_id,)
        ).fetchall()]
        return result

    def latest_succeeded_runs(self) -> dict[str, dict[str, Any]]:
        """factor_id -> latest succeeded research_run row (by created_at)."""
        rows = self.connection.execute(
            """WITH ranked AS (
                SELECT rf.factor_id, r.run_id, r.mode, r.status, r.progress,
                       r.config_json, r.created_at, r.completed_at,
                       ROW_NUMBER() OVER (
                           PARTITION BY rf.factor_id
                           ORDER BY r.created_at DESC, r.run_id DESC
                       ) AS position
                FROM research_run r
                JOIN run_factor rf ON rf.run_id = r.run_id
                WHERE r.status = 'succeeded'
            )
            SELECT factor_id, run_id, mode, status, progress,
                   config_json, created_at, completed_at
            FROM ranked WHERE position = 1"""
        ).fetchall()
        return {str(row["factor_id"]): dict(row) for row in rows}

    def create_report(
        self, name: str, spec: dict[str, Any], *, run_id: str | None = None,
    ) -> str:
        report_id = str(uuid.uuid4())
        now = _now()
        with self.connection:
            self.connection.execute(
                """INSERT INTO report
                (report_id, name, spec_json, status, progress, run_id, created_at)
                VALUES (?, ?, ?, 'queued', 0, ?, ?)""",
                (report_id, name, json.dumps(spec, ensure_ascii=False, sort_keys=True),
                 run_id, now),
            )
            self._audit("report_created", run_id=run_id,
                        details={"report_id": report_id, "name": name})
        return report_id

    def claim_report(self, report_id: str) -> None:
        """Atomically admit one worker; completed reports are immutable."""
        with self.connection:
            result = self.connection.execute(
                "UPDATE report SET status='running', started_at=?, progress=0.01 "
                "WHERE report_id=? AND status='queued'", (_now(), report_id),
            )
            if result.rowcount != 1:
                row = self.report_detail(report_id)
                raise ValueError(f"Report is not queued: {report_id} ({row['status']})")

    def set_report_status(
        self, report_id: str, status: str, *, progress: float | None = None,
        error: str | None = None, run_id: str | None = None, path: str | Path | None = None,
    ) -> None:
        fields = ["status=?", "error=?"]
        values: list[object] = [status, error]
        if progress is not None:
            fields.append("progress=?")
            values.append(max(0.0, min(1.0, progress)))
        if run_id is not None:
            fields.append("run_id=?")
            values.append(run_id)
        if path is not None:
            fields.append("path=?")
            values.append(str(Path(path)))
        if status == "running":
            fields.append("started_at=COALESCE(started_at, ?)")
            values.append(_now())
        if status in {"succeeded", "failed", "cancelled"}:
            fields.append("completed_at=?")
            values.append(_now())
        values.append(report_id)
        with self.connection:
            self.connection.execute(
                f"UPDATE report SET {', '.join(fields)} WHERE report_id=?", values
            )
            self._audit("report_status", run_id=run_id,
                        details={"report_id": report_id, "status": status,
                                 "error": error})

    def list_reports(self) -> list[dict[str, Any]]:
        result = []
        for row in self.connection.execute(
            "SELECT * FROM report ORDER BY created_at DESC"
        ).fetchall():
            item = dict(row)
            item["spec"] = json.loads(item.pop("spec_json"))
            result.append(item)
        return result

    def report_detail(self, report_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM report WHERE report_id=?", (report_id,)
        ).fetchone()
        if row is None:
            raise KeyError(report_id)
        result = dict(row)
        result["spec"] = json.loads(result.pop("spec_json"))
        return result

    def write_metrics(self, run_id: str, rows: Iterable[dict[str, Any]]) -> None:
        with self.connection:
            for row in rows:
                self.connection.execute(
                    """INSERT OR REPLACE INTO factor_metric
                    (run_id, factor_id, index_code, period, value_type, orientation,
                     cost_scenario, metric_name, metric_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (run_id, row["factor_id"], row.get("index_code", ""),
                     row.get("period", "all"), row.get("value_type", "neutralized"),
                     row.get("orientation", "original"),
                     row.get("cost_scenario", "base_5bps"), row["metric_name"],
                     row.get("metric_value")),
                )

    def add_artifact(
        self, run_id: str, path: str | Path, kind: str,
        factor_id: str | None = None, metadata: dict[str, Any] | None = None,
    ) -> str:
        artifact_path = Path(path).resolve()
        digest = hashlib.sha256(storage_io.read_bytes(artifact_path)).hexdigest()
        artifact_id = str(uuid.uuid4())
        with self.connection:
            self.connection.execute(
                """INSERT INTO artifact
                (artifact_id, run_id, factor_id, kind, path, sha256, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (artifact_id, run_id, factor_id, kind, str(artifact_path), digest,
                 json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)),
            )
        return artifact_id

    def artifact_detail(self, artifact_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM artifact WHERE artifact_id=?", (artifact_id,)
        ).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def research_artifacts_for_factor(self, factor_id: str) -> list[dict[str, Any]]:
        """Run-level artifacts (performance/nav data) plus this factor's own charts.

        Chart artifacts carry ``factor_id``, so they must be filtered by it: a
        batch run contains many factors and their charts share the same
        (run, index, variant) keys. Performance/nav parquets are run-level
        (``factor_id`` is NULL) and are sliced by ``factor_name`` at read time.
        """
        rows = self.connection.execute(
            """SELECT DISTINCT a.*, r.created_at
            FROM artifact a
            JOIN research_run r ON r.run_id=a.run_id
            JOIN run_factor rf ON rf.run_id=a.run_id
            WHERE rf.factor_id=?
              AND (a.factor_id = rf.factor_id OR a.factor_id IS NULL)
              AND a.kind IN ('layered_performance', 'layered_nav', 'layered_nav_chart')
            ORDER BY r.created_at DESC, a.kind, a.path""",
            (factor_id,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            result.append(item)
        return result

    def metrics_for_factor(self, factor_id: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute(
            """SELECT m.*, r.mode, r.created_at
            FROM factor_metric m JOIN research_run r ON r.run_id=m.run_id
            WHERE m.factor_id=? ORDER BY r.created_at DESC, m.index_code, m.period""",
            (factor_id,),
        ).fetchall()]

    def promote(self, factor_id: str, run_id: str, note: str = "") -> None:
        run = self.connection.execute(
            "SELECT * FROM research_run WHERE run_id=?", (run_id,)
        ).fetchone()
        if run is None:
            raise KeyError(run_id)
        if run["mode"] != "formal" or run["status"] != "succeeded":
            raise ValueError("only successful formal runs can be promoted")
        linked = self.connection.execute(
            "SELECT revision_id FROM run_factor WHERE run_id=? AND factor_id=?",
            (run_id, factor_id),
        ).fetchone()
        if linked is None:
            raise ValueError("run does not contain factor")
        with self.connection:
            self.connection.execute(
                """INSERT INTO official_baseline
                (factor_id, run_id, revision_id, note, promoted_at) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(factor_id) DO UPDATE SET run_id=excluded.run_id,
                revision_id=excluded.revision_id, note=excluded.note,
                promoted_at=excluded.promoted_at""",
                (factor_id, run_id, linked["revision_id"], note, _now()),
            )
            self._audit("baseline_promoted", factor_id=factor_id,
                        revision_id=linked["revision_id"], run_id=run_id,
                        details={"note": note})

    def _audit(
        self, event_type: str, *, factor_id: str | None = None,
        revision_id: str | None = None, run_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.connection.execute(
            """INSERT INTO audit_event
            (event_id, event_type, factor_id, revision_id, run_id, details_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), event_type, factor_id, revision_id, run_id,
             json.dumps(details or {}, ensure_ascii=False, sort_keys=True), _now()),
        )
