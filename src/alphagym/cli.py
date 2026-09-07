from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from clickhouse_connect.driver.exceptions import ClickHouseError

from alphagym import storage_io
from alphagym.api import Workspace
from alphagym.combine import METHODS, factor_weights
from alphagym.config import resolve_root
from alphagym.equity_data import EquityDataBundle
from alphagym.factor_cache import FactorValueCache, build_factor_cache
from alphagym.factor_research_service import FactorResearchService
from alphagym.factor_store import FactorStore
from alphagym.factors import REGISTRY
from alphagym.factors.base import FactorDefinition
from alphagym.factors.compute import compute_factors
from alphagym.factors.library import seed_definitions
from alphagym.ingest import import_qmt
from alphagym.optional import OptionalDependencyError, require
from alphagym.report_engine import ReportEngine
from alphagym.report_spec import (
    FactorSelection,
    MetricFilter,
    ReportSpec,
    load_spec,
    resolve_factors,
)
from alphagym.reporting import build_series
from alphagym.research import evaluate_factor_batch
from alphagym.serialization import dumps
from alphagym.services import create_report
from alphagym.signal_export import export_signal


def _root(value: str | None) -> Path:
    return resolve_root(value)


def _emit(args: argparse.Namespace, value: object) -> None:
    if getattr(args, "json", False):
        print(dumps(value))
    else:
        print(value)


def _emit_task_result(args, row, keys) -> int:
    result = {key: row.get(key) for key in keys}
    result["ok"] = row["status"] == "succeeded"
    result["error"] = None if result["ok"] else {
        "code": "TaskFailed" if row["status"] == "failed" else "TaskCancelled",
        "message": row.get("error") or f"Task {row['status']}",
    }
    _emit(args, result)
    return 0 if result["ok"] else 1


def _cmd_data_audit(args: argparse.Namespace) -> int:
    bundle = EquityDataBundle.from_root(_root(args.root))
    result = bundle.audit(formal=not args.smoke, index_code=args.index)
    payload = {"ok": result.ok, "errors": result.errors, "warnings": result.warnings,
               "rows": result.rows}
    if not result.ok:
        payload["error"] = {"code": "DataContractError", "message": "; ".join(result.errors)}
    _emit(args, payload)
    return 0 if result.ok else 1


def _cmd_snapshot(args: argparse.Namespace) -> int:
    bundle = EquityDataBundle.from_root(_root(args.root))
    _emit(args, {"path": str(bundle.build_snapshot(args.destination, formal=not args.smoke))})
    return 0


def _cmd_import_qmt(args: argparse.Namespace) -> int:
    result = import_qmt(args.datadir, _root(args.root))
    if not args.no_refresh_factor_cache:
        _spawn_cache_build(_root(args.root), "--all")
    _emit(args, result)
    return 0


def _cmd_import_parquet(args):
    from alphagym.ingest import import_parquet

    _emit(args, import_parquet(args.input, args.table, _root(args.root), mode=args.mode))
    return 0


def _cmd_sync_tushare(args):
    from alphagym.tushare_import import sync_tushare

    _emit(args, sync_tushare(_root(args.root), token=args.token, start=args.start,
                            end=args.end, dataset=args.dataset, rate_limit=args.rate_limit,
                            overlap_days=args.overlap_days, daily_basic=not args.no_daily_basic,
                            symbols=args.symbol))
    return 0


def _cmd_storage(args):
    from alphagym.migration import migrate_workspace
    from alphagym.storage_io import store_for

    root = _root(args.root)
    if args.storage_command == "migrate":
        result = migrate_workspace(args.source, root)
    elif args.storage_command == "backup":
        from alphagym.database_admin import backup_database

        result = backup_database(root, args.name, base=args.base)
    elif args.storage_command == "restore":
        from alphagym.database_admin import restore_database

        result = restore_database(root, args.name, source_database=args.source_database)
    else:
        store = store_for(root, initialize=True)
        result = {"backend": "clickhouse", "database": store.config.database,
                  "workspace": store.config.workspace, "ready": store.available()}
    _emit(args, result)
    return 0


def _cmd_factor_list(args: argparse.Namespace) -> int:
    if getattr(args, "root", None):
        factors = Workspace(_root(args.root)).list_factors(args.family)
        if args.json:
            _emit(args, factors)
            return 0
        for factor in factors:
            print(f"{factor['name']}\t{factor['family']}\t"
                  f"{factor['hypothesis_id']}\t{factor['expected_direction']}")
        return 0
    specs = REGISTRY.list(args.family)
    if args.json:
        _emit(args, [
            {
                "name": spec.name, "family": spec.family,
                "hypothesis_id": spec.hypothesis_id,
                "expected_direction": spec.expected_direction,
            }
            for spec in specs
        ])
        return 0
    for spec in specs:
        print(f"{spec.name}\t{spec.family}\t{spec.hypothesis_id}\t{spec.expected_direction}")
    return 0


def _cmd_factor_sync(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        store.bootstrap(seed_definitions())
        _emit(args, {"path": str(store.path)})
    return 0


def _definition_from_args(args: argparse.Namespace) -> FactorDefinition:
    return FactorDefinition(
        factor_id=args.factor_id, name=args.name or args.factor_id,
        formula=args.formula, hypothesis_id=args.hypothesis_id,
        family=args.family, formula_version=args.formula_version,
        description=args.description or "", expected_direction=args.direction,
        tags=tuple(args.tag or ()), status=args.status,
    )


def _cmd_factor_save(args: argparse.Namespace) -> int:
    root = _root(args.root)
    with FactorStore.from_root(root) as store:
        revision = store.save_definition(_definition_from_args(args))
        _emit(args, {"revision_id": revision})
    if not args.no_cache_refresh:
        _spawn_cache_build(root, "--factor", args.factor_id)
    return 0


def _spawn_cache_build(root: Path, *extra: str) -> None:
    """Detached background factor-value cache refresh (space-for-time)."""
    subprocess.Popen(
        [
            sys.executable, "-m", "alphagym.cli", "factor", "cache-build",
            "--root", str(root), *extra,
        ],
        cwd=str(Path.cwd()),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _cmd_factor_cache_status(args: argparse.Namespace) -> int:
    root = _root(args.root)
    cache = FactorValueCache(root)
    manifest = cache.load_manifest()
    with FactorStore.from_root(root) as store:
        by_id = {item["factor_id"]: item for item in store.list_factors()}
    payload: dict[str, Any] = {
        "root": str(root),
        "signature_matches": cache.signature_matches(),
        "data_signature": manifest.get("data_signature"),
        "universes": {},
    }
    for key, universe in manifest.get("universes", {}).items():
        cached = [
            factor_id for factor_id, entry in universe.get("factors", {}).items()
            if factor_id in by_id
            and entry.get("revision_id") == by_id[factor_id]["current_revision_id"]
        ]
        payload["universes"][key] = {
            "index_code": universe.get("index_code"),
            "mode": universe.get("mode"),
            "cached": len(cached),
            "total": len(by_id),
            "missing": sorted(set(by_id) - set(cached)),
        }
    _emit(args, payload)
    return 0


def _cmd_factor_cache_build(args: argparse.Namespace) -> int:
    root = _root(args.root)
    with FactorStore.from_root(root) as store:
        by_id = {item["factor_id"]: item for item in store.list_factors()}
    if args.all:
        factor_ids = [
            factor_id for factor_id, item in by_id.items()
            if item["status"] != "deprecated"
        ]
    else:
        factor_ids = list(args.factor or [])
        if not factor_ids:
            raise ValueError("cache-build 需要 --factor 或 --all")
    summary = build_factor_cache(
        root,
        factor_ids,
        index_code=args.index,
        mode=args.mode,
        workers=args.workers,
        seed_from_panels=not args.no_seed,
        progress=lambda done, total, label: print(
            f"[cache-build] {done}/{total} {label}", file=sys.stderr, flush=True
        ),
    )
    _emit(args, summary)
    return 0


def _cmd_factor_validate(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        _emit(args, store.validate_formula(args.formula))
    return 0


def _cmd_factor_show(args: argparse.Namespace) -> int:
    _emit(args, Workspace(_root(args.root)).factor(args.factor_id))
    return 0


def _cmd_factor_export(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        _emit(args, {"path": str(store.export_catalog(args.output))})
    return 0


def _cmd_factor_import(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        _emit(args, {"imported": store.import_catalog(args.input)})
    return 0


def _cmd_factor_run(args: argparse.Namespace) -> int:
    workspace = Workspace(_root(args.root))
    workspace.initialize()
    factors = args.factor or [row["factor_id"] for row in workspace.list_factors()]
    if args.panel:
        result = workspace.run_panel(
            factors, args.panel, mode=args.mode,
            point_in_time_audit_passed=args.point_in_time_audit_passed,
        )
    else:
        if not args.start_date or not args.end_date:
            raise ValueError("automatic run requires --start-date and --end-date")
        result = workspace.run_factors(
            factors, start_date=args.start_date, end_date=args.end_date,
            index_code=args.index, mode=args.mode,
        )
    _emit(args, result)
    return 0


def _cmd_factor_execute_run(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        service = FactorResearchService(store)
        output = (
            service.execute_panel_run(args.run_id, args.panel)
            if args.panel
            else service.execute_auto_run(args.run_id)
        )
        _emit(args, {"path": str(output)})
    return 0


def _cmd_factor_runs(args: argparse.Namespace) -> int:
    _emit(args, Workspace(_root(args.root)).list_runs())
    return 0


def _cmd_factor_promote(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        store.promote(args.factor_id, args.run_id, args.note or "")
        _emit(args, {"factor_id": args.factor_id, "run_id": args.run_id})
    return 0


def _cmd_factor_serve(args: argparse.Namespace) -> int:
    uvicorn = require("uvicorn", "web")
    require("fastapi", "web")

    from alphagym.factor_web import create_app

    uvicorn.run(create_app(_root(args.root)), host="127.0.0.1", port=args.port)
    return 0


def _cmd_operator_list(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        items = store.operators.list()
        if args.json:
            _emit(args, [
                {"name": item.name, "version": item.version,
                 "code_hash": item.code_hash, "description": item.description}
                for item in items
            ])
            return 0
        for item in items:
            print(f"{item.name}\t{item.version}\t{item.code_hash}")
    return 0


def _cmd_field_list(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        items = store.fields.list()
        if args.json:
            _emit(args, [
                {
                    "name": item.name, "dataset": item.dataset, "column": item.column,
                    "dtype": item.dtype, "frequency": item.frequency,
                    "event_time": item.event_time, "available_time": item.available_time,
                    "formal": item.formal, "unit": item.unit, "description": item.description,
                }
                for item in items
            ])
            return 0
        for item in items:
            print(f"{item.name}\t{item.dataset}\t{item.frequency}\t{item.available_time or item.event_time}")
    return 0


def _cmd_factor_compute(args: argparse.Namespace) -> int:
    bundle = EquityDataBundle.from_root(_root(args.root))
    bundle.audit(formal=not args.smoke, index_code=args.index).require_ok()
    dates = [pd.Timestamp(value) for value in args.signal_date]
    result = compute_factors(bundle.daily, bundle.fundamentals, dates, args.factor)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    storage_io.write_frame(result, output, index=False)
    _emit(args, {"path": str(output)})
    return 0


def _cmd_single_factor(args: argparse.Namespace) -> int:
    panel = storage_io.read_frame(args.panel)
    monthly, summary = evaluate_factor_batch(panel)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    storage_io.write_csv(monthly, output / "monthly.csv", index=False)
    storage_io.write_csv(summary, output / "summary.csv", index=False)
    _emit(args, {"path": str(output)})
    return 0


def _cmd_combine(args: argparse.Namespace) -> int:
    history = storage_io.read_frame(args.history).set_index(args.date_column)
    weights = factor_weights(history, args.method)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    storage_io.write_csv(weights.rename_axis("factor_name").reset_index(), output, index=False)
    _emit(args, {"path": str(output)})
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    config = yaml.safe_load(storage_io.read_text(Path(args.config), encoding="utf-8")) if args.config else {}
    if not args.smoke:
        bundle = EquityDataBundle.from_root(_root(args.root or config.get("data_root")))
        for index in config.get("indices", ["000300.SH", "000905.SH", "000852.SH"]):
            bundle.audit(formal=True, index_code=index).require_ok()
    output = Path(args.output) if args.output else (
        _root(args.root or config.get("data_root")) / "artifacts" / "equity_factor_series"
    )
    metadata = {"config": str(Path(args.config).resolve())} if args.config else {}
    _emit(args, {"path": str(build_series(output, smoke=args.smoke, metadata=metadata))})
    return 0


# ------------------------------------------------------------ report commands


def _report_create(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    with FactorStore.from_root(_root(args.root)) as store:
        result = create_report(store, spec, ensure_runs=args.ensure_runs)
    if args.run:
        result.update(Workspace(_root(args.root)).start_report(result["report_id"]))
    _emit(args, result)
    return 0


def _report_execute(args: argparse.Namespace, report_id: str | None = None) -> int:
    target = report_id or args.report_id
    with FactorStore.from_root(_root(args.root)) as store:
        output = ReportEngine(store).execute_report(target)
    _emit(args, {
        "report_id": target, "status": "succeeded", "path": str(output),
    })
    return 0


def _report_status(args: argparse.Namespace) -> int:
    row = Workspace(_root(args.root)).report(args.report_id)
    _emit(args, {
        "report_id": row["report_id"], "name": row["name"],
        "status": row["status"], "progress": row["progress"],
        "error": row["error"], "run_id": row["run_id"], "path": row["path"],
        "created_at": row["created_at"], "completed_at": row["completed_at"],
    })
    return 0


def _report_wait(args: argparse.Namespace) -> int:
    row = Workspace(_root(args.root)).wait_report(
        args.report_id, timeout=args.timeout if args.timeout is not None else 3600,
    )
    return _emit_task_result(args, row, ("report_id", "status", "progress", "path"))


def _report_list(args: argparse.Namespace) -> int:
    rows = Workspace(_root(args.root)).list_reports()
    if args.status:
        rows = [row for row in rows if row["status"] in args.status.split(",")]
    if args.json:
        _emit(args, rows)
        return 0
    for row in rows:
        print(
            f"{row['report_id'][:8]}\t{row['status']}\t{row['progress']:.0%}\t"
            f"{row['name']}\t{row['created_at']}"
        )
    return 0


def _report_show(args: argparse.Namespace) -> int:
    _emit(args, Workspace(_root(args.root)).report_manifest(args.report_id))
    return 0


def _report_export(args: argparse.Namespace) -> int:
    root = _root(args.root)
    with FactorStore.from_root(root) as store:
        row = store.report_detail(args.report_id)
    report_dir = Path(row["path"]) if row.get("path") else (
        root / "factor_library" / "reports" / args.report_id
    )
    if not storage_io.exists(report_dir / "manifest.json"):
        raise FileNotFoundError(f"报告产物尚未生成：{report_dir}")
    output = Path(args.output).expanduser().resolve()
    for item in storage_io.iterdir(report_dir):
        if storage_io.exists(item) and not item.is_dir():
            storage_io.export_resource(item, output / item.name)
    _emit(args, {"report_id": args.report_id, "path": str(output)})
    return 0


def _report_build_family(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        families = {item["family"] for item in store.list_factors()}
        if args.family not in families:
            raise ValueError(f"未知家族：{args.family}；可用：{sorted(families)}")
        spec = ReportSpec(
            name=f"{args.family} 家族因子报告",
            mode=args.mode,
            factors=FactorSelection(
                families=[args.family],
                sort_by=MetricFilter(metric="rank_ic", period="validation"),
                top_n=args.top_n,
            ),
            description=f"家族 {args.family} 全部因子的自动汇总报告（按验证期 |Rank IC| 取前 {args.top_n}）。",
        )
        resolved = resolve_factors(store, spec.factors, index_code=spec.universe.index_code)
        report_id = store.create_report(spec.name, spec.to_dict())
    print(
        f"[build-family] {args.family}: {len(resolved)} 个因子 -> report {report_id}",
        file=sys.stderr, flush=True,
    )
    return _report_execute(args, report_id)


# -------------------------------------------------------------- run commands


def _cmd_signal_export(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        bundle = export_signal(store, args.report_id, args.method, top_n=args.top_n,
                               smooth_months=args.smooth_months)
    _emit(args, {
        "path": str(bundle.path),
        "method": bundle.method,
        "asof": str(bundle.asof.date()),
        "effective_trade_date": str(bundle.effective_trade_date.date()),
        "top_n": bundle.top_n,
        "symbols": len(bundle.symbols),
        "signal_csv": str(bundle.path / "signal_latest.csv"),
        "state_json": str(bundle.path / "state.json"),
    })
    return 0


# -------------------------------------------------------------- run commands


def _cmd_run_show(args: argparse.Namespace) -> int:
    _emit(args, Workspace(_root(args.root)).run(args.run_id))
    return 0


def _cmd_run_wait(args: argparse.Namespace) -> int:
    row = Workspace(_root(args.root)).wait_run(
        args.run_id, timeout=args.timeout if args.timeout is not None else 3600,
    )
    return _emit_task_result(args, row, ("run_id", "status", "progress", "completed_at"))


def _cmd_status(args: argparse.Namespace) -> int:
    _emit(args, Workspace(_root(args.root)).status())
    return 0


def _cmd_workflow(args: argparse.Namespace) -> int:
    from alphagym.workflows.catalog import describe_workflow, list_workflows, run_workflow

    if args.workflow_command == "list":
        _emit(args, list_workflows())
    elif args.workflow_command == "describe":
        _emit(args, describe_workflow(args.name))
    else:
        config = yaml.safe_load(storage_io.read_text(Path(args.config), encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Workflow config must be a mapping")
        if args.root:
            config["root"] = _root(args.root)
        result = run_workflow(args.name, config)
        _emit(args, result)
        return 0 if result["ok"] else 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alphagym",
        description="AlphaGYM — Quant Research Toolbox",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    workflow = commands.add_parser("workflow", help="Discover and run installed research workflows")
    workflow_commands = workflow.add_subparsers(dest="workflow_command", required=True)
    workflow_commands.add_parser("list").set_defaults(func=_cmd_workflow)
    describe = workflow_commands.add_parser("describe")
    describe.add_argument("name")
    describe.set_defaults(func=_cmd_workflow)
    execute = workflow_commands.add_parser("run")
    execute.add_argument("name")
    execute.add_argument("--config", required=True)
    execute.add_argument("--root")
    execute.set_defaults(func=_cmd_workflow)

    data = commands.add_parser("equity-data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    parquet = data_commands.add_parser("import-parquet")
    parquet.add_argument("--root")
    parquet.add_argument("--input", required=True)
    parquet.add_argument("--table", required=True, choices=list(EquityDataBundle.TABLES))
    parquet.add_argument("--mode", choices=("upsert", "replace"), default="upsert")
    parquet.set_defaults(func=_cmd_import_parquet)
    tushare = data_commands.add_parser("sync-tushare")
    tushare.add_argument("--root")
    tushare.add_argument("--token", help="Prefer TUSHARE_TOKEN environment variable")
    tushare.add_argument("--start")
    tushare.add_argument("--end")
    tushare.add_argument("--dataset", choices=("market", "securities", "fundamentals", "all"), default="market")
    tushare.add_argument("--symbol", action="append")
    tushare.add_argument("--rate-limit", type=float, default=.3)
    tushare.add_argument("--overlap-days", type=int, default=7)
    tushare.add_argument("--no-daily-basic", action="store_true")
    tushare.set_defaults(func=_cmd_sync_tushare)
    storage = commands.add_parser("storage")
    storage_commands = storage.add_subparsers(dest="storage_command", required=True)
    for operation in ("init", "migrate", "backup", "restore"):
        command = storage_commands.add_parser(operation)
        command.add_argument("--root")
        if operation == "migrate":
            command.add_argument("--source", required=True)
        if operation in {"backup", "restore"}:
            command.add_argument("--name", required=True)
        if operation == "backup":
            command.add_argument("--base")
        if operation == "restore":
            command.add_argument("--source-database", required=True)
        command.set_defaults(func=_cmd_storage)
    audit = data_commands.add_parser("audit")
    audit.add_argument("--root")
    audit.add_argument("--index")
    audit.add_argument("--smoke", action="store_true")
    audit.set_defaults(func=_cmd_data_audit)
    snapshot = data_commands.add_parser("build-snapshot")
    snapshot.add_argument("--root")
    snapshot.add_argument("--destination", required=True)
    snapshot.add_argument("--smoke", action="store_true")
    snapshot.set_defaults(func=_cmd_snapshot)
    import_qmt = data_commands.add_parser("import-qmt")
    import_qmt.add_argument("--datadir", required=True)
    import_qmt.add_argument("--root")
    import_qmt.add_argument("--no-refresh-factor-cache", action="store_true")
    import_qmt.set_defaults(func=_cmd_import_qmt)

    factor = commands.add_parser("factor")
    factor_commands = factor.add_subparsers(dest="factor_command", required=True)
    factor_list = factor_commands.add_parser("list")
    factor_list.add_argument("--family")
    factor_list.add_argument("--root")
    factor_list.add_argument("--json", action="store_true")
    factor_list.set_defaults(func=_cmd_factor_list)
    sync = factor_commands.add_parser("sync")
    sync.add_argument("--root")
    sync.set_defaults(func=_cmd_factor_sync)
    for command_name in ("create", "update"):
        save = factor_commands.add_parser(command_name)
        save.add_argument("--root")
        save.add_argument("--factor-id", required=True)
        save.add_argument("--name")
        save.add_argument("--formula", required=True)
        save.add_argument("--hypothesis-id", required=True)
        save.add_argument("--family", required=True)
        save.add_argument("--formula-version", default="1.0")
        save.add_argument("--description")
        save.add_argument("--direction", choices=("positive", "negative", "unknown"), default="unknown")
        save.add_argument("--status", choices=("active", "blocked", "deprecated"), default="active")
        save.add_argument("--tag", action="append")
        save.add_argument("--no-cache-refresh", action="store_true")
        save.add_argument("--json", action="store_true")
        save.set_defaults(func=_cmd_factor_save)
    validate = factor_commands.add_parser("validate")
    validate.add_argument("--root")
    validate.add_argument("--formula", required=True)
    validate.set_defaults(func=_cmd_factor_validate)
    show = factor_commands.add_parser("show")
    show.add_argument("--root")
    show.add_argument("factor_id")
    show.set_defaults(func=_cmd_factor_show)
    export = factor_commands.add_parser("export")
    export.add_argument("--root")
    export.add_argument("--output", required=True)
    export.set_defaults(func=_cmd_factor_export)
    import_ = factor_commands.add_parser("import")
    import_.add_argument("--root")
    import_.add_argument("--input", required=True)
    import_.set_defaults(func=_cmd_factor_import)
    run = factor_commands.add_parser("run")
    run.add_argument("--root")
    run.add_argument("--panel")
    run.add_argument("--factor", action="append")
    run.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    run.add_argument("--index", default="ALL_A")
    run.add_argument("--start-date")
    run.add_argument("--end-date")
    run.add_argument("--point-in-time-audit-passed", action="store_true")
    run.set_defaults(func=_cmd_factor_run)
    execute_run = factor_commands.add_parser("_execute-run")
    execute_run.add_argument("--root")
    execute_run.add_argument("--run-id", required=True)
    execute_run.add_argument("--panel")
    execute_run.set_defaults(func=_cmd_factor_execute_run)
    runs = factor_commands.add_parser("runs")
    runs.add_argument("--root")
    runs.set_defaults(func=_cmd_factor_runs)
    promote = factor_commands.add_parser("promote")
    promote.add_argument("--root")
    promote.add_argument("--factor-id", required=True)
    promote.add_argument("--run-id", required=True)
    promote.add_argument("--note")
    promote.add_argument("--json", action="store_true")
    promote.set_defaults(func=_cmd_factor_promote)
    serve = factor_commands.add_parser("serve")
    serve.add_argument("--root")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=_cmd_factor_serve)
    compute = factor_commands.add_parser("compute")
    compute.add_argument("--root")
    compute.add_argument("--index", required=True)
    compute.add_argument("--signal-date", action="append", required=True)
    compute.add_argument("--factor", action="append")
    compute.add_argument("--output", required=True)
    compute.add_argument("--smoke", action="store_true")
    compute.set_defaults(func=_cmd_factor_compute)
    cache_status = factor_commands.add_parser("cache-status")
    cache_status.add_argument("--root")
    cache_status.add_argument("--json", action="store_true")
    cache_status.set_defaults(func=_cmd_factor_cache_status)
    cache_build = factor_commands.add_parser("cache-build")
    cache_build.add_argument("--root")
    cache_build.add_argument("--factor", action="append")
    cache_build.add_argument("--all", action="store_true")
    cache_build.add_argument("--index", default="ALL_A")
    cache_build.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    cache_build.add_argument("--workers", type=int, default=4)
    cache_build.add_argument("--no-seed", action="store_true")
    cache_build.add_argument("--json", action="store_true")
    cache_build.set_defaults(func=_cmd_factor_cache_build)

    research = commands.add_parser("research")
    research_commands = research.add_subparsers(dest="research_command", required=True)
    single = research_commands.add_parser("single-factor")
    single.add_argument("--panel", required=True)
    single.add_argument("--output", required=True)
    single.set_defaults(func=_cmd_single_factor)
    combine = research_commands.add_parser("combine-factors")
    combine.add_argument("--history", required=True)
    combine.add_argument("--date-column", default="signal_date")
    combine.add_argument("--method", choices=tuple(METHODS), required=True)
    combine.add_argument("--output", required=True)
    combine.set_defaults(func=_cmd_combine)

    signal = commands.add_parser("signal")
    signal_commands = signal.add_subparsers(dest="signal_command", required=True)
    signal_export = signal_commands.add_parser("export")
    signal_export.add_argument("--root")
    signal_export.add_argument("--report-id", required=True)
    signal_export.add_argument("--method", required=True, help="combo.json 中的合成方法 key，如 ml_xgb 或 equal")
    signal_export.add_argument("--top-n", type=int, default=50)
    signal_export.add_argument("--smooth-months", type=int, default=1,
                               help="分数平滑月数（>1 用滚动均值压低换手）")
    signal_export.add_argument("--json", action="store_true")
    signal_export.set_defaults(func=_cmd_signal_export)

    report = commands.add_parser("report")
    report_commands = report.add_subparsers(dest="report_command", required=True)
    series = report_commands.add_parser("build-series")
    series.add_argument("--config")
    series.add_argument("--root")
    series.add_argument("--output")
    series.add_argument("--smoke", action="store_true")
    series.set_defaults(func=_cmd_report)
    report_create = report_commands.add_parser("create")
    report_create.add_argument("--root")
    report_create.add_argument("--spec", required=True, help="报告 spec 的 YAML/JSON 文件路径")
    report_create.add_argument("--run", action="store_true", help="创建后启动后台任务，用 report wait 等待")
    report_create.add_argument("--ensure-runs", action="store_true",
                               help="先为缺少覆盖回测的因子补跑自动回测")
    report_create.add_argument("--json", action="store_true")
    report_create.set_defaults(func=_report_create)
    report_run = report_commands.add_parser("run")
    report_run.add_argument("--root")
    report_run.add_argument("--report-id", required=True)
    report_run.add_argument("--json", action="store_true")
    report_run.set_defaults(func=_report_execute)
    report_status = report_commands.add_parser("status")
    report_status.add_argument("--root")
    report_status.add_argument("--report-id", required=True)
    report_status.add_argument("--json", action="store_true")
    report_status.set_defaults(func=_report_status)
    report_wait = report_commands.add_parser("wait")
    report_wait.add_argument("--root")
    report_wait.add_argument("--report-id", required=True)
    report_wait.add_argument("--timeout", type=float)
    report_wait.add_argument("--json", action="store_true")
    report_wait.set_defaults(func=_report_wait)
    report_list = report_commands.add_parser("list")
    report_list.add_argument("--root")
    report_list.add_argument("--status", help="逗号分隔的状态过滤，如 succeeded,failed")
    report_list.add_argument("--json", action="store_true")
    report_list.set_defaults(func=_report_list)
    report_show = report_commands.add_parser("show")
    report_show.add_argument("--root")
    report_show.add_argument("--report-id", required=True)
    report_show.add_argument("--json", action="store_true")
    report_show.set_defaults(func=_report_show)
    report_export = report_commands.add_parser("export")
    report_export.add_argument("--root")
    report_export.add_argument("--report-id", required=True)
    report_export.add_argument("--output", required=True)
    report_export.add_argument("--json", action="store_true")
    report_export.set_defaults(func=_report_export)
    report_family = report_commands.add_parser("build-family")
    report_family.add_argument("--root")
    report_family.add_argument("--family", required=True)
    report_family.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    report_family.add_argument("--top-n", type=int, default=20)
    report_family.add_argument("--json", action="store_true")
    report_family.set_defaults(func=_report_build_family)

    run = commands.add_parser("run")
    run_commands = run.add_subparsers(dest="run_command", required=True)
    run_show = run_commands.add_parser("show")
    run_show.add_argument("--root")
    run_show.add_argument("--run-id", required=True)
    run_show.add_argument("--json", action="store_true")
    run_show.set_defaults(func=_cmd_run_show)
    run_wait = run_commands.add_parser("wait")
    run_wait.add_argument("--root")
    run_wait.add_argument("--run-id", required=True)
    run_wait.add_argument("--timeout", type=float)
    run_wait.add_argument("--json", action="store_true")
    run_wait.set_defaults(func=_cmd_run_wait)

    status = commands.add_parser("status")
    status.add_argument("--root")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=_cmd_status)

    operator = commands.add_parser("operator")
    operator_commands = operator.add_subparsers(dest="operator_command", required=True)
    operator_list = operator_commands.add_parser("list")
    operator_list.add_argument("--root")
    operator_list.add_argument("--json", action="store_true")
    operator_list.set_defaults(func=_cmd_operator_list)

    field = commands.add_parser("field")
    field_commands = field.add_subparsers(dest="field_command", required=True)
    field_list = field_commands.add_parser("list")
    field_list.add_argument("--root")
    field_list.add_argument("--json", action="store_true")
    field_list.set_defaults(func=_cmd_field_list)
    def add_json_flags(command_parser):
        if not any(action.dest == "json" for action in command_parser._actions):
            command_parser.add_argument("--json", action="store_true")
        for action in command_parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for child in action.choices.values():
                    add_json_flags(child)
    add_json_flags(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in arguments
    args = parser.parse_args([item for item in arguments if item != "--json"])
    args.json = as_json
    driver_logger = logging.getLogger("clickhouse_connect.driver.httpclient")
    prior_disabled = driver_logger.disabled
    if as_json:
        driver_logger.disabled = True
    try:
        return int(args.func(args))
    except (ValueError, KeyError, OSError, OptionalDependencyError, sqlite3.Error, yaml.YAMLError, ClickHouseError) as error:
        if isinstance(error, ClickHouseError):
            from alphagym.storage import StorageError

            error = StorageError("ClickHouse operation failed; check service, schema and permissions")
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {"ok": False, "error": {
                        "code": type(error).__name__, "message": str(error)}},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
        else:
            print(f"错误：{error}", file=sys.stderr)
        return 1
    finally:
        driver_logger.disabled = prior_disabled
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
