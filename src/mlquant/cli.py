from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mlquant.adapters import QmtDailyAdapter, QmtDividendAdapter
from mlquant.combine import METHODS, factor_weights
from mlquant.equity_data import DataContractError, EquityDataBundle
from mlquant.factor_cache import FactorValueCache, build_factor_cache
from mlquant.factor_research_service import FactorResearchService
from mlquant.factor_store import FactorStore
from mlquant.factors import REGISTRY
from mlquant.factors.base import FactorDefinition
from mlquant.factors.compute import compute_factors
from mlquant.factors.library import seed_definitions
from mlquant.report_engine import ReportEngine
from mlquant.report_spec import (
    FactorSelection,
    MetricFilter,
    ReportSpec,
    load_spec,
    resolve_factors,
)
from mlquant.reporting import build_series
from mlquant.research import evaluate_factor_batch
from mlquant.signal_export import export_signal


def _root(value: str | None) -> Path:
    resolved = value or os.environ.get("MLQUANT_DATA_ROOT")
    if not resolved:
        raise DataContractError("--root or MLQUANT_DATA_ROOT is required")
    return Path(resolved).expanduser().resolve()


def _emit(args: argparse.Namespace, value: object) -> None:
    if getattr(args, "json", False):
        print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    else:
        print(value)


def _cmd_data_audit(args: argparse.Namespace) -> int:
    bundle = EquityDataBundle.from_root(_root(args.root))
    result = bundle.audit(formal=not args.smoke, index_code=args.index)
    print(json.dumps({"ok": result.ok, "errors": result.errors, "warnings": result.warnings, "rows": result.rows}, ensure_ascii=False, indent=2))
    return 0 if result.ok else 2


def _cmd_snapshot(args: argparse.Namespace) -> int:
    bundle = EquityDataBundle.from_root(_root(args.root))
    print(bundle.build_snapshot(args.destination, formal=not args.smoke))
    return 0


def _cmd_import_qmt(args: argparse.Namespace) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    datadir = Path(args.datadir).expanduser().resolve()
    if not datadir.is_dir():
        raise DataContractError(f"QMT datadir not found: {datadir}")
    target = _root(args.root) / "equity"
    target.mkdir(parents=True, exist_ok=True)
    adjustment_path = target / "adjustments.parquet"
    adjustment_temp = target / "adjustments.parquet.tmp"
    QmtDividendAdapter(datadir).read().to_parquet(adjustment_temp, index=False)
    adjustment_temp.replace(adjustment_path)
    daily = QmtDailyAdapter(datadir)
    symbols = daily.symbols()
    daily_path = target / "daily.parquet"
    daily_temp = target / "daily.parquet.tmp"
    writer = None
    for start in range(0, len(symbols), 500):
        frame = daily.read(symbols[start : start + 500])
        if frame.empty:
            continue
        table = pa.Table.from_pandas(frame)
        if writer is None:
            writer = pq.ParquetWriter(daily_temp, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()
        daily_temp.replace(daily_path)
    print(target)
    if not args.no_refresh_factor_cache:
        # An incremental price/volume update changes the data signature, which
        # invalidates the whole factor-value cache; rebuild it in the
        # background so backtests stay fast and point-in-time correct.
        _spawn_cache_build(_root(args.root), "--all")
    return 0


def _cmd_factor_list(args: argparse.Namespace) -> int:
    if getattr(args, "root", None):
        with FactorStore.from_root(_root(args.root)) as store:
            factors = store.list_factors(args.family)
            if args.json:
                _emit(args, factors)
                return 0
            for factor in factors:
                print(
                    f"{factor['name']}\t{factor['family']}\t"
                    f"{factor['hypothesis_id']}\t{factor['expected_direction']}"
                )
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
        print(store.path)
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
            sys.executable, "-m", "mlquant.cli", "factor", "cache-build",
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
        print(json.dumps(store.validate_formula(args.formula), ensure_ascii=False, indent=2))
    return 0


def _cmd_factor_show(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        print(json.dumps(store.factor_detail(args.factor_id), ensure_ascii=False, indent=2))
    return 0


def _cmd_factor_export(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        print(store.export_catalog(args.output))
    return 0


def _cmd_factor_import(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        print(store.import_catalog(args.input))
    return 0


def _cmd_factor_run(args: argparse.Namespace) -> int:
    root = _root(args.root)
    with FactorStore.from_root(root) as store:
        store.bootstrap(seed_definitions())
        factors = args.factor or [row["factor_id"] for row in store.list_factors()]
        if args.panel:
            config = {
                "panel": str(Path(args.panel).expanduser().resolve()),
                "point_in_time_audit_passed": bool(args.point_in_time_audit_passed),
            }
        else:
            if not args.start_date or not args.end_date:
                raise ValueError("automatic run requires --start-date and --end-date")
            config = {
                "source": "automatic",
                "index_code": args.index,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "point_in_time_audit_passed": args.mode == "formal",
            }
        run_id = store.create_run(factors, mode=args.mode, config=config)
        service = FactorResearchService(store)
        output = (
            service.execute_panel_run(run_id, args.panel)
            if args.panel
            else service.execute_auto_run(run_id)
        )
        print(output)
    return 0


def _cmd_factor_execute_run(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        service = FactorResearchService(store)
        output = (
            service.execute_panel_run(args.run_id, args.panel)
            if args.panel
            else service.execute_auto_run(args.run_id)
        )
        print(output)
    return 0


def _cmd_factor_runs(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        print(json.dumps(store.list_runs(), ensure_ascii=False, indent=2))
    return 0


def _cmd_factor_promote(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        store.promote(args.factor_id, args.run_id, args.note or "")
        _emit(args, {"factor_id": args.factor_id, "run_id": args.run_id})
    return 0


def _cmd_factor_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from mlquant.factor_web import create_app

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
    result.to_parquet(output, index=False)
    print(output)
    return 0


def _cmd_single_factor(args: argparse.Namespace) -> int:
    panel = pd.read_parquet(args.panel)
    monthly, summary = evaluate_factor_batch(panel)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output / "monthly.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    print(output)
    return 0


def _cmd_combine(args: argparse.Namespace) -> int:
    history = pd.read_parquet(args.history).set_index(args.date_column)
    weights = factor_weights(history, args.method)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    weights.rename_axis("factor_name").reset_index().to_csv(output, index=False)
    print(output)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not args.smoke:
        bundle = EquityDataBundle.from_root(_root(args.root or config.get("data_root")))
        for index in config["indices"]:
            bundle.audit(formal=True, index_code=index).require_ok()
    default_base = Path(args.root or os.environ.get("MLQUANT_DATA_ROOT", ".")).expanduser().resolve()
    output = Path(args.output or (default_base / "artifacts" / "equity_factor_series"))
    print(build_series(output, smoke=args.smoke, metadata={"config": str(Path(args.config).resolve())}))
    return 0


# ------------------------------------------------------------ report commands


def _report_create(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    with FactorStore.from_root(_root(args.root)) as store:
        resolved = resolve_factors(store, spec.factors, index_code=spec.universe.index_code)
        minimum = 1 if spec.combine is None else 2
        if len(resolved) < minimum:
            if spec.combine is None:
                raise ValueError("报告至少需要 1 个因子")
            raise ValueError("报告至少需要 2 个因子（相关矩阵与合成对比依赖截面数据）")
        if args.ensure_runs:
            missing = ReportEngine(store).missing_run_factors(spec, resolved)
            if missing:
                print(
                    f"[ensure-runs] 补跑 {len(missing)} 个因子："
                    + "、".join(item["factor_id"] for item in missing),
                    file=sys.stderr, flush=True,
                )
                ReportEngine(store).backfill_runs(spec, resolved)
        report_id = store.create_report(spec.name, spec.to_dict())
    if args.run:
        return _report_execute(args, report_id)
    _emit(args, {
        "report_id": report_id, "status": "queued",
        "name": spec.name, "factors": len(resolved),
    })
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
    with FactorStore.from_root(_root(args.root)) as store:
        row = store.report_detail(args.report_id)
    _emit(args, {
        "report_id": row["report_id"], "name": row["name"],
        "status": row["status"], "progress": row["progress"],
        "error": row["error"], "run_id": row["run_id"], "path": row["path"],
        "created_at": row["created_at"], "completed_at": row["completed_at"],
    })
    return 0


def _report_wait(args: argparse.Namespace) -> int:
    deadline = time.time() + float(args.timeout or 3600)
    with FactorStore.from_root(_root(args.root)) as store:
        while True:
            row = store.report_detail(args.report_id)
            if row["status"] in {"succeeded", "failed", "cancelled"}:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"报告 {args.report_id} 在 {args.timeout or 3600}s 内未完成（当前 {row['status']}）"
                )
            time.sleep(2)
    _emit(args, {
        "report_id": row["report_id"], "status": row["status"],
        "progress": row["progress"], "error": row["error"], "path": row["path"],
    })
    return 0 if row["status"] == "succeeded" else 1


def _report_list(args: argparse.Namespace) -> int:
    with FactorStore.from_root(_root(args.root)) as store:
        rows = store.list_reports()
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
    root = _root(args.root)
    with FactorStore.from_root(root) as store:
        row = store.report_detail(args.report_id)
    report_dir = Path(row["path"]) if row.get("path") else (
        root / "factor_library" / "reports" / args.report_id
    )
    manifest_path = report_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"报告产物尚未生成：{report_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _emit(args, manifest)
    return 0


def _report_export(args: argparse.Namespace) -> int:
    root = _root(args.root)
    with FactorStore.from_root(root) as store:
        row = store.report_detail(args.report_id)
    report_dir = Path(row["path"]) if row.get("path") else (
        root / "factor_library" / "reports" / args.report_id
    )
    if not report_dir.is_dir():
        raise FileNotFoundError(f"报告产物尚未生成：{report_dir}")
    output = Path(args.output).expanduser().resolve()
    shutil.copytree(report_dir, output, dirs_exist_ok=True)
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
    with FactorStore.from_root(_root(args.root)) as store:
        row = store.run_detail(args.run_id)
    _emit(args, row)
    return 0


def _cmd_run_wait(args: argparse.Namespace) -> int:
    deadline = time.time() + float(args.timeout or 3600)
    with FactorStore.from_root(_root(args.root)) as store:
        while True:
            row = store.connection.execute(
                "SELECT status, progress, error, completed_at FROM research_run WHERE run_id=?",
                (args.run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(args.run_id)
            if row["status"] in {"succeeded", "failed", "cancelled"}:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"run {args.run_id} 在 {args.timeout or 3600}s 内未完成（当前 {row['status']}）"
                )
            time.sleep(2)
    _emit(args, {
        "run_id": args.run_id, "status": row["status"],
        "progress": row["progress"], "error": row["error"],
        "completed_at": row["completed_at"],
    })
    return 0 if row["status"] == "succeeded" else 1


def _cmd_status(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    root = _root(args.root)
    payload: dict[str, object] = {"root": str(root), "equity": {}, "factor_library": {}}
    for name in ("daily", "adjustments", "fundamentals", "index_members",
                 "industries", "calendar", "securities", "status"):
        path = root / "equity" / f"{name}.parquet"
        if not path.is_file():
            continue
        payload["equity"][name] = {"size": path.stat().st_size}
        try:
            schema = pq.ParquetFile(path).schema_arrow
            payload["equity"][name]["columns"] = sorted(schema.names)
            payload["equity"][name]["rows"] = pq.ParquetFile(path).metadata.num_rows
        except Exception as error:  # noqa: BLE001 - best-effort schema reporting
            payload["equity"][name]["schema_error"] = str(error)
    with FactorStore.from_root(root) as store:
        payload["factor_library"] = {
            "factors": store.connection.execute("SELECT COUNT(*) FROM factor").fetchone()[0],
            "runs_succeeded": store.connection.execute(
                "SELECT COUNT(*) FROM research_run WHERE status='succeeded'"
            ).fetchone()[0],
            "runs_active": [
                dict(row) for row in store.connection.execute(
                    "SELECT run_id, status, progress FROM research_run "
                    "WHERE status IN ('queued','running') ORDER BY created_at"
                ).fetchall()
            ],
            "reports": store.connection.execute("SELECT COUNT(*) FROM report").fetchone()[0],
            "reports_active": [
                dict(row) for row in store.connection.execute(
                    "SELECT report_id, status, progress FROM report "
                    "WHERE status IN ('queued','running') ORDER BY created_at"
                ).fetchall()
            ],
            "latest_run": dict(row) if (
                row := store.connection.execute(
                    "SELECT run_id, status, completed_at FROM research_run "
                    "WHERE status='succeeded' ORDER BY completed_at DESC LIMIT 1"
                ).fetchone()
            ) else None,
        }
    _emit(args, payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlquant", description="Point-in-time A-share factor research")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("equity-data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
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
    series.add_argument("--config", default="config/equity_research.yaml")
    series.add_argument("--root")
    series.add_argument("--output")
    series.add_argument("--smoke", action="store_true")
    series.set_defaults(func=_cmd_report)
    report_create = report_commands.add_parser("create")
    report_create.add_argument("--root")
    report_create.add_argument("--spec", required=True, help="报告 spec 的 YAML/JSON 文件路径")
    report_create.add_argument("--run", action="store_true", help="创建后立即同步执行")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DataContractError as error:
        parser.error(str(error))
    except (ValueError, KeyError, FileNotFoundError, TimeoutError) as error:
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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
