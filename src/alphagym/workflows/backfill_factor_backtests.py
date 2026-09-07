"""Backfill backtests for every currently-runnable factor, in batches.

One research run per batch so each generated panel stays manageable and the
webui factor dashboard can load panels per family. Re-runnable: factors whose
latest succeeded run already used the same config are skipped.

Set ``--workers N`` (N > 1) to run several batches concurrently in separate
processes; each worker opens its own factor store and computes its own batch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from alphagym.factor_research_service import AutoRunDataError, FactorResearchService
from alphagym.factor_store import FactorStore
from alphagym.report_engine import ReportEngine
from alphagym.report_spec import FactorSelection, MetricFilter, ReportSpec, resolve_factors
from alphagym.workflows._support import finish, invoke, progress, validate_config

START_DATE = "2014-01-01"
END_DATE = "2025-12-31"
INDEX_CODE = "ALL_A"
BATCH_SIZE = 20


def _family_order(factors: list[dict[str, str]]) -> list[str]:
    order = (
        "value", "growth", "momentum", "liquidity", "risk", "quality", "technical",
    )
    ordered: list[str] = []
    for family in order:
        ordered.extend(item["factor_id"] for item in factors if item["family"] == family)
    ordered.extend(
        item["factor_id"]
        for item in factors
        if item["family"] not in order
    )
    return ordered


def _config(mode: str, start_date: str = START_DATE, end_date: str = END_DATE) -> dict[str, object]:
    return {
        "source": "automatic",
        "index_code": INDEX_CODE,
        "start_date": start_date,
        "end_date": end_date,
        "point_in_time_audit_passed": mode == "formal",
    }


def _already_succeeded(
    store: FactorStore, factor_id: str, config: dict[str, object], mode: str,
) -> bool:
    rows = store.connection.execute(
        """SELECT r.mode, r.config_json
        FROM research_run r JOIN run_factor rf ON rf.run_id = r.run_id
        WHERE rf.factor_id = ? AND r.status = 'succeeded'
        ORDER BY r.created_at DESC LIMIT 1""",
        (factor_id,),
    ).fetchall()
    if not rows:
        return False
    latest = json.loads(rows[0]["config_json"])
    return (
        rows[0]["mode"] == mode
        and latest.get("index_code") == config["index_code"]
        and latest.get("start_date") == config["start_date"]
        and latest.get("end_date") == config["end_date"]
    )


def _collect_todo(
    store: FactorStore, factor_ids: list[str], mode: str, market_only: bool,
    start_date: str, end_date: str,
) -> tuple[list[str], list[str]]:
    service = FactorResearchService(store)
    registry = store.load_registry() if market_only else None
    todo: list[str] = []
    blocked: list[str] = []
    config = _config(mode, start_date, end_date)
    for factor_id in factor_ids:
        if market_only and registry is not None:
            fields = registry.get(factor_id).input_fields
            if any(field.startswith("financial.") for field in fields):
                continue
        try:
            service.validate_auto_request(
                [factor_id], mode=mode, index_code=INDEX_CODE,
                start_date=start_date, end_date=end_date,
            )
        except (AutoRunDataError, ValueError):
            blocked.append(factor_id)
            continue
        if not _already_succeeded(store, factor_id, config, mode):
            todo.append(factor_id)
    return todo, blocked


def _run_batch(
    store: FactorStore, batch: list[str], mode: str, start_date: str, end_date: str,
) -> tuple[str, str]:
    service = FactorResearchService(store)
    run_id = store.create_run(batch, mode=mode, config=_config(mode, start_date, end_date))
    try:
        output = service.execute_auto_run(run_id)
        return run_id, f"OK -> {output}"
    except Exception as error:  # noqa: BLE001 - one bad batch must not stop the rest
        return run_id, f"FAILED: {error}"


def _spawn_child(
    root: str, mode: str, market_only: bool, batch: list[str], start_date: str, end_date: str,
) -> subprocess.Popen:
    command = [
        sys.executable, "-m", "alphagym.workflows.backfill_factor_backtests",
        "--root", root, "--mode", mode, "--factors", ",".join(batch),
        "--start-date", start_date, "--end-date", end_date,
    ]
    if market_only:
        command.append("--market-only")
    return subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _build_family_reports(store: FactorStore, mode: str, top_n: int) -> None:
    """Pre-built per-family research reports so the report area opens instantly."""
    families = sorted({item["family"] for item in store.list_factors()})
    engine = ReportEngine(store)
    for family in families:
        spec = ReportSpec(
            name=f"{family} 家族因子报告",
            mode=mode,
            factors=FactorSelection(
                families=[family],
                sort_by=MetricFilter(metric="rank_ic", period="validation"),
                top_n=top_n,
            ),
            description=f"家族 {family} 全部因子的自动汇总报告（按验证期 |Rank IC| 取前 {top_n}）。",
        )
        resolved = resolve_factors(store, spec.factors, index_code=spec.universe.index_code)
        if len(resolved) < 2:
            progress(f"[reports] skip family {family}: only {len(resolved)} factors", flush=True)
            continue
        progress(
            f"[reports] build {family}: {len(resolved)} factors", flush=True
        )
        try:
            report_id = store.create_report(spec.name, spec.to_dict())
            output = engine.execute_report(report_id)
            progress(f"[reports] done {family} -> {output}", flush=True)
        except Exception as error:  # noqa: BLE001 - one bad family must not stop the rest
            progress(f"[reports] FAILED {family}: {error}", flush=True)




@dataclass(frozen=True, slots=True, kw_only=True)
class Config:
    """Explicit workflow inputs; paths never depend on the source checkout."""
    root: Path
    mode: str = 'formal'
    start_date: str = START_DATE
    end_date: str = END_DATE
    market_only: bool = False
    factors: str | None = None
    workers: int = 1
    no_reports: bool = False
    report_top_n: int = 20
    json: bool = False

    def __post_init__(self):
        validate_config(self, {'mode': ('smoke', 'formal')})


def run(args: Config) -> dict:
    """Execute with Python inputs and return artifact metadata; never exits the host."""
    root = str(Path(args.root).expanduser().resolve())
    mode: str = args.mode

    with FactorStore.from_root(root) as store:
        if args.factors:
            explicit = [item for item in args.factors.split(",") if item]
            todo, blocked = _collect_todo(
                store, explicit, mode, args.market_only, args.start_date, args.end_date
            )
            if blocked:
                raise AutoRunDataError("Requested factors lack valid inputs: " + ", ".join(blocked))
            if not todo:
                progress(f"batch skipped: all {len(explicit)} factors already succeeded", flush=True)
                return finish(output=None)
            run_id, message = _run_batch(store, todo, mode, args.start_date, args.end_date)
            progress(f"[batch] run={run_id} factors={len(todo)} {message}", flush=True)
            return finish(0 if store.run_detail(run_id)["status"] == "succeeded" else 1,
                          results=[{"run_id": run_id, "message": message}])

        factor_ids = [item["factor_id"] for item in store.list_factors()]
        pending, blocked = _collect_todo(
            store, factor_ids, mode, args.market_only, args.start_date, args.end_date
        )
        # Reorder into families for a coherent dashboard panel.
        families = {item["factor_id"]: item["family"] for item in store.list_factors()}
        todo = _family_order(
            [{"factor_id": fid, "family": families[fid]} for fid in pending]
        )
        if not todo:
            progress("nothing to do: all runnable factors already backtested", flush=True)
            if not args.no_reports:
                _build_family_reports(store, mode, args.report_top_n)
            return finish(output=None)
        batches = [todo[pos:pos + BATCH_SIZE] for pos in range(0, len(todo), BATCH_SIZE)]
        progress(
            json.dumps(
                {"todo": len(todo), "blocked": len(blocked),
                 "batches": len(batches), "workers": args.workers},
                ensure_ascii=False,
            ),
            flush=True,
        )

        if args.workers <= 1:
            results = []
            for number, batch in enumerate(batches, start=1):
                started = time.time()
                progress(f"[batch {number}/{len(batches)}] factors={len(batch)} first={batch[0]}", flush=True)
                run_id, message = _run_batch(
                    store, batch, mode, args.start_date, args.end_date
                )
                progress(f"[batch {number}] {time.time() - started:.0f}s run={run_id} {message}", flush=True)
                results.append({"run_id": run_id, "status": store.run_detail(run_id)["status"]})
            if not args.no_reports:
                _build_family_reports(store, mode, args.report_top_n)
            return finish(0 if all(row["status"] == "succeeded" for row in results) else 1,
                          results=results)

        # Parallel: spawn one child process per batch, capped at --workers.
        running: list[tuple[str, list[str], subprocess.Popen]] = []
        results = []
        next_batch = 0
        while next_batch < len(batches) or running:
            while next_batch < len(batches) and len(running) < args.workers:
                batch = batches[next_batch]
                process = _spawn_child(
                    root, mode, args.market_only, batch, args.start_date, args.end_date
                )
                progress(f"[spawn {next_batch + 1}/{len(batches)}] factors={len(batch)} first={batch[0]}", flush=True)
                running.append((f"batch {next_batch + 1}", batch, process))
                next_batch += 1
            finished = [item for item in running if item[2].poll() is not None]
            for label, batch, process in finished:
                code = process.wait()
                results.append({"factors": len(batch), "label": label, "exit_code": code})
                progress(f"[done] {label} exit={code}", flush=True)
                running.remove((label, batch, process))
            if finished:
                continue
            time.sleep(2)
        progress(json.dumps({"summary": results}, ensure_ascii=False), flush=True)
        if not args.no_reports:
            _build_family_reports(store, mode, args.report_top_n)
    return finish(0 if all(row["exit_code"] == 0 for row in results) else 1, results=results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument(
        "--market-only", action="store_true",
        help="only backtest factors whose inputs are pure price-volume fields (market.*)",
    )
    parser.add_argument("--factors", help="comma-separated explicit factor list (single batch)")
    parser.add_argument("--workers", type=int, default=1, help="concurrent batch processes")
    parser.add_argument(
        "--no-reports", action="store_true",
        help="skip pre-built family reports after the backfill",
    )
    parser.add_argument("--report-top-n", type=int, default=20,
                        help="factor count per pre-built family report")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    return invoke(run, Config, args)


if __name__ == "__main__":
    raise SystemExit(main())
