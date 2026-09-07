"""Backfill single-factor auto-backtests for the curated ML pool on broad indices.

The standard ``scripts/backfill_factor_backtests.py`` only covers ALL_A. The
three broad indices (CSI300/CSI500/CSI1000) already have point-in-time members
with official benchmark weights, so the curated ML pool can be backtested on
them the same way. One run per index (batch splittable for memory), executed
sequentially so concurrent runs never contend for RAM.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from alphagym import storage_io
from alphagym.factor_research_service import FactorResearchService
from alphagym.factor_store import FactorStore
from alphagym.workflows._support import finish, invoke, progress, validate_config

DEFAULT_INDICES = ("000300.SH", "000905.SH", "000852.SH")
DEFAULT_SPLITS = {
    "development": ["2014-01-01", "2020-12-31"],
    "validation": ["2021-01-01", "2023-12-31"],
    "test": ["2024-01-01", "2025-12-31"],
}


def _covered(store: FactorStore, factor_id: str, index_code: str, start: str, end: str) -> bool:
    """Coverage is checked against the run's panel data, not just its config.

    A run whose config claims the window but whose panel stops short (e.g. a
    stale cache or a latest-month gap) is treated as uncovered so the missing
    tail gets computed.
    """
    run = store.latest_succeeded_runs().get(factor_id)
    if run is None:
        return False
    config = json.loads(run["config_json"]) if run.get("config_json") else None
    if not (
        isinstance(config, dict)
        and config.get("index_code") == index_code
        and str(config.get("start_date", "")) <= start
        and str(config.get("end_date", "")) >= end
    ):
        return False
    panel_row = store.connection.execute(
        "SELECT path FROM artifact WHERE run_id=? AND kind='factor_values'",
        (run["run_id"],),
    ).fetchone()
    if panel_row is None or not storage_io.exists(Path(str(panel_row["path"]))):
        return False
    panel_max = pd.to_datetime(
        storage_io.read_frame(Path(str(panel_row["path"])), columns=["signal_date"])["signal_date"]
    ).max()
    # 30 天：允许月末信号日的错位（最新月末 vs 窗口末），但一个月缺口必须补算
    return (pd.Timestamp(end) - panel_max) <= pd.Timedelta(days=30)




@dataclass(frozen=True, slots=True, kw_only=True)
class Config:
    """Explicit workflow inputs; paths never depend on the source checkout."""
    root: Path
    pool: Path
    indices: str = ','.join(DEFAULT_INDICES)
    start_date: str = '2014-01-01'
    end_date: str = '2025-12-31'
    batch_size: int = 20
    monitoring: bool = False
    json: bool = False

    def __post_init__(self):
        validate_config(self, {})


def run(args: Config) -> dict:
    """Execute with Python inputs and return artifact metadata; never exits the host."""

    payload = yaml.safe_load(storage_io.read_text(Path(args.pool), encoding="utf-8"))
    pool = [str(item) for item in payload["pool"]]
    indices = [item.strip() for item in args.indices.split(",") if item.strip()]
    results = []
    for index_code in indices:
        with FactorStore.from_root(Path(args.root)) as store:
            missing = [
                factor_id for factor_id in pool
                if not _covered(store, factor_id, index_code, args.start_date, args.end_date)
            ]
            if not missing:
                progress(f"[backfill] {index_code}: all {len(pool)} factors covered, skip", flush=True)
                continue
            progress(
                f"[backfill] {index_code}: {len(missing)}/{len(pool)} factors missing, "
                f"running in batches of {args.batch_size}",
                flush=True,
            )
            service = FactorResearchService(store)
            for offset in range(0, len(missing), args.batch_size):
                batch = missing[offset:offset + args.batch_size]
                config = {
                    "source": "automatic",
                    "index_code": index_code,
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "point_in_time_audit_passed": True,
                    "splits": DEFAULT_SPLITS,
                    "universe": {
                        "industries": {"include": [], "exclude": []},
                        "symbols": {"include": [], "exclude": []},
                    },
                }
                if args.monitoring:
                    config["monitoring_2026_included"] = True
                run_id = store.create_run(batch, mode="formal", config=config)
                progress(
                    f"[backfill] {index_code} batch {offset // args.batch_size + 1}: "
                    f"run {run_id} factors {len(batch)}",
                    flush=True,
                )
                try:
                    output = service.execute_auto_run(run_id)
                    results.append({"index_code": index_code, "run_id": run_id,
                                    "factors": len(batch), "status": "succeeded"})
                    progress(f"[backfill] {index_code}: OK -> {output}", flush=True)
                except Exception as error:  # noqa: BLE001 - report and continue
                    results.append({"index_code": index_code, "run_id": run_id,
                                    "factors": len(batch), "status": "failed",
                                    "error": str(error)})
                    progress(f"[backfill] {index_code}: FAILED: {error}", flush=True)
    if args.json:
        progress(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return finish(0 if all(item["status"] == "succeeded" for item in results) else 1, results=results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--indices", default=",".join(DEFAULT_INDICES))
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--batch-size", type=int, default=20, help="factors per run (memory)")
    parser.add_argument("--monitoring", action="store_true",
                        help="end date extends into 2026; those rows are label-only")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    return invoke(run, Config, args)


if __name__ == "__main__":
    raise SystemExit(main())
