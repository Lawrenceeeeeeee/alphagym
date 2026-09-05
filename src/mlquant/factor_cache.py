"""Point-in-time factor value cache: space for time.

Every backtest used to rebuild its factor panel from the raw daily table
(13M+ rows for the full A-share history), recomputing identical factor
values again and again. This module persists precomputed factor values so
a backtest only computes factors it has never seen for the current
(data version, universe, factor revision) combination.

The on-disk layout mirrors tushare's ``factor_value`` long table — one row
per (factor, signal_date, symbol) with a single value column — partitioned
per factor inside per-universe directories:

    {root}/factor_library/value_cache/
        manifest.json                              # data signature + coverage
        {universe_key}/
            {factor_id}--{revision[:8]}.parquet    # signal_date, symbol, factor_value
            forward_return.parquet                 # signal_date, symbol, forward_return

Invalidation rules:
  * market data files change        -> the whole cache is stale (the full
                                       history is recomputed from the new files);
  * a factor gets a new revision    -> only that factor is recomputed;
  * coverage is tracked as signal-date intervals; a factor is reused only when
    its stored intervals fully cover the requested window.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mlquant.factor_store import FactorStore

DATA_FILES = ("daily", "adjustments", "fundamentals", "index_members", "industries")


def data_signature(root: Path) -> dict[str, Any]:
    """Version fingerprint of the market data files the cache depends on."""
    signature: dict[str, Any] = {}
    for name in DATA_FILES:
        path = root / "equity" / f"{name}.parquet"
        if not path.is_file():
            continue
        stat = path.stat()
        signature[name] = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    return signature


def _universe_payload(
    index_code: str, universe_config: dict[str, Any] | None,
) -> dict[str, Any]:
    config = universe_config or {}
    industries = config.get("industries") or {}
    symbols = config.get("symbols") or {}
    return {
        "index_code": str(index_code),
        "industries": {
            "include": sorted(industries.get("include", ())),
            "exclude": sorted(industries.get("exclude", ())),
        },
        "symbols": {
            "include": sorted(symbols.get("include", ())),
            "exclude": sorted(symbols.get("exclude", ())),
        },
    }


def universe_key(index_code: str, universe_config: dict[str, Any] | None = None) -> str:
    """Stable cache key for one (index, industry filter, symbol list) universe."""
    payload = _universe_payload(index_code, universe_config)
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode()
    ).hexdigest()
    return digest[:16]


def _merge_intervals(
    existing: list[list[str]] | None, start: object, end: object,
) -> list[list[str]]:
    """Merge signal-coverage intervals, expressed as month periods."""
    low = pd.Timestamp(start).to_period("M")
    high = pd.Timestamp(end).to_period("M")
    points = [
        (pd.Period(a, freq="M"), pd.Period(b, freq="M")) for a, b in (existing or [])
    ]
    points.append((low, high))
    points.sort()
    merged: list[tuple[pd.Period, pd.Period]] = []
    for interval_low, interval_high in points:
        if merged and interval_low <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval_high))
        else:
            merged.append((interval_low, interval_high))
    return [[str(a), str(b)] for a, b in merged]


def _covers(intervals: list[list[str]] | None, start: object, end: object) -> bool:
    """Whether the stored month-period intervals cover every month in [start, end]."""
    low = pd.Timestamp(start).to_period("M")
    high = pd.Timestamp(end).to_period("M")
    months: list[pd.Period] = [low]
    while months[-1] < high:
        months.append(months[-1] + 1)
    stored = sorted(
        (pd.Period(a, freq="M"), pd.Period(b, freq="M"))
        for a, b in (intervals or [])
    )
    position = 0
    for month in months:
        while position < len(stored) and stored[position][1] < month:
            position += 1
        if position >= len(stored) or stored[position][0] > month:
            return False
    return True


class FactorValueCache:
    """Read/store precomputed point-in-time factor values."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.cache_dir = self.root / "factor_library" / "value_cache"
        self.manifest_path = self.cache_dir / "manifest.json"

    # ------------------------------------------------------------ manifest

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"version": 1, "data_signature": {}, "universes": {}}
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"version": 1, "data_signature": {}, "universes": {}}
        if not isinstance(manifest, dict):
            return {"version": 1, "data_signature": {}, "universes": {}}
        return manifest

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temp = self.manifest_path.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, self.manifest_path)

    def signature_matches(self) -> bool:
        return self.load_manifest().get("data_signature") == data_signature(self.root)

    def invalidate_if_stale(self) -> bool:
        """Wipe everything when the underlying data files changed.

        Per the research rules, an incremental price/volume update requires
        recomputing the whole history, so a signature change invalidates the
        entire cache rather than patching individual factors.
        """
        current = data_signature(self.root)
        if self.load_manifest().get("data_signature") == current:
            return False
        if self.cache_dir.is_dir():
            for path in self.cache_dir.iterdir():
                if path.name == "manifest.json":
                    continue
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
        self._write_manifest(
            {"version": 1, "data_signature": current, "universes": {}}
        )
        return True

    # -------------------------------------------------------------- paths

    def _universe_dir(self, key: str) -> Path:
        return self.cache_dir / key

    def _factor_path(self, key: str, factor_id: str, revision_id: str) -> Path:
        return self._universe_dir(key) / f"{factor_id}--{revision_id[:8]}.parquet"

    # -------------------------------------------------------------- reads

    def read_values(
        self, key: str, locked: dict[str, str], start: object, end: object,
    ) -> dict[str, pd.DataFrame]:
        """Cached values for factors whose revision and coverage match."""
        manifest = self.load_manifest()
        if manifest.get("data_signature") != data_signature(self.root):
            return {}
        universe = manifest.get("universes", {}).get(key)
        if not universe:
            return {}
        result: dict[str, pd.DataFrame] = {}
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        for factor_id, revision_id in locked.items():
            entry = (universe.get("factors") or {}).get(factor_id)
            if not entry or entry.get("revision_id") != revision_id:
                continue
            if not _covers(entry.get("intervals"), start, end):
                continue
            path = self._factor_path(key, factor_id, revision_id)
            if not path.is_file():
                continue
            frame = pd.read_parquet(path)
            frame = frame[
                (frame["signal_date"] >= start) & (frame["signal_date"] <= end)
            ]
            result[factor_id] = frame.reset_index(drop=True)
        return result

    def read_forward(
        self, key: str, start: object, end: object,
    ) -> pd.DataFrame | None:
        manifest = self.load_manifest()
        if manifest.get("data_signature") != data_signature(self.root):
            return None
        universe = manifest.get("universes", {}).get(key)
        if not universe:
            return None
        entry = universe.get("forward") or {}
        if not _covers(entry.get("intervals"), start, end):
            return None
        path = self._universe_dir(key) / "forward_return.parquet"
        if not path.is_file():
            return None
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        frame = pd.read_parquet(path)
        return frame[
            (frame["signal_date"] >= start) & (frame["signal_date"] <= end)
        ].reset_index(drop=True)

    # ------------------------------------------------------------- writes

    def store(
        self,
        key: str,
        blocks: dict[str, pd.DataFrame],
        revisions: dict[str, str],
        forward: pd.DataFrame | None,
        *,
        index_code: str,
        universe_config: dict[str, Any] | None,
        mode: str,
        signal_min: object,
        signal_max: object,
    ) -> None:
        """Append freshly computed values into the cache.

        Values are immutable per (factor revision, universe, data version):
        a new revision replaces that factor's file, and a data update wipes
        everything (see ``invalidate_if_stale``).
        """
        self.invalidate_if_stale()
        universe_dir = self._universe_dir(key)
        universe_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC).isoformat()
        manifest = self.load_manifest()
        universe = manifest.setdefault("universes", {}).setdefault(key, {
            **_universe_payload(index_code, universe_config),
            "mode": mode,
            "factors": {},
            "forward": {},
        })
        factors_entry = universe.setdefault("factors", {})
        for factor_id, block in blocks.items():
            if block.empty:
                continue
            revision_id = revisions[factor_id]
            entry = factors_entry.get(factor_id)
            if entry and entry.get("revision_id") != revision_id:
                entry = None
            path = self._factor_path(key, factor_id, revision_id)
            if entry is None and path.is_file():
                path.unlink(missing_ok=True)
            if path.is_file():
                previous = pd.read_parquet(path)
                merged = pd.concat([previous, block], ignore_index=True)
            else:
                merged = block
            merged = (
                merged.drop_duplicates(["signal_date", "symbol"], keep="last")
                .sort_values(["signal_date", "symbol"])
                .reset_index(drop=True)
            )
            temp = path.with_suffix(".tmp.parquet")
            merged.to_parquet(temp, index=False)
            os.replace(temp, path)
            factors_entry[factor_id] = {
                "revision_id": revision_id,
                "intervals": _merge_intervals(
                    (entry or {}).get("intervals"), signal_min, signal_max
                ),
                "computed_at": now,
            }
        if forward is not None and not forward.empty:
            forward_path = universe_dir / "forward_return.parquet"
            if forward_path.is_file():
                previous = pd.read_parquet(forward_path)
                merged = pd.concat([previous, forward], ignore_index=True)
            else:
                merged = forward
            merged = (
                merged.drop_duplicates(["signal_date", "symbol"], keep="last")
                .sort_values(["signal_date", "symbol"])
                .reset_index(drop=True)
            )
            temp = forward_path.with_suffix(".tmp.parquet")
            merged.to_parquet(temp, index=False)
            os.replace(temp, forward_path)
            universe["forward"] = {
                "intervals": _merge_intervals(
                    (universe.get("forward") or {}).get("intervals"),
                    signal_min, signal_max,
                ),
                "computed_at": now,
            }
        self._write_manifest(manifest)


# ---------------------------------------------------------------- builders


def _data_bounds(root: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Full trading range of the imported daily table."""
    import pyarrow.parquet as pq

    path = root / "equity" / "daily.parquet"
    file = pq.ParquetFile(path)
    minimum: object = None
    maximum: object = None
    for group in range(file.metadata.num_row_groups):
        statistics = file.metadata.row_group(group).column(0).statistics
        if statistics is not None and statistics.has_min_max:
            if minimum is None or statistics.min < minimum:
                minimum = statistics.min
            if maximum is None or statistics.max > maximum:
                maximum = statistics.max
    if minimum is not None and maximum is not None:
        return pd.Timestamp(minimum).normalize(), pd.Timestamp(maximum).normalize()
    column = pd.read_parquet(path, columns=["trade_date"])["trade_date"]
    return pd.Timestamp(column.min()).normalize(), pd.Timestamp(column.max()).normalize()


def _build_chunk_worker(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute one signal chunk of one factor batch in a (sub)process."""
    from mlquant.factor_research_service import FactorResearchService

    root = Path(payload["root"])
    with FactorStore.from_root(root) as store:
        service = FactorResearchService(store)
        daily, fundamentals, month_ends, opened, next_month, open_prices = (
            service.prepare_auto_compute(
                payload["locked"], payload["index_code"],
                pd.Timestamp(payload["start"]), pd.Timestamp(payload["end"]),
            )
        )
        signal_min = pd.Timestamp(payload["signal_min"])
        signal_max = pd.Timestamp(payload["signal_max"])
        signals = [date for date in month_ends if signal_min <= date <= signal_max]
        blocks, forward = service._compute_block(
            signals, opened, next_month, open_prices, daily, fundamentals,
            payload["factor_ids"], payload["locked"], payload["index_code"],
            payload.get("universe") or {}, bool(payload["formal"]),
        )
    out_dir = Path(payload["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for factor_id, block in blocks.items():
        path = out_dir / f"{factor_id}.parquet"
        block.to_parquet(path, index=False)
        results.append({"factor_id": factor_id, "path": str(path),
                        "rows": len(block)})
    if forward is not None and not forward.empty:
        path = out_dir / "forward_return.parquet"
        forward.to_parquet(path, index=False)
        results.append({"factor_id": None, "path": str(path),
                        "rows": len(forward)})
    return results


def _run_payloads(
    payloads: list[dict[str, Any]], workers: int,
) -> list[list[dict[str, Any]]]:
    # Windows spawn re-imports __main__ from its file; a stdin/heredoc main
    # module crashes every worker and leaves pool.map waiting forever, so
    # fall back to in-process computation when there is no real main file.
    main_file = getattr(__import__("__main__"), "__file__", "") or ""
    spawn_safe = Path(main_file).is_file()
    if workers <= 1 or len(payloads) <= 1 or not spawn_safe:
        return [_build_chunk_worker(payload) for payload in payloads]
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers) as pool:
        return pool.map(_build_chunk_worker, payloads)


def _seed_from_run_panels(
    root: Path,
    cache: FactorValueCache,
    key: str,
    locked: dict[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    mode: str,
    index_code: str,
    universe_config: dict[str, Any] | None,
) -> list[str]:
    """Import factor columns from recent run panels built on current data.

    A finished backtest panel already contains exactly the values the cache
    stores, so importing it saves recomputation. Panels older than the newest
    data file, from a different universe, or with different revisions are
    skipped.
    """
    signature_mtimes = max(
        (int(item["mtime_ns"]) for item in data_signature(root).values()), default=0
    )
    seeded: list[str] = []
    with FactorStore.from_root(root) as store:
        rows = store.connection.execute(
            """SELECT a.path, r.run_id, r.created_at, r.mode, r.config_json
            FROM artifact a JOIN research_run r ON r.run_id=a.run_id
            WHERE a.kind='factor_values'"""
        ).fetchall()
        for row in rows:
            try:
                created = pd.Timestamp(row["created_at"]).value
                config = json.loads(row["config_json"])
            except (TypeError, ValueError):
                continue
            if created < signature_mtimes:
                continue
            if str(row["mode"]) != mode or str(config.get("index_code")) != index_code:
                continue
            if universe_key(index_code, config.get("universe") or {}) != key:
                continue
            revisions = dict(store.connection.execute(
                "SELECT factor_id, revision_id FROM run_factor WHERE run_id=?",
                (row["run_id"],),
            ).fetchall())
            revisions = {str(k): str(v) for k, v in revisions.items()}
            targets = {
                factor_id for factor_id, revision in revisions.items()
                if locked.get(factor_id) == revision
            }
            if not targets:
                continue
            path = Path(row["path"])
            if not path.is_file():
                continue
            panel = pd.read_parquet(
                path, columns=["signal_date", "symbol", "factor_name", "raw_value"]
            )
            panel = panel[panel["factor_name"].isin(targets)]
            if panel.empty:
                continue
            blocks: dict[str, pd.DataFrame] = {}
            for factor_id, block in panel.groupby("factor_name", observed=True):
                blocks[str(factor_id)] = (
                    block[["signal_date", "symbol", "raw_value"]]
                    .rename(columns={"raw_value": "factor_value"})
                    .reset_index(drop=True)
                )
            cache.store(
                key, blocks,
                {factor_id: revisions[factor_id] for factor_id in blocks},
                None,
                index_code=index_code, universe_config=universe_config,
                mode=mode, signal_min=start, signal_max=end,
            )
            seeded.extend(sorted(blocks))
    return sorted(set(seeded))


def build_factor_cache(
    root: str | Path,
    factor_ids: list[str],
    *,
    index_code: str = "ALL_A",
    universe_config: dict[str, Any] | None = None,
    mode: str = "formal",
    workers: int = 4,
    batch_size: int = 25,
    seed_from_panels: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Ensure cached factor values over the full data range for ``factor_ids``."""
    root = Path(root).expanduser().resolve()
    cache = FactorValueCache(root)
    cache.invalidate_if_stale()
    with FactorStore.from_root(root) as store:
        by_id = {item["factor_id"]: item for item in store.list_factors()}
        unknown = [factor_id for factor_id in factor_ids if factor_id not in by_id]
        if unknown:
            raise ValueError(f"未知因子：{unknown}")
        locked = {factor_id: by_id[factor_id]["current_revision_id"]
                  for factor_id in factor_ids}
    start, end = _data_bounds(root)
    key = universe_key(index_code, universe_config)
    with FactorStore.from_root(root) as store:
        cached = cache.read_values(key, locked, start, end)
        forward = cache.read_forward(key, start, end)
    summary: dict[str, Any] = {
        "root": str(root), "universe_key": key, "index_code": index_code,
        "mode": mode, "start": str(start.date()), "end": str(end.date()),
        "computed": [], "reused": len(cached), "seeded": [], "errors": {},
    }
    missing = [factor_id for factor_id in factor_ids if factor_id not in cached]
    if seed_from_panels and missing:
        seeded = _seed_from_run_panels(
            root, cache, key, locked, start, end, mode=mode,
            index_code=index_code, universe_config=universe_config,
        )
        missing = [factor_id for factor_id in missing if factor_id not in set(seeded)]
        summary["seeded"] = seeded

    total = len(missing) + (1 if forward is None else 0)
    done = 0

    month_ends = _month_ends(root, start, end)
    chunks = _signal_chunks(month_ends, workers)
    if not month_ends:
        return summary

    if forward is None:
        with tempfile.TemporaryDirectory() as temporary:
            payloads = [
                {
                    "root": str(root),
                    "factor_ids": [],
                    "locked": {},
                    "index_code": index_code,
                    "universe": universe_config or {},
                    "formal": mode == "formal",
                    "start": str(start.date()), "end": str(end.date()),
                    "signal_min": str(chunk[0].date()), "signal_max": str(chunk[-1].date()),
                    "out_dir": str(Path(temporary) / f"chunk-{position}"),
                }
                for position, chunk in enumerate(chunks)
            ]
            results = _run_payloads(payloads, workers)
            forwards = []
            for chunk_result in results:
                for item in chunk_result:
                    if item["factor_id"] is None:
                        forwards.append(pd.read_parquet(item["path"]))
            if forwards:
                cache.store(
                    key, {}, {}, pd.concat(forwards, ignore_index=True),
                    index_code=index_code, universe_config=universe_config,
                    mode=mode, signal_min=month_ends[0], signal_max=month_ends[-1],
                )
        done += 1
        if progress:
            progress(done, total, "forward returns")

    for position in range(0, len(missing), batch_size):
        batch = missing[position:position + batch_size]
        batch_locked = {factor_id: locked[factor_id] for factor_id in batch}
        with tempfile.TemporaryDirectory() as temporary:
            payloads = [
                {
                    "root": str(root),
                    "factor_ids": batch,
                    "locked": batch_locked,
                    "index_code": index_code,
                    "universe": universe_config or {},
                    "formal": mode == "formal",
                    "start": str(start.date()), "end": str(end.date()),
                    "signal_min": str(chunk[0].date()), "signal_max": str(chunk[-1].date()),
                    "out_dir": str(Path(temporary) / f"chunk-{position}"),
                }
                for position, chunk in enumerate(chunks)
            ]
            results = _run_payloads(payloads, workers)
            blocks: dict[str, pd.DataFrame] = {}
            forwards = []
            for chunk_result in results:
                for item in chunk_result:
                    if item["factor_id"] is None:
                        forwards.append(pd.read_parquet(item["path"]))
                        continue
                    frame = pd.read_parquet(item["path"])
                    blocks[item["factor_id"]] = (
                        frame if item["factor_id"] not in blocks
                        else pd.concat([blocks[item["factor_id"]], frame],
                                       ignore_index=True)
                    )
            cache.store(
                key, blocks, batch_locked,
                pd.concat(forwards, ignore_index=True) if forwards else None,
                index_code=index_code, universe_config=universe_config,
                mode=mode, signal_min=month_ends[0], signal_max=month_ends[-1],
            )
            summary["computed"].extend(batch)
        done += 1
        if progress:
            progress(done, total, f"batch {batch[0]}..{batch[-1]}")
    return summary


def _month_ends(
    root: Path, start: pd.Timestamp, end: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Month-end signal dates over [start, end], from the daily table."""
    column = pd.read_parquet(
        root / "equity" / "daily.parquet", columns=["trade_date"]
    )["trade_date"]
    opened = pd.Series(pd.to_datetime(column.unique())).sort_values()
    ends = opened.groupby(opened.dt.to_period("M")).max().tolist()
    return [date for date in ends if start <= date <= end]


def _signal_chunks(
    month_ends: list[pd.Timestamp], workers: int,
) -> list[list[pd.Timestamp]]:
    workers = max(1, workers)
    size = max(1, math.ceil(len(month_ends) / workers))
    return [
        month_ends[start:start + size]
        for start in range(0, len(month_ends), size)
    ]
