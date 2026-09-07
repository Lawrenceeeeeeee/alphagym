"""Unit and integration tests for the precomputed factor-value cache."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from alphagym import storage_io
from alphagym.factor_cache import (
    FactorValueCache,
    _covers,
    _merge_intervals,
    data_signature,
    universe_key,
)
from alphagym.factor_research_service import FactorResearchService
from alphagym.factor_store import FactorStore
from alphagym.factors.base import FactorDefinition


def _definition(factor_id: str, formula: str) -> FactorDefinition:
    return FactorDefinition(
        factor_id=factor_id, name=factor_id, formula=formula,
        hypothesis_id=f"hypothesis_{factor_id.lower()}", family="momentum",
    )


def _write_market_data(root) -> None:
    dates = pd.bdate_range("2022-12-01", "2024-06-30")
    symbols = [f"{index:06d}.SZ" for index in range(1, 11)]
    rows = []
    for day_number, date in enumerate(dates):
        for symbol_number, symbol in enumerate(symbols, start=1):
            noise = np.sin(day_number / 3.0 + symbol_number) * 0.04 * symbol_number
            close = 10 + symbol_number + day_number * (0.005 + symbol_number / 10000) + noise
            rows.append({
                "trade_date": date,
                "symbol": symbol,
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000 + symbol_number,
                "amount": close * (1_000_000 + symbol_number),
            })
    equity = root / "equity"
    equity.mkdir(parents=True)
    storage_io.write_frame(pd.DataFrame(rows), equity / "daily.parquet", index=False)
    storage_io.write_frame(pd.DataFrame({
        "trade_date": [dates[0]] * len(symbols),
        "symbol": symbols,
        "adjust_factor": [1.0] * len(symbols),
    }), equity / "adjustments.parquet", index=False)


def _run_config(start: str, end: str) -> dict:
    return {
        "source": "automatic",
        "index_code": "ALL_A",
        "start_date": start,
        "end_date": end,
        "point_in_time_audit_passed": False,
    }


# ------------------------------------------------------------- storage units


def test_interval_merge_and_coverage() -> None:
    merged = _merge_intervals(None, "2023-01-01", "2023-03-31")
    assert merged == [["2023-01", "2023-03"]]
    merged = _merge_intervals(merged, "2023-06-01", "2023-09-30")
    assert merged == [["2023-01", "2023-03"], ["2023-06", "2023-09"]]
    merged = _merge_intervals(merged, "2023-04-01", "2023-05-31")
    assert merged == [["2023-01", "2023-09"]]
    assert _covers(merged, "2023-02-15", "2023-08-01")
    assert not _covers(merged, "2023-01-01", "2023-11-30")
    assert _covers([["2023-01", "2023-01"], ["2023-02", "2023-02"]], "2023-01-01", "2023-02-28")
    assert not _covers([["2023-01", "2023-01"], ["2023-03", "2023-03"]], "2023-01-01", "2023-03-31")


def test_universe_key_is_deterministic_and_order_free() -> None:
    first = universe_key("ALL_A", {"industries": {"include": ["801010", "801020"]}})
    second = universe_key("ALL_A", {"industries": {"include": ["801020", "801010"]}})
    assert first == second
    assert first != universe_key("000300.SH")


def test_store_read_and_revision_invalidation(tmp_path) -> None:
    _write_market_data(tmp_path)
    with FactorStore.from_root(tmp_path) as store:
        revision = store.save_definition(_definition("AUTO", "=RETURN(market.adj_close, 5)"))
    cache = FactorValueCache(tmp_path)
    key = universe_key("ALL_A")
    start, end = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")
    block = pd.DataFrame({
        "signal_date": pd.to_datetime(["2023-01-31", "2023-02-28"]),
        "symbol": ["000001.SZ", "000001.SZ"],
        "factor_value": [0.01, 0.02],
    })
    forward = pd.DataFrame({
        "signal_date": pd.to_datetime(["2023-01-31"]),
        "symbol": ["000001.SZ"],
        "forward_return": [0.03],
    })
    cache.store(
        key, {"AUTO": block}, {"AUTO": revision}, forward,
        index_code="ALL_A", universe_config=None, mode="smoke",
        signal_min=start, signal_max=end,
    )
    values = cache.read_values(key, {"AUTO": revision}, start, end)
    assert list(values) == ["AUTO"]
    assert len(values["AUTO"]) == 2
    cached_forward = cache.read_forward(key, start, end)
    assert cached_forward is not None and len(cached_forward) == 1
    # A different revision must not be served.
    with FactorStore.from_root(tmp_path) as store:
        new_revision = store.save_definition(_definition("AUTO", "=RETURN(market.adj_close, 10)"))
    assert cache.read_values(key, {"AUTO": new_revision}, start, end) == {}
    # Stale data signature invalidates everything.
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("AUTO", "=RETURN(market.adj_close, 5)"))
    after = cache.read_values(key, {"AUTO": revision}, start, end)
    assert list(after) == ["AUTO"]
    assert after["AUTO"].equals(values["AUTO"])
    cache.invalidate_if_stale()
    # Publishing a new database version invalidates the cache; touching a local
    # filename has no bearing on the authoritative dataset.
    path = tmp_path / "equity" / "adjustments.parquet"
    storage_io.write_frame(storage_io.read_frame(path), path, index=False)
    assert cache.signature_matches() is False
    assert cache.invalidate_if_stale() is True
    assert cache.read_values(key, {"AUTO": revision}, start, end) == {}
    manifest = json.loads(storage_io.read_text(cache.manifest_path, encoding="utf-8"))
    assert manifest["data_signature"] == data_signature(tmp_path)


# ------------------------------------------------------------ run integration


def test_auto_run_warms_cache_and_second_run_reuses_it(tmp_path, monkeypatch) -> None:
    _write_market_data(tmp_path)
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("AUTO", "=RETURN(market.adj_close, 5)"))
    with FactorStore.from_root(tmp_path) as store:
        service = FactorResearchService(store)
        first_run = store.create_run(["AUTO"], mode="smoke",
                                     config=_run_config("2023-01-01", "2024-03-31"))
        service.execute_auto_run(first_run)
    cache = FactorValueCache(tmp_path)
    assert cache.signature_matches()
    key = universe_key("ALL_A")
    with FactorStore.from_root(tmp_path) as store:
        locked = {item["factor_id"]: item["revision_id"]
                  for item in store.run_detail(first_run)["factors"]}
    cached = cache.read_values(key, locked, pd.Timestamp("2023-01-01"), pd.Timestamp("2024-03-31"))
    assert list(cached) == ["AUTO"]
    assert not cached["AUTO"].empty

    def no_compute(*_args, **_kwargs):
        raise AssertionError("cache hit must not recompute factor values")

    monkeypatch.setattr(FactorResearchService, "_compute_block", no_compute)
    with FactorStore.from_root(tmp_path) as store:
        service = FactorResearchService(store)
        second_run = store.create_run(["AUTO"], mode="smoke",
                                      config=_run_config("2023-01-01", "2024-03-31"))
        service.execute_auto_run(second_run)
        assert store.run_detail(second_run)["status"] == "succeeded"


def _panel_frame(store: FactorStore, run_id: str) -> pd.DataFrame:
    row = store.connection.execute(
        "SELECT path FROM artifact WHERE run_id=? AND kind='factor_values'",
        (run_id,),
    ).fetchone()
    assert row is not None
    return storage_io.read_frame(row["path"])


def test_panel_identity_between_cached_and_fresh_paths(tmp_path) -> None:
    """A cache-assembled panel must equal the freshly computed panel."""
    _write_market_data(tmp_path)
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("AUTO", "=RETURN(market.adj_close, 5)"))
    with FactorStore.from_root(tmp_path) as store:
        service = FactorResearchService(store)
        first_run = store.create_run(["AUTO"], mode="smoke",
                                     config=_run_config("2023-01-01", "2024-03-31"))
        service.execute_auto_run(first_run)
        fresh = _panel_frame(store, first_run)
    with FactorStore.from_root(tmp_path) as store:
        service = FactorResearchService(store)
        second_run = store.create_run(["AUTO"], mode="smoke",
                                      config=_run_config("2023-01-01", "2024-03-31"))
        service.execute_auto_run(second_run)
        cached = _panel_frame(store, second_run)
    assert fresh.equals(cached)


def test_new_revision_triggers_recompute_only_for_that_factor(tmp_path) -> None:
    _write_market_data(tmp_path)
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("AUTO", "=RETURN(market.adj_close, 5)"))
        store.save_definition(_definition("OTHER", "=RETURN(market.adj_close, 3)"))
    with FactorStore.from_root(tmp_path) as store:
        service = FactorResearchService(store)
        run = store.create_run(["AUTO", "OTHER"], mode="smoke",
                               config=_run_config("2023-01-01", "2024-03-31"))
        service.execute_auto_run(run)
    cache = FactorValueCache(tmp_path)
    key = universe_key("ALL_A")
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("AUTO", "=RETURN(market.adj_close, 8)"))
        locked = {item["factor_id"]: item["revision_id"]
                  for item in store.run_detail(run)["factors"]}
        current = {item["factor_id"]: item["current_revision_id"]
                   for item in store.list_factors()}
    cached = cache.read_values(key, current, pd.Timestamp("2023-01-01"), pd.Timestamp("2024-03-31"))
    # OTHER still matches its cached revision; AUTO (new revision) does not.
    assert list(cached) == ["OTHER"]
    assert locked["OTHER"] == current["OTHER"]
    assert locked["AUTO"] != current["AUTO"]


# ------------------------------------------------------------ build command


def test_build_factor_cache_computes_missing_only(tmp_path) -> None:
    _write_market_data(tmp_path)
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("AUTO", "=RETURN(market.adj_close, 5)"))
        store.save_definition(_definition("OTHER", "=RETURN(market.adj_close, 3)"))
    from alphagym.factor_cache import build_factor_cache

    summary = build_factor_cache(tmp_path, ["AUTO", "OTHER"], workers=1)
    assert sorted(summary["computed"]) == ["AUTO", "OTHER"]
    assert summary["reused"] == 0
    cache = FactorValueCache(tmp_path)
    key = universe_key("ALL_A")
    with FactorStore.from_root(tmp_path) as store:
        locked = {item["factor_id"]: item["current_revision_id"]
                  for item in store.list_factors()}
    cached = cache.read_values(key, locked, pd.Timestamp("2022-12-01"), pd.Timestamp("2024-06-30"))
    assert set(cached) == {"AUTO", "OTHER"}
    # A second build over the same inputs is a no-op.
    summary = build_factor_cache(tmp_path, ["AUTO", "OTHER"], workers=1)
    assert summary["computed"] == []
    assert summary["reused"] == 2


def test_build_factor_cache_rejects_unknown_ids(tmp_path) -> None:
    from alphagym.factor_cache import build_factor_cache

    with pytest.raises(ValueError, match="未知因子"):
        build_factor_cache(tmp_path, ["GHOST"], workers=1)
