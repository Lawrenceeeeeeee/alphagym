"""Integration checks against a disposable real ClickHouse instance."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from alphagym.storage import ClickHouseStore
from alphagym.tushare_import import TushareImporter


def test_versions_upsert_and_aborted_batch(tmp_path):
    store = ClickHouseStore(tmp_path, initialize=True)
    initial = pd.DataFrame({"symbol": ["A", "B"], "value": [1., 2.]})
    first = store.write_frame("bars", initial, keys=["symbol"])
    store.write_frame("bars", pd.DataFrame({"symbol": ["A"], "value": [3.]}),
                      mode="upsert", keys=["symbol"])
    assert store.read_frame("bars").set_index("symbol")["value"].to_dict() == {"A": 3., "B": 2.}
    pd.testing.assert_frame_equal(store.read_frame("bars", as_of=first["version"]), initial)
    with pytest.raises(RuntimeError), store.batch() as batch:
        batch.frame("bars", pd.DataFrame({"symbol": ["A"], "value": [99.]}), keys=["symbol"])
        batch.blob("checkpoint", b"incomplete")
        raise RuntimeError("network interrupted")
    assert store.manifest("checkpoint") is None
    assert store.read_frame("bars").set_index("symbol").loc["A", "value"] == 3.
    assert not list(tmp_path.rglob("*.parquet"))


def test_index_nulls_and_projection_filters(tmp_path):
    store = ClickHouseStore(tmp_path, initialize=True)
    frame = pd.DataFrame({"value": [1., float("nan")], "date": pd.to_datetime(
        ["2024-01-01", "2024-01-02"]
    ).astype("datetime64[ns]")},
                         index=pd.Index(["a", "b"], name="factor"))
    store.write_frame("matrix", frame, index=True)
    pd.testing.assert_frame_equal(store.read_frame("matrix"), frame)
    result = store.read_frame("matrix", columns=["value"], filters=[("date", "<=", pd.Timestamp("2024-01-01"))])
    assert result.index.tolist() == ["a"]
    assert result.columns.tolist() == ["value"]


class FakeTushare:
    def trade_cal(self, **kwargs):
        return pd.DataFrame({"cal_date": ["20240102"], "is_open": [1]})

    def daily(self, **kwargs):
        return pd.DataFrame({"ts_code": ["600000.SH"], "trade_date": ["20240102"],
                             "open": [10.], "high": [11.], "low": [9.], "close": [10.],
                             "vol": [100.], "amount": [100.]})

    def adj_factor(self, **kwargs):
        return pd.DataFrame({"ts_code": ["600000.SH"], "trade_date": ["20240102"], "adj_factor": [1.]})

    def daily_basic(self, **kwargs):
        return pd.DataFrame({"ts_code": ["600000.SH"], "turnover_rate": [1.], "circ_mv": [100.]})


def test_tushare_idempotency_units_and_checkpoint(tmp_path):
    importer = TushareImporter(FakeTushare(), tmp_path, start="2024-01-01", end="2024-01-03", rate_limit=0)
    importer.run_market()
    importer.run_market()
    frame = importer.store.read_frame("equity/daily.parquet")
    assert len(frame) == 1
    assert frame.iloc[0]["volume"] == 10000
    assert frame.iloc[0]["amount"] == 100000
    assert frame.iloc[0]["turnover"] == .01
    assert frame.iloc[0]["float_market_cap"] == 1000000
    checkpoint = json.loads(importer.store.read_blob("sync/tushare/market/checkpoint.json"))
    assert checkpoint["completed_through"] == "2024-01-02"


def test_tushare_partial_response_does_not_publish(tmp_path):
    pro = FakeTushare()
    pro.adj_factor = lambda **kwargs: pd.DataFrame()
    importer = TushareImporter(pro, tmp_path, start="2024-01-01", end="2024-01-03", rate_limit=0)
    with pytest.raises(ValueError, match="Incomplete"):
        importer.run_market()
    assert importer.store.manifest("equity/daily.parquet") is None
    assert importer.store.manifest("sync/tushare/market/checkpoint.json") is None


def test_financial_releases_never_join_future_values():
    indicator = pd.DataFrame({"ts_code": ["600000.SH"], "end_date": ["20231231"],
                              "ann_date": ["20240301"], "eps": [1.]})
    income = pd.DataFrame({"ts_code": ["600000.SH"], "end_date": ["20231231"],
                           "ann_date": ["20240401"], "revenue": [100.]})
    result = TushareImporter._merge_fundamental(indicator, income)
    assert len(result) == 2
    assert pd.isna(result.iloc[0]["revenue"])
    assert result.iloc[1]["revenue"] == 100.
    assert result.iloc[1]["eps"] == 1.


def test_schema_evolution_preserves_older_rows(tmp_path):
    store = ClickHouseStore(tmp_path, initialize=True)
    first = store.write_frame("bars", pd.DataFrame({"symbol": ["A", "B"], "close": [1., 2.]}), keys=["symbol"])
    store.write_frame("bars", pd.DataFrame({"symbol": ["A"], "close": [3.], "volume": [100.]}),
                      mode="upsert", keys=["symbol"])
    frame = store.read_frame("bars").set_index("symbol")
    assert frame.loc["A", "volume"] == 100.
    assert frame.loc["B", "close"] == 2.
    assert pd.isna(frame.loc["B", "volume"])
    assert "volume" not in store.read_frame("bars", as_of=first["version"])


def test_late_publication_cannot_enter_an_existing_snapshot(tmp_path):
    from alphagym.storage import WriteBatch

    store = ClickHouseStore(tmp_path, initialize=True)
    pending = WriteBatch(store)
    pending.frame("bars", pd.DataFrame({"value": [99.]}))
    first = store.write_frame("bars", pd.DataFrame({"value": [1.]}))
    pending.commit()
    assert store.read_frame("bars").iloc[0]["value"] == 99.
    assert store.read_frame("bars", as_of=first["version"]).iloc[0]["value"] == 1.


def test_parquet_workspace_migration_is_resumable_and_preserves_source(tmp_path):
    from alphagym.migration import migrate_workspace

    source, target = tmp_path / "old", tmp_path / "new"
    (source / "equity").mkdir(parents=True)
    frame = pd.DataFrame({"symbol": ["600000.SH"], "trade_date": pd.to_datetime(["2024-01-02"]),
                          "open": [10.], "high": [11.], "low": [9.], "close": [10.],
                          "volume": [100.], "amount": [1000.]})
    path = source / "equity" / "daily.parquet"
    frame.to_parquet(path, index=False)
    before = path.read_bytes()
    assert migrate_workspace(source, target)["files_imported"] == 1
    assert migrate_workspace(source, target)["files_skipped"] == 1
    assert path.read_bytes() == before
    assert not (target / "equity" / "daily.parquet").is_file()
    stored = ClickHouseStore(target).read_frame("equity/daily.parquet")
    frame["trade_date"] = frame["trade_date"].astype("datetime64[ns]")
    pd.testing.assert_frame_equal(stored[frame.columns], frame)


def test_workspace_migration_normalizes_legacy_tushare_units(tmp_path):
    from alphagym.migration import migrate_workspace

    source, target = tmp_path / "old", tmp_path / "new"
    (source / "equity").mkdir(parents=True)
    frame = pd.DataFrame({"symbol": ["600000.SH"], "trade_date": pd.to_datetime(["2024-01-02"]),
                          "open": [10.], "high": [11.], "low": [9.], "close": [10.],
                          "volume": [100.], "amount": [200.], "turnover": [1.5],
                          "float_market_cap": [300.]})
    frame.to_parquet(source / "equity/daily.parquet", index=False)
    metadata = {"units": {"volume": "手", "amount": "千元", "turnover": "%",
                           "float_market_cap": "万元"}}
    (source / "equity/metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    migrate_workspace(source, target)
    store = ClickHouseStore(target)
    row = store.read_frame("equity/daily.parquet").iloc[0]
    assert row[["volume", "amount", "turnover", "float_market_cap"]].tolist() == [10000., 200000., .015, 3000000.]
    migrated = json.loads(store.read_blob("equity/metadata.json"))
    assert migrated["units"] == {"volume": "股", "amount": "元", "turnover": "ratio",
                                  "float_market_cap": "元"}


def test_zero_column_parquet_is_preserved_as_an_empty_frame(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from alphagym.migration import migrate_workspace

    source, target = tmp_path / "old", tmp_path / "new"
    path = source / "experiments" / "rejected.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({}), path)
    migrate_workspace(source, target)
    store = ClickHouseStore(target)
    assert store.read_frame("experiments/rejected.parquet").empty
    assert store.count_frame("experiments/rejected.parquet") == 0
    assert list(store.iter_batches("experiments/rejected.parquet")) == []


def test_nested_interchange_fields_are_preserved_as_json(tmp_path):
    import numpy as np

    store = ClickHouseStore(tmp_path, initialize=True)
    store.write_frame("instrument_snapshot", pd.DataFrame({
        "instrument": ["BTC-USDT-SWAP"],
        "quotes": [np.array(["USDT", "USD"], dtype=object)],
        "changes": [[{"field": "tick", "value": "0.1"}]],
    }))
    row = store.read_frame("instrument_snapshot").iloc[0]
    assert json.loads(row["quotes"]) == ["USDT", "USD"]
    assert json.loads(row["changes"])[0]["field"] == "tick"


def test_financial_pagination_uses_provider_limit(tmp_path):
    class Paged:
        def fina_indicator(self, **kwargs):
            assert kwargs["limit"] == 100
            return pd.DataFrame({"id": range(kwargs["offset"], min(kwargs["offset"] + 100, 205))})

    importer = TushareImporter(Paged(), tmp_path, rate_limit=0)
    assert len(importer._pages("fina_indicator")) == 205


def test_streaming_counts_and_pinned_market_version(tmp_path):
    import pyarrow as pa

    from alphagym import storage_io

    store = storage_io.store_for(tmp_path, initialize=True)
    store.write_frame("equity/custom", pd.DataFrame({"value": [1., 2.]}))
    store.write_blob("equity/metadata.json", b"old metadata")
    with storage_io.snapshot(tmp_path) as version:
        store.write_frame("equity/custom", pd.DataFrame({"value": [3.]}))
        store.write_blob("equity/metadata.json", b"new metadata")
        assert storage_io.current_version(tmp_path) == version
        assert len(storage_io.read_frame(tmp_path / "equity/custom")) == 2
        assert storage_io.read_bytes(tmp_path / "equity/metadata.json") == b"old metadata"
    store.write_blob("reports/example", b"report")
    assert storage_io.current_version(tmp_path) < store.watermark()
    assert store.count_frame("equity/custom") == 1
    result = pa.Table.from_batches(list(store.iter_batches("equity/custom", batch_size=1)))
    assert result.to_pandas()["value"].tolist() == [3.]


def test_catalog_migration_and_workspace_relocation(tmp_path):
    import sqlite3

    from alphagym import storage_io
    from alphagym.factor_store import FactorStore
    from alphagym.factors.library import seed_definitions
    from alphagym.migration import migrate_workspace

    source, target, moved = (tmp_path / name for name in ("old", "new", "moved"))
    (source / "factor_library").mkdir(parents=True)
    with FactorStore() as legacy:
        legacy.bootstrap(seed_definitions()[:1])
        run_id = legacy.create_run([], mode="smoke", config={"source_path": str(source / "equity/daily.parquet")})
        with sqlite3.connect(source / "factor_library/catalog.sqlite") as disk:
            legacy.connection.backup(disk)
    result = migrate_workspace(source, target)
    assert result["catalog_rows"] > 0
    # Emulate losing the separate migration resource receipt after the catalog
    # commit; its transactional catalog receipt still makes a retry safe.
    storage_io.unlink(target / "migration/catalog.json")
    assert migrate_workspace(source, target)["catalog_rows"] == result["catalog_rows"]
    moved.mkdir()
    (moved / "storage.yaml").write_bytes((target / "storage.yaml").read_bytes())
    with FactorStore.from_root(moved) as catalog:
        assert len(catalog.list_factors()) == 1
        assert catalog.run_detail(run_id)["config"]["source_path"] == str(moved / "equity/daily.parquet")
    assert not list(target.rglob("*.sqlite"))
