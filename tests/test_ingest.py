from __future__ import annotations

import pandas as pd
import pytest

from mlquant import DataContractError
from mlquant.ingest import import_qmt
from mlquant.storage_io import store_for


def test_empty_import_does_not_replace_existing_pair(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    equity = tmp_path / "data" / "equity"
    equity.mkdir(parents=True)
    (equity / "daily.parquet").write_bytes(b"existing daily")
    (equity / "adjustments.parquet").write_bytes(b"existing adjustments")
    monkeypatch.setattr("mlquant.ingest.QmtDividendAdapter.read", lambda self: pd.DataFrame({
        "trade_date": [pd.Timestamp("2020-01-01")], "symbol": ["600000.SH"], "adjust_factor": [1.0],
    }))
    with pytest.raises(DataContractError, match="no valid daily"):
        import_qmt(source, tmp_path / "data")
    assert (equity / "daily.parquet").read_bytes() == b"existing daily"
    assert (equity / "adjustments.parquet").read_bytes() == b"existing adjustments"
    assert not list(equity.glob("*.tmp"))


def test_streaming_ingest_cleans_up_after_batch_failure(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr("mlquant.ingest.QmtDividendAdapter.read", lambda self: pd.DataFrame({
        "trade_date": [pd.Timestamp("2020-01-01")], "symbol": ["600000.SH"], "adjust_factor": [1.0],
    }))
    monkeypatch.setattr("mlquant.ingest.QmtDailyAdapter.symbols", lambda self: ["600000.SH", "600001.SH"])

    def read(self, symbols):
        if symbols == ["600001.SH"]:
            raise OSError("synthetic batch read failure")
        return pd.DataFrame({"trade_date": [pd.Timestamp("2020-01-01")], "symbol": symbols,
                             "open": [10.], "high": [11.], "low": [9.], "close": [10.],
                             "volume": [100.], "amount": [1000.]})

    monkeypatch.setattr("mlquant.ingest.QmtDailyAdapter.read", read)
    with pytest.raises(OSError, match="batch read failure"):
        import_qmt(source, tmp_path / "data", batch_size=1)
    store = store_for(tmp_path / "data")
    assert store.manifest("equity/daily.parquet") is None
    assert store.manifest("equity/adjustments.parquet") is None
