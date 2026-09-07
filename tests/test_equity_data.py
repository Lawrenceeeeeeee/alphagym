from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphagym import storage_io
from alphagym.equity_data import (
    _QMT_DAILY_DTYPE,
    DataContractError,
    EquityDataBundle,
    read_qmt_daily_dat,
)


def test_bundle_audit_and_roundtrip(bundle: EquityDataBundle, tmp_path) -> None:
    assert bundle.audit(formal=False, index_code="000300.SH").ok
    manifest = bundle.build_snapshot(tmp_path / "equity", formal=False)
    assert storage_io.exists(manifest)


def test_formal_rejects_current_industry_snapshot(bundle: EquityDataBundle) -> None:
    bundle.metadata["industry_snapshot_only"] = True
    assert not bundle.audit(formal=True).ok
    assert bundle.audit(formal=False).ok
    assert "NON-FORMAL" in bundle.audit(formal=False).warnings[0]


def test_formal_rejects_equal_imputed_benchmark_weights(bundle: EquityDataBundle) -> None:
    bundle.metadata["benchmark_weights_imputed_equal"] = True
    formal = bundle.audit(formal=True)
    smoke = bundle.audit(formal=False)
    assert any("equal-imputed" in error for error in formal.errors)
    assert any("equal-imputed" in warning for warning in smoke.warnings)


def test_missing_index_is_blocked_independently(bundle: EquityDataBundle) -> None:
    result = bundle.audit(formal=False, index_code="000852.SH")
    assert not result.ok
    assert any("000852.SH" in error for error in result.errors)


def test_financial_point_in_time_ignores_future_announcement(bundle: EquityDataBundle) -> None:
    date = pd.Timestamp("2024-05-01")
    before = bundle.point_in_time("fundamentals", date)
    future = bundle.fundamentals.iloc[[0]].copy()
    future["stat_date"] = pd.Timestamp("2025-03-31")
    future["available_date"] = pd.Timestamp("2025-04-30")
    future["roe"] = 999
    bundle.fundamentals = pd.concat([bundle.fundamentals, future], ignore_index=True)
    after = bundle.point_in_time("fundamentals", date)
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))


def test_overlapping_industry_intervals_fail(bundle: EquityDataBundle) -> None:
    duplicate = bundle.industries.iloc[[0]].copy()
    duplicate["valid_from"] = pd.Timestamp("2020-01-01")
    bundle.industries = pd.concat([bundle.industries, duplicate], ignore_index=True)
    assert any("overlapping" in error for error in bundle.audit().errors)


def test_formal_gate_requires_full_2014_2025_history(bundle: EquityDataBundle) -> None:
    result = bundle.audit(formal=True, index_code="000300.SH")
    assert any("incomplete 2014-2025" in error for error in result.errors)


def test_qmt_dat_parser_units(tmp_path) -> None:
    path = tmp_path / "000001.DAT"
    record = np.zeros(1, dtype=_QMT_DAILY_DTYPE)
    record["date"] = int(pd.Timestamp("2024-01-02", tz="Asia/Shanghai").timestamp())
    record["open"], record["high"], record["low"], record["close"] = 10000, 11000, 9000, 10500
    record["volume"], record["amount"] = 123, 456789
    with path.open("wb") as handle:
        handle.write(b"QMTDATA0")
        record.tofile(handle)
    frame = read_qmt_daily_dat(path, "000001.SZ")
    assert frame.loc[0, "close"] == 10.5
    assert frame.loc[0, "volume"] == 123
    assert frame.loc[0, "amount"] == 456789


def test_qmt_dat_parser_skips_invalid_ohlc_rows(tmp_path) -> None:
    path = tmp_path / "000001.DAT"
    records = np.zeros(2, dtype=_QMT_DAILY_DTYPE)
    records["date"] = [
        int(pd.Timestamp("2024-01-02", tz="Asia/Shanghai").timestamp()),
        int(pd.Timestamp("2024-01-03", tz="Asia/Shanghai").timestamp()),
    ]
    records["open"] = [10000, 0]
    records["high"] = [11000, 11000]
    records["low"] = [9000, 0]
    records["close"] = [10500, 10500]
    records["volume"], records["amount"] = 123, 456789
    with path.open("wb") as handle:
        handle.write(b"QMTDATA0")
        records.tofile(handle)
    frame = read_qmt_daily_dat(path, "000001.SZ")
    assert len(frame) == 1
    assert frame.loc[0, "trade_date"] == pd.Timestamp("2024-01-02")


def test_bad_qmt_record_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.DAT"
    path.write_bytes(b"123456789")
    with pytest.raises(DataContractError):
        read_qmt_daily_dat(path, "000001.SZ")
