from __future__ import annotations

import json

import pandas as pd
import pytest

from mlquant import storage_io
from mlquant.reassessment_audit import audit_reassessment_data, require_reassessment_data


def test_missing_data_fails_closed_and_writes_structured_audit(tmp_path) -> None:
    result = audit_reassessment_data(tmp_path)
    assert not result["ok"]
    assert result["error"]["code"] == "REASSESSMENT_DATA_BLOCKED"
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="Missing"):
        require_reassessment_data(tmp_path, output)
    assert json.loads(storage_io.read_text(output / "audit_ALL_A.json"))["ok"] is False


def test_backfilled_industry_and_missing_limits_never_pass_as_formal(tmp_path) -> None:
    equity = tmp_path / "equity"
    equity.mkdir()
    storage_io.write_text(equity / "metadata.json", json.dumps({
        "formal": True, "sw1_gap_fill": "carry-forward/backfill",
        "limit_status_unavailable": True, "st_status_unavailable": True,
    }))
    result = audit_reassessment_data(tmp_path)
    codes = {issue["code"] for issue in result["issues"]}
    assert {"NON_PIT_INDUSTRY", "MISSING_PRICE_LIMITS", "MISSING_HISTORICAL_ST"} <= codes


def test_index_member_dropped_from_snapshot_cannot_remain_active(tmp_path) -> None:
    equity = tmp_path / "equity"
    equity.mkdir()
    members = pd.DataFrame({
        "index_code": ["000300.SH"] * 3,
        "symbol": ["A", "B", "B"],
        "valid_from": pd.to_datetime(["2023-12-01", "2023-12-01", "2024-01-01"]),
        "valid_to": pd.to_datetime([None, "2023-12-31", None]),
        "benchmark_weight": [50., 50., 100.],
    })
    storage_io.write_frame(members, equity / "index_members.parquet", index=False)
    result = audit_reassessment_data(tmp_path, index_code="000300.SH")
    issue = next(item for item in result["issues"] if item["code"] == "STALE_INDEX_INTERVAL")
    assert issue["severity"] == "error"
    counts = result["facts"]["index_members_2024_01_31"]["000300.SH"]
    assert counts["interval_count"] == 2
    assert counts["snapshot_count"] == 1
