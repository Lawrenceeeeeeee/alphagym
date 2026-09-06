"""Fail-closed preflight for the ML tradeability experiments.

The generic raw-factor workflow deliberately has weaker data requirements.
These checks are specific to neutralized ML and tradable portfolio claims.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlquant import storage_io


def audit_reassessment_data(root: Path, *, index_code: str = "ALL_A") -> dict:
    equity = Path(root) / "equity"
    issues: list[dict] = []
    facts: dict = {}

    def issue(code: str, message: str, *, severity: str = "error") -> None:
        issues.append({"code": code, "message": message, "severity": severity})

    required = ["daily", "adjustments", "calendar", "fundamentals", "industries", "status"]
    if index_code != "ALL_A":
        required.append("index_members")
    for name in required:
        if not storage_io.exists(equity / f"{name}.parquet"):
            issue("MISSING_TABLE", f"Missing equity/{name}.parquet")
    metadata_path = equity / "metadata.json"
    metadata = json.loads(storage_io.read_text(metadata_path, encoding="utf-8")) if storage_io.exists(metadata_path) else {}
    if not storage_io.exists(metadata_path):
        issue("MISSING_PROVENANCE", "Missing equity/metadata.json; formal provenance is unverified")
    if metadata.get("industry_snapshot_only") or "backfill" in str(
        metadata.get("sw1_gap_fill", "")
    ).lower():
        issue("NON_PIT_INDUSTRY", "Industry history contains a snapshot or backward-filled intervals")
    if storage_io.exists(equity / "industries.parquet"):
        industries = storage_io.read_frame(equity / "industries.parquet")
        bad = industries["industry_code"].isna() | industries["industry_code"].eq("UNKNOWN")
        if "source" in industries:
            bad |= industries["source"].eq("fallback")
        facts["unknown_industry_rows"] = int(bad.sum())
        if bad.any():
            issue("UNKNOWN_INDUSTRY", f"{int(bad.sum())} industry rows lack genuine PIT SW1 classification")
    if storage_io.exists(equity / "fundamentals.parquet"):
        financial = storage_io.read_frame(equity / "fundamentals.parquet")
        if "available_date" not in financial:
            issue("MISSING_FINANCIAL_AVAILABILITY", "Financial available_date is missing")
        else:
            available = pd.to_datetime(financial["available_date"], errors="coerce")
            fiscal = pd.to_datetime(financial["stat_date"], errors="coerce")
            bad = available.isna() | fiscal.isna() | (available < fiscal)
            facts["invalid_financial_dates"] = int(bad.sum())
            if bad.any():
                issue("INVALID_FINANCIAL_AVAILABILITY", f"{int(bad.sum())} invalid financial dates")
        issue("FINANCIAL_VINTAGES_UNVERIFIED", "Fiscal-period deduplication in the importer discards revisions; source vintages must be verified", severity="warning")
    if storage_io.exists(equity / "index_members.parquet"):
        members = storage_io.read_frame(equity / "index_members.parquet")
        for column in ("valid_from", "valid_to"):
            members[column] = pd.to_datetime(members[column])
        when = pd.Timestamp("2024-01-31")
        active = members[(members.valid_from <= when) & (members.valid_to.isna() | (members.valid_to >= when))]
        snapshots = {}
        for code, block in members.groupby("index_code", observed=True):
            history = block[block.valid_from <= when]
            latest = history[history.valid_from == history.valid_from.max()]
            current = active[active.index_code == code]
            snapshots[str(code)] = {
                "interval_count": int(current.symbol.nunique()),
                "snapshot_count": int(latest.symbol.nunique()),
                "interval_weight_sum": float(current.benchmark_weight.sum()),
                "snapshot_weight_sum": float(latest.benchmark_weight.sum()),
            }
            # Same-index complete snapshots replace the whole old basket.
            dates = block[["valid_from"]].drop_duplicates().sort_values("valid_from")
            next_dates = pd.Series(dates.valid_from.shift(-1).to_numpy(), index=dates.valid_from)
            next_date = block.valid_from.map(next_dates)
            stale = next_date.notna() & (block.valid_to.isna() | (block.valid_to >= next_date))
            if stale.any():
                issue("STALE_INDEX_INTERVAL", f"{code}: {int(stale.sum())} intervals survive a newer complete snapshot", severity="error" if index_code == code else "warning")
        facts["index_members_2024_01_31"] = snapshots
    for key, code in (("limit_status_unavailable", "MISSING_PRICE_LIMITS"), ("st_status_unavailable", "MISSING_HISTORICAL_ST")):
        if metadata.get(key):
            issue(code, f"Data provenance declares {key}=true; realistic execution cannot be certified")
    errors = [item for item in issues if item["severity"] == "error"]
    return {
        "ok": not errors, "index_code": index_code, "facts": facts, "issues": issues,
        "error": {"code": "REASSESSMENT_DATA_BLOCKED", "message": "; ".join(item["message"] for item in errors)} if errors else None,
    }


def require_reassessment_data(root: Path, output: Path, *, index_code: str = "ALL_A") -> dict:
    result = audit_reassessment_data(root, index_code=index_code)
    output.mkdir(parents=True, exist_ok=True)
    storage_io.write_text(output / f"audit_{index_code}.json",
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not result["ok"]:
        raise ValueError(result["error"]["message"])
    return result
