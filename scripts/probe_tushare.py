"""Probe which Tushare Pro interfaces the current token can actually access.

Reads TUSHARE_TOKEN from the project ``.env`` and calls each target interface
with a minimal query.  Prints a JSON summary of OK / denied / error per
interface, plus the exact API error message so point-tier gaps are visible.
"""
from __future__ import annotations

import json
from pathlib import Path

import tushare as ts

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_token() -> str:
    for candidate in (PROJECT_ROOT / ".env",):
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TUSHARE_TOKEN="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    raise SystemExit("TUSHARE_TOKEN not found in .env")


# name -> (interface, kwargs)
TARGETS: dict[str, tuple[str, dict[str, str]]] = {
    "daily": ("daily", {"trade_date": "20240103"}),
    "adj_factor": ("adj_factor", {"trade_date": "20240103"}),
    "daily_basic": ("daily_basic", {"trade_date": "20240103"}),
    "trade_cal": ("trade_cal", {"exchange": "SSE", "start_date": "20240101", "end_date": "20240110"}),
    "stock_basic": ("stock_basic", {"list_status": "L"}),
    "namechange": ("namechange", {"ts_code": "000001.SZ"}),
    "fina_indicator": ("fina_indicator", {"ts_code": "000001.SZ", "period": "20231231"}),
    "income": ("income", {"ts_code": "000001.SZ", "period": "20231231"}),
    "balancesheet": ("balancesheet", {"ts_code": "000001.SZ", "period": "20231231"}),
    "cashflow": ("cashflow", {"ts_code": "000001.SZ", "period": "20231231"}),
    "index_weight": ("index_weight", {"index_code": "000300.SH", "start_date": "20240101", "end_date": "20240131"}),
    "index_member": ("index_member", {"index_code": "399300.SZ"}),
    "index_classify": ("index_classify", {"level": "L1", "src": "SW2021"}),
}


def main() -> int:
    pro = ts.pro_api(load_token())
    results: dict[str, dict[str, object]] = {}
    for name, (interface, kwargs) in TARGETS.items():
        try:
            frame = getattr(pro, interface)(**kwargs)
            if frame is None or getattr(frame, "empty", True):
                results[name] = {"status": "ok_empty", "rows": 0}
            else:
                results[name] = {
                    "status": "ok",
                    "rows": len(frame),
                    "columns": list(frame.columns)[:12],
                }
        except Exception as error:  # noqa: BLE001 - report raw API denial text
            results[name] = {"status": "error", "error": str(error)[:400]}
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
