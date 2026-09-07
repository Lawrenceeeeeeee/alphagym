from __future__ import annotations

import pandas as pd

from alphagym.adapters import ParquetBundleAdapter, QmtDailyAdapter
from alphagym.equity_data import SCHEMAS


def test_parquet_adapter_normalizes_common_aliases(bundle, tmp_path) -> None:
    paths = {}
    for table in SCHEMAS:
        frame = getattr(bundle, table).copy()
        if table == "fundamentals":
            frame = frame.rename(columns={"symbol": "code", "available_date": "avail_date"})
        path = tmp_path / f"{table}.parquet"
        frame.to_parquet(path, index=False)
        paths[table] = path
    imported = ParquetBundleAdapter(paths).load()
    assert {"symbol", "available_date"}.issubset(imported.fundamentals)
    pd.testing.assert_series_equal(
        imported.fundamentals["roe"].reset_index(drop=True),
        bundle.fundamentals["roe"].reset_index(drop=True),
    )


def test_qmt_adapter_enumerates_only_a_share_prefixes(tmp_path) -> None:
    for exchange, numbers in {
        "SH": ("600000", "000300", "501006"),
        "SZ": ("000001", "300001", "114317", "399001"),
        "BJ": ("430001", "830001"),
    }.items():
        folder = tmp_path / exchange / "86400"
        folder.mkdir(parents=True)
        for number in numbers:
            (folder / f"{number}.DAT").touch()
    assert QmtDailyAdapter(tmp_path).symbols() == [
        "600000.SH",
        "000001.SZ",
        "300001.SZ",
        "430001.BJ",
        "830001.BJ",
    ]
