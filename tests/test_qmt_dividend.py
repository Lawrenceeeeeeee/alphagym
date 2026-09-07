from __future__ import annotations

import pandas as pd

from alphagym.qmt_dividend import _ex_date, build_adjustments, parse_dividend_key


def test_parse_dividend_key() -> None:
    key = b"SH|600626|4000|1370188800000\x01\x8c\xee\x02\x00\x00\x00\x00"
    assert parse_dividend_key(key) == ("SH", "600626", "4000", 1370188800000)
    assert parse_dividend_key(b"not-a-key") is None
    assert parse_dividend_key(b"SH|600626") is None


def test_ex_date_is_shanghai_midnight() -> None:
    # 1370188800000 ms == 2013-06-02 16:00 UTC == 2013-06-03 00:00 Asia/Shanghai
    assert _ex_date(1370188800000) == pd.Timestamp("2013-06-03")


def test_build_adjustments_is_cumulative_per_symbol() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["600626.SH", "600626.SH", "000001.SZ"],
            "ex_date": pd.to_datetime(["2013-06-03", "2014-06-19", "2013-01-01"]),
            "factor": [1.5, 2.0, 1.1],
        }
    )
    out = build_adjustments(events)
    assert list(out.columns) == ["trade_date", "symbol", "adjust_factor"]
    assert out.loc[out["symbol"] == "600626.SH", "adjust_factor"].tolist() == [1.5, 3.0]
    assert out.loc[out["symbol"] == "000001.SZ", "adjust_factor"].tolist() == [1.1]
