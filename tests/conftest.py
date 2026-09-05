from __future__ import annotations

import pandas as pd
import pytest

from mlquant.equity_data import EquityDataBundle


@pytest.fixture
def bundle() -> EquityDataBundle:
    dates = pd.bdate_range("2024-01-02", periods=300)
    symbols = ["000001.SZ", "600000.SH"]
    daily_rows = []
    status_rows = []
    for number, symbol in enumerate(symbols):
        for index, date in enumerate(dates):
            close = 10 + number + index * 0.01
            daily_rows.append({
                "trade_date": date, "symbol": symbol, "open": close - .01, "high": close + .1,
                "low": close - .1, "close": close, "volume": 1000 + index, "amount": 1e6,
                "adj_close": close, "turnover": .01 + number * .001,
                "float_market_cap": 1e10 * (number + 1),
            })
            status_rows.append({
                "trade_date": date, "symbol": symbol, "is_st": False, "is_pt": False,
                "is_suspended": False, "limit_up": close * 1.1, "limit_down": close * .9,
            })
    fundamentals = []
    for symbol in symbols:
        for quarter in pd.date_range("2022-03-31", "2024-12-31", freq="QE"):
            q = quarter.quarter
            fundamentals.append({
                "symbol": symbol, "stat_date": quarter, "available_date": quarter + pd.Timedelta(days=45),
                "roe": 10 + q, "gross_margin": 30 + q, "rev_yoy": 8.0, "np_yoy": 7.0,
                "bps": 5.0, "eps": q * .2, "ocfps": q * .25,
                "net_profit": q * 1e8, "revenue": q * 1e9,
            })
    return EquityDataBundle(
        daily=pd.DataFrame(daily_rows),
        adjustments=pd.DataFrame([{"trade_date": dates[0], "symbol": s, "adjust_factor": 1.0} for s in symbols]),
        calendar=pd.DataFrame({"trade_date": dates, "is_open": True}),
        securities=pd.DataFrame({"symbol": symbols, "list_date": pd.Timestamp("2010-01-01"), "delist_date": pd.NaT}),
        status=pd.DataFrame(status_rows),
        fundamentals=pd.DataFrame(fundamentals),
        industries=pd.DataFrame({
            "symbol": symbols, "industry_code": ["801010", "801020"], "industry_name": ["农林牧渔", "煤炭"],
            "valid_from": pd.Timestamp("2010-01-01"), "valid_to": pd.NaT,
            "source": "gold", "version": "1",
        }),
        index_members=pd.DataFrame({
            "index_code": "000300.SH", "symbol": symbols, "valid_from": pd.Timestamp("2010-01-01"),
            "valid_to": pd.NaT, "benchmark_weight": [.4, .6],
        }),
    )
