from __future__ import annotations

import pandas as pd

from mlquant.signal_export import _next_month_signal_date, _next_open_date


def test_next_open_date_is_strictly_after_signal() -> None:
    calendar = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-08-31", "2026-09-01", "2026-09-02"]),
            "is_open": [True, True, True],
        }
    )
    assert _next_open_date(pd.Timestamp("2026-08-31"), calendar) == pd.Timestamp("2026-09-01")


def test_next_month_signal_date_requires_complete_calendar_month() -> None:
    partial = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03"]),
            "is_open": [True, True, True],
        }
    )
    assert _next_month_signal_date(pd.Timestamp("2026-08-31"), partial) is None

    complete = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-09-29", "2026-09-30"]),
            "is_open": [True, False],
        }
    )
    assert _next_month_signal_date(pd.Timestamp("2026-08-31"), complete) == pd.Timestamp(
        "2026-09-29"
    )
