from __future__ import annotations

import pandas as pd

PERIODS = {
    "development": (pd.Timestamp("2014-01-01"), pd.Timestamp("2020-12-31")),
    "validation": (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31")),
    "test": (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")),
}


def assign_period(
    dates: pd.Series,
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> pd.Series:
    result = pd.Series(pd.NA, index=dates.index, dtype="string")
    normalized = pd.to_datetime(dates)
    for name, (start, end) in (periods or PERIODS).items():
        result.loc[normalized.between(start, end)] = name
    return result


def assert_selection_isolation(selection_dates: pd.Series) -> None:
    dates = pd.to_datetime(selection_dates)
    if (dates >= PERIODS["test"][0]).any():
        raise ValueError("test/2026 observations cannot select direction, factors, or combine parameters")


def turnover_by_date(panel: pd.DataFrame, value_column: str) -> pd.Series:
    ranks = panel.pivot(index="signal_date", columns="symbol", values=value_column).rank(axis=1, pct=True)
    return ranks.diff().abs().mean(axis=1).rename("rank_turnover")

