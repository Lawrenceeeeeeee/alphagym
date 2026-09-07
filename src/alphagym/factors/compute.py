from __future__ import annotations

import pandas as pd

from alphagym.factors import REGISTRY, FactorContext


def compute_factors(
    daily: pd.DataFrame,
    fundamentals: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    names: list[str] | None = None,
) -> pd.DataFrame:
    selected = names or [spec.name for spec in REGISTRY.list()]
    rows: list[pd.DataFrame] = []
    for signal_date in map(pd.Timestamp, signal_dates):
        context = FactorContext(signal_date.normalize(), daily, fundamentals)
        for name in selected:
            values = REGISTRY.get(name).calculator(context).rename("raw_value")
            if values.empty:
                continue
            frame = values.rename_axis("symbol").reset_index()
            frame["signal_date"] = signal_date.normalize()
            frame["factor_name"] = name
            frame["neutralized_value"] = pd.NA
            frame["available_date"] = signal_date.normalize()
            rows.append(frame)
    columns = ["signal_date", "symbol", "factor_name", "raw_value", "neutralized_value", "available_date"]
    return pd.concat(rows, ignore_index=True)[columns] if rows else pd.DataFrame(columns=columns)

