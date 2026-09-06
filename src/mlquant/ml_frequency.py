"""Experimental daily/weekly PIT feature panels for ML portfolio research.

The formal report engine remains monthly. This module rebuilds the curated
factor set at a denser signal calendar. Market factors are rolled on daily
observations; quarterly fields become visible only on ``available_date`` and
are then carried forward until a newer filing becomes available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlquant import storage_io
from mlquant.ml_composite import pit_industry

MARKET_FACTORS = {
    "AMIHUD_7D",
    "TURNOVER_MEAN_5D",
    "TURNOVER_VOL_20D",
    "VOLATILITY_60D",
    "UPSIDE_VOL_60D",
    "MAX_RETURN_20D",
    "REVERSAL_5D",
    "REVERSAL_20D",
    "MOMENTUM_120D",
    "ATR_14D_PCT",
    "KELTNER_LOWER_20_10_PCT",
    "BOLL_LOWER_20D_PCT",
    "DONCHIAN_WIDTH_20D",
}
FINANCIAL_FACTORS = {
    "BP_MRQ",
    "SP_TTM",
    "CFP_TTM",
    "ROE_STABILITY_8Q",
    "GROSS_MARGIN_STABILITY_8Q",
    "REVENUE_YOY",
    "NET_PROFIT_YOY",
}


def neutralize_frequency_features(
    wide: pd.DataFrame,
    industries: pd.DataFrame,
    cap: pd.Series,
) -> pd.DataFrame:
    """Matrix-oriented equivalent of the formal 5-MAD/WLS cross-section.

    The monthly implementation builds pandas frames once per factor and date.
    That is intentionally explicit but prohibitively slow at daily frequency.
    Here the same operations are performed on NumPy arrays while constructing
    the industry design matrix only once per valid mask.
    """
    dates = wide.index.get_level_values("signal_date").unique()
    industry = pit_industry(industries, dates).reindex(wide.index)
    cap = cap.reindex(wide.index)
    result = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns, dtype=np.float32)
    for date in dates:
        block = wide.xs(date, level="signal_date")
        block_index = pd.MultiIndex.from_product([[date], block.index], names=wide.index.names)
        block_industry = industry.reindex(block_index).to_numpy()
        block_cap = cap.reindex(block_index).to_numpy(float)
        positive_cap = np.isfinite(block_cap) & (block_cap > 0)
        base_valid = positive_cap & pd.notna(block_industry)
        if not base_valid.all():
            raise ValueError(
                f"missing PIT industry/positive float cap on {pd.Timestamp(date).date()}: "
                f"{int((~base_valid).sum())} rows; neutralization must not silently impute context"
            )
        if base_valid.sum() < 3:
            continue
        raw = block.to_numpy(float)
        medians = np.nanmedian(raw, axis=0)
        mads = np.nanmedian(np.abs(raw - medians), axis=0)
        lower = medians - 5 * mads
        upper = medians + 5 * mads
        lower = np.where(mads > 0, lower, -np.inf)
        upper = np.where(mads > 0, upper, np.inf)
        winsorized = np.minimum(np.maximum(raw, lower), upper)
        means = np.nanmean(winsorized, axis=0)
        deviations = np.nanstd(winsorized, axis=0)
        usable = np.isfinite(deviations) & (deviations > 0)
        standardized = np.full(raw.shape, np.nan, dtype=float)
        np.divide(
            winsorized - means,
            deviations,
            out=standardized,
            where=usable,
        )
        labels, codes = np.unique(
            block_industry[base_valid].astype(str), return_inverse=True
        )
        dummies = np.eye(len(labels), dtype=float)[codes]
        design = np.column_stack([dummies, np.log(block_cap[base_valid])])
        design = design[:, np.var(design, axis=0) > 0]
        design = np.column_stack([np.ones(base_valid.sum()), design])
        base_positions = np.flatnonzero(base_valid)
        groups: dict[bytes, list[int]] = {}
        for column_position in np.flatnonzero(usable):
            local_valid = np.isfinite(raw[base_valid, column_position])
            groups.setdefault(local_valid.tobytes(), []).append(int(column_position))
        block_result = np.full(raw.shape, np.nan, dtype=np.float32)
        for key, column_positions in groups.items():
            local_valid = np.frombuffer(key, dtype=bool)
            if local_valid.sum() < 3:
                continue
            positions = base_positions[local_valid]
            x = design[local_valid]
            y = standardized[np.ix_(positions, column_positions)]
            weights = block_cap[positions] / np.median(block_cap[positions])
            square_root = np.sqrt(weights)
            coefficient = np.linalg.lstsq(
                x * square_root[:, None], y * square_root[:, None], rcond=None
            )[0]
            residual = y - x @ coefficient
            normalized_weight = weights / weights.sum()
            residual_mean = np.sum(residual * normalized_weight[:, None], axis=0)
            residual_std = np.sqrt(
                np.sum(
                    (residual - residual_mean) ** 2 * normalized_weight[:, None], axis=0
                )
            )
            valid_std = np.isfinite(residual_std) & (residual_std > 0)
            normalized = np.full(residual.shape, np.nan, dtype=np.float32)
            normalized[:, valid_std] = (
                (residual[:, valid_std] - residual_mean[valid_std])
                / residual_std[valid_std]
            ).astype(np.float32)
            block_result[np.ix_(positions, column_positions)] = normalized
        result.loc[block_index, :] = block_result
    return result


def frequency_signal_dates(
    calendar: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    frequency: str,
) -> pd.DatetimeIndex:
    opened = pd.to_datetime(
        calendar.loc[calendar["is_open"].astype(bool), "trade_date"]
    ).drop_duplicates().sort_values()
    if frequency == "daily":
        return pd.DatetimeIndex(opened[(opened >= pd.Timestamp(start)) & (opened <= pd.Timestamp(end))])
    frame = pd.DataFrame({"trade_date": opened})
    if frequency == "weekly":
        periods = frame["trade_date"].dt.to_period("W-FRI")
    elif frequency == "monthly":
        periods = frame["trade_date"].dt.to_period("M")
    else:
        raise ValueError(f"frequency must be daily/weekly/monthly, got {frequency}")
    values = frame.groupby(periods)["trade_date"].max()
    # Never promote the last observed bar of an unfinished week/month to a
    # rebalance. Calendar coverage must certify the entire period first.
    covered = values.index.end_time.normalize() <= pd.to_datetime(calendar["trade_date"]).max()
    values = values[covered & (values >= pd.Timestamp(start)) & (values <= pd.Timestamp(end))]
    return pd.DatetimeIndex(values.to_numpy()).sort_values()


def _rolling(series: pd.Series, window: int, minimum: int, operation: str) -> pd.Series:
    result = getattr(
        series.groupby(level=0, observed=True).rolling(window, min_periods=minimum),
        operation,
    )()
    return result.reset_index(level=0, drop=True).reindex(series.index)


def _ewm(series: pd.Series, span: int, minimum: int) -> pd.Series:
    result = series.groupby(level=0, observed=True).ewm(
        span=span, adjust=False, min_periods=minimum
    ).mean()
    return result.reset_index(level=0, drop=True).reindex(series.index)


def market_features(
    daily: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    factor_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    requested = set(factor_ids) & MARKET_FACTORS
    unknown = set(factor_ids) - MARKET_FACTORS - FINANCIAL_FACTORS
    if unknown:
        raise ValueError(f"unsupported high-frequency factors: {sorted(unknown)}")
    indexed = daily.set_index(["symbol", "trade_date"]).sort_index()
    observations = daily[daily["trade_date"].isin(signal_dates)][
        ["trade_date", "symbol", "close", "float_market_cap"]
    ].rename(columns={"trade_date": "signal_date"})
    target = pd.MultiIndex.from_frame(observations[["symbol", "signal_date"]])
    close = indexed["adj_close"]
    high = indexed["adj_high"]
    low = indexed["adj_low"]
    turnover = indexed["turnover"]
    amount = indexed["amount"].replace(0, np.nan)
    returns = close.groupby(level=0, observed=True).pct_change(fill_method=None)
    previous_close = close.groupby(level=0, observed=True).shift()
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    output: dict[str, np.ndarray] = {}

    def put(name: str, values: pd.Series) -> None:
        if name in requested:
            output[name] = values.reindex(target).to_numpy(np.float32)

    illiquidity = returns.abs() / amount
    put("AMIHUD_7D", _rolling(illiquidity, 7, 6, "mean") * 1e8)
    put("TURNOVER_MEAN_5D", _rolling(turnover, 5, 4, "mean"))
    put("TURNOVER_VOL_20D", _rolling(turnover, 20, 16, "std"))
    put("VOLATILITY_60D", _rolling(returns, 60, 48, "std"))
    put("UPSIDE_VOL_60D", _rolling(returns.clip(lower=0), 60, 48, "std"))
    put("MAX_RETURN_20D", _rolling(returns, 20, 16, "max"))
    for window, name, sign in (
        (5, "REVERSAL_5D", -1.0),
        (20, "REVERSAL_20D", -1.0),
        (120, "MOMENTUM_120D", 1.0),
    ):
        shifted = close.groupby(level=0, observed=True).shift(window)
        put(name, sign * (close / shifted - 1.0))
    atr14 = _rolling(true_range, 14, 11, "mean")
    put("ATR_14D_PCT", atr14 / close.replace(0, np.nan))
    mid = _rolling(close, 20, 16, "mean")
    std = _rolling(close, 20, 16, "std")
    put("BOLL_LOWER_20D_PCT", (mid - 2 * std) / close.replace(0, np.nan) - 1.0)
    keltner_mid = _ewm(close, 20, 20)
    keltner_atr = _rolling(true_range, 10, 8, "mean")
    put(
        "KELTNER_LOWER_20_10_PCT",
        (keltner_mid - keltner_atr) / close.replace(0, np.nan) - 1.0,
    )
    upper = _rolling(high, 20, 16, "max")
    lower = _rolling(low, 20, 16, "min")
    put("DONCHIAN_WIDTH_20D", (upper - lower) / close.replace(0, np.nan))
    features = pd.DataFrame(output, index=target)
    features.index.names = ["symbol", "signal_date"]
    features = features.reorder_levels(["signal_date", "symbol"]).sort_index()
    observations = observations.set_index(["signal_date", "symbol"]).sort_index()
    return features, observations


def financial_events(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Replay disclosures, including revisions, without consulting future rows.

    Each field uses the latest non-missing fiscal observation, matching the
    FactorRegistry operators. A late older filing can update TTM dependencies
    but cannot replace a newer fiscal period as the current observation.
    """
    frame = fundamentals.copy()
    frame["stat_date"] = pd.to_datetime(frame["stat_date"]).dt.normalize()
    frame["available_date"] = pd.to_datetime(frame["available_date"]).dt.normalize()
    if frame[["symbol", "stat_date", "available_date"]].isna().any().any():
        raise ValueError("financial PIT requires symbol, stat_date and available_date")
    if (frame["available_date"] < frame["stat_date"]).any():
        raise ValueError("financial available_date precedes fiscal period end")
    frame = frame.drop_duplicates()
    if frame.duplicated(["symbol", "stat_date", "available_date"]).any():
        raise ValueError("conflicting financial revisions on the same available_date")
    fields = ("bps", "revenue", "ocfps", "net_profit", "roe", "gross_margin")
    for column in fields:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows = []
    for symbol, history in frame.groupby("symbol", observed=True, sort=True):
        state: dict[pd.Timestamp, dict[str, float]] = {}
        for available, disclosed in history.sort_values(
            ["available_date", "stat_date"]
        ).groupby("available_date", sort=True):
            for record in disclosed.itertuples(index=False):
                state[record.stat_date] = {field: getattr(record, field) for field in fields}
            event = {"symbol": symbol, "available_date": available}
            for column in fields:
                values = {
                    date: state[date][column] for date in sorted(state)
                    if pd.notna(state[date][column])
                }
                if not values:
                    continue
                latest = max(values)
                current = values[latest]
                prior_same = values.get(latest - pd.DateOffset(years=1), np.nan)
                if column == "bps":
                    event[column] = current
                if column in ("revenue", "ocfps"):
                    prior_fy = values.get(pd.Timestamp(latest.year - 1, 12, 31), np.nan)
                    event[f"{column}_ttm"] = (
                        current if latest.month == 12 else current + prior_fy - prior_same
                    )
                if column in ("revenue", "net_profit"):
                    event[f"{column}_yoy"] = (
                        current / abs(prior_same) - 1 if prior_same != 0 else np.nan
                    )
                if column in ("roe", "gross_margin"):
                    tail = list(values.values())[-8:]
                    event[f"{column}_stability"] = (
                        -float(np.std(tail, ddof=1)) if len(tail) >= 6 else np.nan
                    )
            rows.append(event)
    columns = [
        "symbol",
        "available_date",
        "bps",
        "revenue_ttm",
        "ocfps_ttm",
        "revenue_yoy",
        "net_profit_yoy",
        "roe_stability",
        "gross_margin_stability",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(["available_date", "symbol"])


def forward_label_end_dates(
    signal_dates: pd.DatetimeIndex, calendar: pd.DataFrame,
) -> pd.Series:
    """Last price timestamp used by each forward open-to-open label."""
    opened = pd.DatetimeIndex(pd.to_datetime(
        calendar.loc[calendar["is_open"].astype(bool), "trade_date"]
    )).drop_duplicates().sort_values()
    signals = signal_dates.drop_duplicates().sort_values()
    positions = opened.searchsorted(signals, side="right")
    executions = pd.Series(pd.NaT, index=signals, dtype="datetime64[ns]")
    valid = positions < len(opened)
    executions.loc[signals[valid]] = opened[positions[valid]]
    return executions.shift(-1).rename("label_end_date")


def add_financial_features(
    market: pd.DataFrame,
    observations: pd.DataFrame,
    fundamentals: pd.DataFrame,
    factor_ids: list[str],
    *,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    requested = set(factor_ids) & FINANCIAL_FACTORS
    if not requested:
        return market
    left = observations.reset_index().sort_values(["signal_date", "symbol"])
    if events is None:
        events = financial_events(fundamentals)
    joined = pd.merge_asof(
        left,
        events,
        left_on="signal_date",
        right_on="available_date",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    ).set_index(["signal_date", "symbol"])
    values: dict[str, pd.Series] = {
        "BP_MRQ": joined["bps"] / joined["close"].replace(0, np.nan),
        "SP_TTM": joined["revenue_ttm"] / joined["float_market_cap"].replace(0, np.nan),
        "CFP_TTM": joined["ocfps_ttm"] / joined["close"].replace(0, np.nan),
        "ROE_STABILITY_8Q": joined["roe_stability"],
        "GROSS_MARGIN_STABILITY_8Q": joined["gross_margin_stability"],
        "REVENUE_YOY": joined["revenue_yoy"],
        "NET_PROFIT_YOY": joined["net_profit_yoy"],
    }
    result = market.copy()
    for name in requested:
        result[name] = values[name].reindex(result.index).astype(np.float32)
    return result.reindex(columns=factor_ids)


def forward_open_returns(
    daily: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    calendar: pd.DataFrame,
) -> pd.Series:
    opened = pd.DatetimeIndex(
        pd.to_datetime(calendar.loc[calendar["is_open"].astype(bool), "trade_date"])
    ).sort_values()
    positions = opened.searchsorted(signal_dates, side="right")
    valid_signal = positions < len(opened)
    signals = signal_dates[valid_signal]
    executions = opened[positions[valid_signal]]
    if len(signals) < 2:
        return pd.Series(dtype=np.float32, name="forward_return")
    needed = set(executions)
    prices = daily[daily["trade_date"].isin(needed)].pivot(
        index="trade_date", columns="symbol", values="adj_open"
    )
    current = prices.reindex(executions[:-1]).to_numpy(float)
    following = prices.reindex(executions[1:]).to_numpy(float)
    values = following / current - 1.0
    index = pd.MultiIndex.from_product(
        [signals[:-1], prices.columns], names=["signal_date", "symbol"]
    )
    return pd.Series(values.reshape(-1), index=index, dtype=np.float32, name="forward_return")


def load_frequency_inputs(
    root: str | Path,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    low = pd.Timestamp(start) - pd.Timedelta(days=600)
    high = pd.Timestamp(end) + pd.Timedelta(days=15)
    daily = storage_io.read_frame(
        root / "equity" / "daily.parquet",
        columns=[
            "trade_date", "symbol", "open", "high", "low", "close", "volume",
            "amount", "turnover", "float_market_cap",
        ],
        filters=[("trade_date", ">=", low), ("trade_date", "<=", high)],
    )
    adjustments = storage_io.read_frame(
        root / "equity" / "adjustments.parquet",
        columns=["trade_date", "symbol", "adjust_factor"],
    )
    adjustments = adjustments[pd.to_datetime(adjustments["trade_date"]) <= high]
    calendar = storage_io.read_frame(root / "equity" / "calendar.parquet")
    fundamentals = storage_io.read_frame(root / "equity" / "fundamentals.parquet")
    for frame in (daily, adjustments, calendar):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    daily = pd.merge_asof(
        daily.sort_values(["trade_date", "symbol"]),
        adjustments.sort_values(["trade_date", "symbol"]),
        on="trade_date",
        by="symbol",
        direction="backward",
    )
    daily["adjust_factor"] = daily["adjust_factor"].fillna(1.0)
    for column in ("open", "high", "low", "close"):
        daily[f"adj_{column}"] = daily[column] * daily["adjust_factor"]
    return daily, fundamentals, calendar, adjustments
