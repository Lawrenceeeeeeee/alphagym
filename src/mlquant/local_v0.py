"""Build and run the explicitly non-formal local A-share v0 study.

The local source set has point-in-time index membership, QMT raw daily bars and
backward-adjustment events, but no historical SW1 classification or official
benchmark weights.  Outputs from this module are therefore diagnostics and carry
an explicit watermark; they cannot pass the formal data audit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from mlquant import storage_io
from mlquant.audit import assign_period
from mlquant.charts import write_charts
from mlquant.combine import METHODS, combine_scores, factor_weights, select_validation_method
from mlquant.factors import REGISTRY
from mlquant.factors.technical import (
    BLOCKED_SOURCE_FACTORS,
    SOURCE_ALIASES,
    TECHNICAL_FACTOR_NAMES,
    TECHNICAL_METADATA,
    compute_technical_features,
)
from mlquant.research import (
    benjamini_hochberg,
    evaluate_factor_batch,
    mad_winsorize,
    zscore,
)

INDEX_FILES = {
    "000300.SH": "constituents_hs300.parquet",
    "000905.SH": "constituents_zz500.parquet",
}
PRICE_VOLUME_FACTORS = (
    "REVERSAL_5D",
    "REVERSAL_20D",
    "MOMENTUM_60D",
    "MOMENTUM_120D",
    "MOMENTUM_120D_SKIP20",
    "MOMENTUM_252D_SKIP20",
    "HIGH_252D_PROXIMITY",
    "TURNOVER_MEAN_20D",
    "TURNOVER_MEAN_60D",
    "TURNOVER_MEAN_120D",
    "TURNOVER_BIAS_20_480D",
    "TURNOVER_BIAS_60_480D",
    "TURNOVER_VOL_20D",
    "TURNOVER_VOL_60D",
    "AMIHUD_20D",
    "VOLATILITY_20D",
    "VOLATILITY_60D",
    "VOLATILITY_120D",
    "UPSIDE_VOL_60D",
    "DOWNSIDE_VOL_60D",
    "IDIO_VOL_60D",
    "IDIO_SKEW_60D",
    "MAX_RETURN_20D",
)
FUNDAMENTAL_FACTORS = tuple(
    spec.name
    for spec in REGISTRY.list()
    if spec.family in {"value", "growth", "quality"} and spec.name != "SP_TTM"
)
SHORT_HORIZON_FACTORS = (
    "REVERSAL_1D",
    "REVERSAL_2D",
    "REVERSAL_3D",
    "REVERSAL_5D",
    "REVERSAL_7D",
    "REVERSAL_10D",
    "REVERSAL_20D",
    "HIGH_20D_PROXIMITY",
    "HIGH_60D_PROXIMITY",
    "HIGH_120D_PROXIMITY",
    "HIGH_252D_PROXIMITY",
    "TURNOVER_MEAN_1D",
    "TURNOVER_MEAN_5D",
    "TURNOVER_MEAN_7D",
    "TURNOVER_MEAN_10D",
    "TURNOVER_MEAN_20D",
    "TURNOVER_VOL_5D",
    "TURNOVER_VOL_7D",
    "TURNOVER_VOL_10D",
    "TURNOVER_VOL_20D",
    "AMIHUD_5D",
    "AMIHUD_7D",
    "AMIHUD_10D",
    "AMIHUD_20D",
    "ABS_RETURN_1D",
    "VOLATILITY_2D",
    "VOLATILITY_3D",
    "VOLATILITY_5D",
    "VOLATILITY_7D",
    "VOLATILITY_10D",
    "VOLATILITY_20D",
    "UPSIDE_VOL_10D",
    "UPSIDE_VOL_20D",
    "UPSIDE_VOL_60D",
    "DOWNSIDE_VOL_10D",
    "DOWNSIDE_VOL_20D",
    "DOWNSIDE_VOL_60D",
    "MAX_RETURN_5D",
    "MAX_RETURN_7D",
    "MAX_RETURN_10D",
    "MAX_RETURN_20D",
)


@dataclass(frozen=True, slots=True)
class LocalSourcePaths:
    qmt_root: Path
    source_root: Path
    data_root: Path

    @property
    def meta_root(self) -> Path:
        return self.source_root / "data" / "universe" / "meta"

    @property
    def equity_root(self) -> Path:
        return self.data_root / "equity"

    @property
    def artifact_root(self) -> Path:
        return self.data_root / "artifacts" / "equity_v0"


def vendor_symbol(value: str) -> str:
    """Convert ``sh.600000``/``600000.SH``/bare six-digit codes to canonical form."""
    text = str(value).strip()
    if len(text) == 9 and text[2] == ".":
        return f"{text[3:]}.{text[:2].upper()}"
    if len(text) == 9 and text[6] == ".":
        return f"{text[:6]}.{text[7:].upper()}"
    if len(text) == 6 and text.isdigit():
        exchange = "SH" if text.startswith(("5", "6", "9")) else "BJ" if text.startswith(("4", "8")) else "SZ"
        return f"{text}.{exchange}"
    raise ValueError(f"unsupported security code: {value!r}")


def _constituent_snapshots(paths: LocalSourcePaths) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for index_code, filename in INDEX_FILES.items():
        frame = storage_io.read_frame(paths.meta_root / filename)
        frame = frame.assign(
            index_code=index_code,
            symbol=frame["code"].map(vendor_symbol),
            valid_from=pd.to_datetime(frame["updateDate"]).dt.normalize(),
        )
        snapshots = sorted(frame["valid_from"].unique())
        end_by_start = {
            start: pd.Timestamp(snapshots[position + 1]) - pd.Timedelta(days=1)
            if position + 1 < len(snapshots)
            else pd.NaT
            for position, start in enumerate(snapshots)
        }
        frame["valid_to"] = frame["valid_from"].map(end_by_start)
        counts = frame.groupby("valid_from", observed=True)["symbol"].transform("count")
        frame["benchmark_weight"] = 1.0 / counts
        frames.append(
            frame[["index_code", "symbol", "valid_from", "valid_to", "benchmark_weight"]]
        )
    return pd.concat(frames, ignore_index=True)


def build_local_contract(paths: LocalSourcePaths) -> dict[str, int]:
    """Create the small canonical tables around an existing QMT daily import."""
    equity = paths.equity_root
    equity.mkdir(parents=True, exist_ok=True)
    members = _constituent_snapshots(paths)
    storage_io.write_frame(members, equity / "index_members.parquet", index=False)
    universe = set(members["symbol"])

    stocks = storage_io.read_frame(paths.meta_root / "stocks.parquet")
    stocks["symbol"] = stocks["code"].map(vendor_symbol)
    stocks = stocks[stocks["symbol"].isin(universe)].copy()
    securities = pd.DataFrame(
        {
            "symbol": stocks["symbol"],
            "list_date": pd.to_datetime(stocks["ipoDate"], errors="coerce"),
            "delist_date": pd.to_datetime(stocks["outDate"], errors="coerce"),
        }
    ).drop_duplicates("symbol", keep="last")
    storage_io.write_frame(securities, equity / "securities.parquet", index=False)

    index_daily = storage_io.read_frame(paths.meta_root / "index_daily.parquet")
    calendar = pd.DataFrame(
        {"trade_date": pd.to_datetime(index_daily["date"]).drop_duplicates().sort_values(), "is_open": True}
    )
    storage_io.write_frame(calendar, equity / "calendar.parquet", index=False)

    aux = storage_io.read_frame(paths.source_root / "data" / "hf_ml" / "bs_daily_aux.parquet")
    aux = aux[aux["code"].isin(universe)].copy()
    status = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(aux["date"]),
            "symbol": aux["code"],
            "is_st": pd.to_numeric(aux["isST"], errors="coerce").fillna(0).astype(bool),
            "is_pt": False,
            "is_suspended": pd.to_numeric(aux["tradestatus"], errors="coerce").fillna(0).ne(1),
            "limit_up": False,
            "limit_down": False,
        }
    )
    storage_io.write_frame(status, equity / "status.parquet", index=False)

    fundamentals = storage_io.read_frame(
        paths.source_root / "data" / "universe" / "fundamentals_quarterly.parquet"
    )
    fundamentals["symbol"] = fundamentals["code"].map(vendor_symbol)
    fundamentals = fundamentals[fundamentals["symbol"].isin(universe)].rename(
        columns={"avail_date": "available_date"}
    )
    fundamentals = fundamentals.drop(columns=["code"])
    storage_io.write_frame(fundamentals, equity / "fundamentals.parquet", index=False)

    current = storage_io.read_frame(paths.meta_root / "industry.parquet")
    current["symbol"] = current["code"].map(vendor_symbol)
    current = current[current["symbol"].isin(universe)]
    industries = pd.DataFrame(
        {
            "symbol": current["symbol"],
            "industry_code": current["industry"].fillna("UNKNOWN"),
            "industry_name": current["industry"].fillna("UNKNOWN"),
            "valid_from": pd.to_datetime(current["updateDate"]),
            "valid_to": pd.NaT,
            "source": "current_snapshot_only",
            "version": "2026-snapshot",
        }
    ).drop_duplicates("symbol", keep="last")
    storage_io.write_frame(industries, equity / "industries.parquet", index=False)

    metadata = {
        "formal": False,
        "watermark": "NON-FORMAL V0: no historical SW1 or official benchmark weights",
        "industry_snapshot_only": True,
        "benchmark_weights_imputed_equal": True,
        "limit_status_unavailable": True,
        "strict_status_start": "2015-01-01",
        "qmt_adjustment": "backward cumulative; adj_price = raw_price * adjust_factor",
    }
    storage_io.write_text(equity / "metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "members": len(members),
        "symbols": len(universe),
        "securities": len(securities),
        "status": len(status),
        "fundamentals": len(fundamentals),
        "industries": len(industries),
    }


def _load_filtered_daily(paths: LocalSourcePaths, symbols: set[str]) -> pd.DataFrame:
    cache = paths.artifact_root / "daily_2013_2026_02_v2.parquet"
    if storage_io.exists(cache):
        return storage_io.read_frame(cache)
    paths.artifact_root.mkdir(parents=True, exist_ok=True)
    source = storage_io.TableReader(paths.equity_root / "daily.parquet")
    wanted = pa.array(sorted(symbols))
    chunks: list[pd.DataFrame] = []
    start = pd.Timestamp("2013-01-01")
    # Jan-Feb 2026 are outcome-only bars needed to close December 2025 signals.
    end = pd.Timestamp("2026-02-28")
    for row_group in range(source.num_row_groups):
        table = source.read_row_group(row_group)
        mask = pc.and_(
            pc.is_in(table["symbol"], value_set=wanted),
            pc.and_(pc.greater_equal(table["trade_date"], start), pc.less_equal(table["trade_date"], end)),
        )
        selected = table.filter(mask)
        if selected.num_rows:
            chunks.append(selected.to_pandas())
    daily = pd.concat(chunks, ignore_index=True).sort_values(["symbol", "trade_date"])
    adjustments = storage_io.read_frame(paths.equity_root / "adjustments.parquet")
    adjustments = adjustments[
        adjustments["symbol"].isin(symbols) & (adjustments["trade_date"] <= end)
    ]
    daily = apply_backward_adjustments(daily, adjustments)
    for column in ("open", "high", "low", "close"):
        daily[f"adj_{column}"] = daily[column] * daily["adjust_factor"]

    aux = storage_io.read_frame(
        paths.source_root / "data" / "hf_ml" / "bs_daily_aux.parquet",
        columns=["code", "date", "turn", "tradestatus", "isST"],
    )
    aux = aux[aux["code"].isin(symbols)].rename(
        columns={"code": "symbol", "date": "trade_date", "turn": "turnover"}
    )
    aux["trade_date"] = pd.to_datetime(aux["trade_date"])
    for column in ("turnover", "tradestatus", "isST"):
        aux[column] = pd.to_numeric(aux[column], errors="coerce")
    daily = daily.merge(aux, on=["trade_date", "symbol"], how="left")
    storage_io.write_frame(daily, cache, index=False)
    return daily


def apply_backward_adjustments(daily: pd.DataFrame, adjustments: pd.DataFrame) -> pd.DataFrame:
    """As-of join sparse cumulative factors, including events before the price slice."""
    left = daily.drop(columns="adjust_factor", errors="ignore").sort_values(
        ["trade_date", "symbol"]
    )
    right = adjustments[["trade_date", "symbol", "adjust_factor"]].sort_values(
        ["trade_date", "symbol"]
    )
    result = pd.merge_asof(
        left,
        right,
        on="trade_date",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )
    result["adjust_factor"] = result["adjust_factor"].fillna(1.0)
    return result.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _rolling(series: pd.Series, window: int, minimum: int, operation: str) -> pd.Series:
    grouped = series.groupby(series.index.get_level_values("symbol"), observed=True)
    rolling = grouped.rolling(window, min_periods=minimum)
    result = getattr(rolling, operation)().reset_index(level=0, drop=True)
    return result.reindex(series.index)


def _market_returns(paths: LocalSourcePaths) -> pd.DataFrame:
    frame = storage_io.read_frame(paths.meta_root / "index_daily.parquet")
    frame = frame[frame["code"].isin(["sh.000300", "sh.000905"])].copy()
    frame["index_code"] = frame["code"].map({"sh.000300": "000300.SH", "sh.000905": "000905.SH"})
    frame["trade_date"] = pd.to_datetime(frame["date"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["market_return"] = frame.groupby("index_code", observed=True)["close"].pct_change(fill_method=None)
    return frame[["index_code", "trade_date", "market_return"]]


def _idio_at_signals(
    daily: pd.DataFrame, signal_rows: pd.DataFrame, market: pd.DataFrame, mode: str
) -> pd.Series:
    result: dict[tuple[str, pd.Timestamp, str], float] = {}
    price_returns = daily.set_index(["symbol", "trade_date"])["adj_close"].groupby(level=0).pct_change()
    base = price_returns.rename("stock_return").reset_index()
    for index_code, market_part in market.groupby("index_code", observed=True):
        joined = base.merge(market_part, on="trade_date", how="left")
        arrays = {
            symbol: group.set_index("trade_date")[["stock_return", "market_return"]]
            for symbol, group in joined.groupby("symbol", observed=True)
        }
        subset = signal_rows[signal_rows["index_code"] == index_code]
        for row in subset.itertuples(index=False):
            history = arrays.get(row.symbol)
            if history is None:
                continue
            sample = history.loc[: row.signal_date].tail(60).dropna()
            if len(sample) < 48 or sample["market_return"].var() == 0:
                continue
            x = np.column_stack([np.ones(len(sample)), sample["market_return"].to_numpy()])
            y = sample["stock_return"].to_numpy()
            residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
            value = pd.Series(residual).std(ddof=1) if mode == "vol" else pd.Series(residual).skew()
            result[(row.symbol, row.signal_date, index_code)] = float(value)
    key = pd.MultiIndex.from_frame(signal_rows[["symbol", "signal_date", "index_code"]])
    return pd.Series(result).reindex(key).reset_index(drop=True)


def _signal_members(paths: LocalSourcePaths, signal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    members = storage_io.read_frame(paths.equity_root / "index_members.parquet")
    securities = storage_io.read_frame(paths.equity_root / "securities.parquet").set_index("symbol")
    open_dates = np.sort(
        storage_io.read_frame(paths.equity_root / "calendar.parquet")["trade_date"].to_numpy(
            dtype="datetime64[ns]"
        )
    )
    rows: list[pd.DataFrame] = []
    for date in signal_dates:
        active = members[
            (members["valid_from"] <= date)
            & (members["valid_to"].isna() | (members["valid_to"] >= date))
        ][["index_code", "symbol"]].copy()
        active["signal_date"] = date
        active["list_date"] = active["symbol"].map(securities["list_date"])
        listed_at = np.searchsorted(open_dates, active["list_date"].to_numpy(dtype="datetime64[ns]"))
        signal_at = np.searchsorted(open_dates, np.datetime64(date), side="right")
        active["listed_trading_days"] = signal_at - listed_at
        rows.append(active)
    return pd.concat(rows, ignore_index=True)


def _daily_factor_panel(paths: LocalSourcePaths, daily: pd.DataFrame) -> pd.DataFrame:
    calendar = storage_io.read_frame(paths.equity_root / "calendar.parquet")
    dates = pd.DatetimeIndex(calendar["trade_date"])
    dates = dates[(dates >= "2014-01-01") & (dates <= "2025-12-31")]
    signal_dates = pd.DatetimeIndex(pd.Series(dates).groupby(dates.to_period("M")).max())
    members = _signal_members(paths, signal_dates)

    indexed = daily.set_index(["symbol", "trade_date"]).sort_index()
    close = indexed["adj_close"]
    stock_return = close.groupby(level=0, observed=True).pct_change(fill_method=None)
    turnover = indexed["turnover"]
    features: dict[str, pd.Series] = {}
    grouped_close = close.groupby(level=0, observed=True)
    features["REVERSAL_5D"] = -(close / grouped_close.shift(5) - 1)
    features["REVERSAL_20D"] = -(close / grouped_close.shift(20) - 1)
    features["MOMENTUM_60D"] = close / grouped_close.shift(60) - 1
    features["MOMENTUM_120D"] = close / grouped_close.shift(120) - 1
    features["MOMENTUM_120D_SKIP20"] = grouped_close.shift(20) / grouped_close.shift(140) - 1
    features["MOMENTUM_252D_SKIP20"] = grouped_close.shift(20) / grouped_close.shift(272) - 1
    features["HIGH_252D_PROXIMITY"] = close / _rolling(close, 252, 200, "max") - 1
    for window in (20, 60, 120):
        features[f"TURNOVER_MEAN_{window}D"] = _rolling(turnover, window, int(window * 0.8), "mean")
    long_turnover = _rolling(turnover, 480, 384, "mean")
    for window in (20, 60):
        features[f"TURNOVER_BIAS_{window}_480D"] = _rolling(
            turnover, window, int(window * 0.8), "mean"
        ) / long_turnover - 1
        features[f"TURNOVER_VOL_{window}D"] = _rolling(turnover, window, int(window * 0.8), "std")
    amihud = stock_return.abs() / indexed["amount"].replace(0, np.nan)
    features["AMIHUD_20D"] = _rolling(amihud, 20, 15, "mean") * 1e8
    for window in (20, 60, 120):
        features[f"VOLATILITY_{window}D"] = _rolling(stock_return, window, int(window * 0.8), "std")
    features["UPSIDE_VOL_60D"] = _rolling(stock_return.clip(lower=0), 60, 48, "std")
    features["DOWNSIDE_VOL_60D"] = _rolling(stock_return.clip(upper=0), 60, 48, "std")
    features["MAX_RETURN_20D"] = _rolling(stock_return, 20, 16, "max")

    signal_index = pd.MultiIndex.from_frame(members[["symbol", "signal_date"]])
    wide = pd.DataFrame({name: value.reindex(signal_index).to_numpy() for name, value in features.items()})
    wide = pd.concat([members.reset_index(drop=True), wide], axis=1)
    market = _market_returns(paths)
    wide["IDIO_VOL_60D"] = _idio_at_signals(daily, wide, market, "vol")
    wide["IDIO_SKEW_60D"] = _idio_at_signals(daily, wide, market, "skew")
    return wide


def _short_horizon_wide(paths: LocalSourcePaths, daily: pd.DataFrame) -> pd.DataFrame:
    """Compute registered short-window variants at monthly signal dates."""
    calendar = storage_io.read_frame(paths.equity_root / "calendar.parquet")
    dates = pd.DatetimeIndex(calendar["trade_date"])
    dates = dates[(dates >= "2014-01-01") & (dates <= "2025-12-31")]
    signal_dates = pd.DatetimeIndex(pd.Series(dates).groupby(dates.to_period("M")).max())
    members = _signal_members(paths, signal_dates)

    indexed = daily.set_index(["symbol", "trade_date"]).sort_index()
    close = indexed["adj_close"]
    grouped_close = close.groupby(level=0, observed=True)
    stock_return = grouped_close.pct_change(fill_method=None)
    turnover = indexed["turnover"]
    amihud = stock_return.abs() / indexed["amount"].replace(0, np.nan)
    features: dict[str, pd.Series] = {}

    for window in (1, 2, 3, 5, 7, 10, 20):
        features[f"REVERSAL_{window}D"] = -(close / grouped_close.shift(window) - 1)
    for window, minimum in ((20, 16), (60, 48), (120, 96), (252, 200)):
        features[f"HIGH_{window}D_PROXIMITY"] = close / _rolling(
            close, window, minimum, "max"
        ) - 1
    for window in (1, 5, 7, 10, 20):
        features[f"TURNOVER_MEAN_{window}D"] = _rolling(
            turnover, window, max(1, int(window * 0.8)), "mean"
        )
    for window in (5, 7, 10, 20):
        features[f"TURNOVER_VOL_{window}D"] = _rolling(
            turnover, window, max(3, int(window * 0.8)), "std"
        )
    for window, minimum in ((5, 4), (7, 5), (10, 8), (20, 15)):
        features[f"AMIHUD_{window}D"] = _rolling(amihud, window, minimum, "mean") * 1e8

    features["ABS_RETURN_1D"] = stock_return.abs()
    for window in (2, 3, 5, 7, 10, 20):
        features[f"VOLATILITY_{window}D"] = _rolling(
            stock_return, window, max(2, int(window * 0.8)), "std"
        )
    for window in (10, 20, 60):
        minimum = int(window * 0.8)
        features[f"UPSIDE_VOL_{window}D"] = _rolling(
            stock_return.clip(lower=0), window, minimum, "std"
        )
        features[f"DOWNSIDE_VOL_{window}D"] = _rolling(
            stock_return.clip(upper=0), window, minimum, "std"
        )
    for window in (5, 7, 10, 20):
        features[f"MAX_RETURN_{window}D"] = _rolling(
            stock_return, window, max(3, int(window * 0.8)), "max"
        )

    signal_index = pd.MultiIndex.from_frame(members[["symbol", "signal_date"]])
    values = {
        name: value.reindex(signal_index).to_numpy()
        for name, value in features.items()
    }
    return pd.concat([members.reset_index(drop=True), pd.DataFrame(values)], axis=1)


def _fundamental_features(fundamentals: pd.DataFrame) -> pd.DataFrame:
    frame = fundamentals.copy()
    frame["stat_date"] = pd.to_datetime(frame["stat_date"])
    frame["available_date"] = pd.to_datetime(frame["available_date"])
    frame = frame.sort_values(["symbol", "stat_date", "available_date"])
    frame = frame.drop_duplicates(["symbol", "stat_date"], keep="last")
    group = frame.groupby("symbol", observed=True)
    for field in ("eps", "ocfps", "roe", "gross_margin"):
        prior = group[field].shift(4)
        if field in ("roe", "gross_margin"):
            frame[f"{field}_yoy_change"] = frame[field] - prior
        else:
            frame[f"{field}_yoy"] = frame[field] / prior.abs().replace(0, np.nan) - 1
    for field in ("eps", "ocfps", "net_profit", "revenue"):
        rolling = group[field].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
        frame[f"{field}_ttm"] = rolling
    frame["net_margin_ttm"] = frame["net_profit_ttm"] / frame["revenue_ttm"].replace(0, np.nan)
    frame["ocf_to_earnings"] = frame["ocfps_ttm"] / frame["eps_ttm"].replace(0, np.nan)
    frame["roe_stability"] = -group["roe"].rolling(8, min_periods=6).std().reset_index(level=0, drop=True)
    frame["gross_margin_stability"] = -group["gross_margin"].rolling(8, min_periods=6).std().reset_index(level=0, drop=True)
    growth = group["net_profit"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    frame["earnings_growth_stability"] = -_rolling(
        growth.set_axis(pd.MultiIndex.from_frame(frame[["symbol", "stat_date"]])), 8, 6, "std"
    ).to_numpy()
    return frame


def _add_fundamental_factors(paths: LocalSourcePaths, wide: pd.DataFrame) -> pd.DataFrame:
    fundamentals = _fundamental_features(storage_io.read_frame(paths.equity_root / "fundamentals.parquet"))
    rows: list[pd.DataFrame] = []
    for date, members in wide.groupby("signal_date", observed=True):
        visible = fundamentals[fundamentals["available_date"] <= date]
        latest = visible.sort_values(["symbol", "stat_date", "available_date"]).groupby(
            "symbol", observed=True
        ).tail(1)
        latest = latest.set_index("symbol")
        part = members.copy().set_index("symbol")
        mappings = {
            "EP_TTM": "eps_ttm",
            "BP_MRQ": "bps",
            "CFP_TTM": "ocfps_ttm",
            "REVENUE_YOY": "rev_yoy",
            "NET_PROFIT_YOY": "np_yoy",
            "EPS_YOY": "eps_yoy",
            "OCFPS_YOY": "ocfps_yoy",
            "ROE_YOY_CHANGE": "roe_yoy_change",
            "GROSS_MARGIN_YOY_CHANGE": "gross_margin_yoy_change",
            "ROE": "roe",
            "GROSS_MARGIN": "gross_margin",
            "NET_MARGIN_TTM": "net_margin_ttm",
            "OCF_TO_EARNINGS": "ocf_to_earnings",
            "ROE_STABILITY_8Q": "roe_stability",
            "GROSS_MARGIN_STABILITY_8Q": "gross_margin_stability",
            "EARNINGS_GROWTH_STABILITY_8Q": "earnings_growth_stability",
        }
        price = part["raw_close"]
        for factor, field in mappings.items():
            values = latest[field].reindex(part.index)
            part[factor] = values / price if factor in {"EP_TTM", "BP_MRQ", "CFP_TTM"} else values
        rows.append(part.reset_index())
    return pd.concat(rows, ignore_index=True)


def _attach_labels_and_eligibility(
    paths: LocalSourcePaths, daily: pd.DataFrame, wide: pd.DataFrame
) -> pd.DataFrame:
    """Attach next-open labels and enforce the same v0 eligibility rules."""
    daily_keyed = daily.set_index(["symbol", "trade_date"])
    signal_key = pd.MultiIndex.from_frame(wide[["symbol", "signal_date"]])
    wide["raw_close"] = daily_keyed["close"].reindex(signal_key).to_numpy()

    calendar = pd.DatetimeIndex(storage_io.read_frame(paths.equity_root / "calendar.parquet")["trade_date"])
    execution_dates = calendar[calendar <= "2026-01-31"]
    execution_signals = sorted(
        pd.Series(execution_dates).groupby(execution_dates.to_period("M")).max().tolist()
    )
    entry_by_signal: dict[pd.Timestamp, pd.Timestamp] = {}
    for signal in execution_signals:
        future = calendar[calendar > signal]
        if len(future):
            entry_by_signal[pd.Timestamp(signal)] = pd.Timestamp(future[0])
    next_signal = {
        pd.Timestamp(left): pd.Timestamp(right) for left, right in pairwise(execution_signals)
    }
    wide["entry_date"] = wide["signal_date"].map(entry_by_signal)
    wide["exit_date"] = wide["signal_date"].map(next_signal).map(entry_by_signal)
    entry_key = pd.MultiIndex.from_frame(
        wide[["symbol", "entry_date"]].rename(columns={"entry_date": "trade_date"})
    )
    exit_key = pd.MultiIndex.from_frame(
        wide[["symbol", "exit_date"]].rename(columns={"exit_date": "trade_date"})
    )
    wide["entry_adj_open"] = daily_keyed["adj_open"].reindex(entry_key).to_numpy()
    wide["exit_adj_open"] = daily_keyed["adj_open"].reindex(exit_key).to_numpy()
    wide["forward_return"] = wide["exit_adj_open"] / wide["entry_adj_open"] - 1
    wide = wide[wide["listed_trading_days"] >= 250].copy()

    status = storage_io.read_frame(paths.equity_root / "status.parquet")
    status = status.rename(columns={"trade_date": "signal_date"})
    wide = wide.merge(
        status[["signal_date", "symbol", "is_st", "is_suspended"]],
        on=["signal_date", "symbol"],
        how="left",
    )
    known_status = wide["is_st"].notna() & wide["is_suspended"].notna()
    return wide[
        known_status & ~wide["is_st"].astype(bool) & ~wide["is_suspended"].astype(bool)
    ].copy()


def _melt_factor_panel(wide: pd.DataFrame, factor_names: tuple[str, ...]) -> pd.DataFrame:
    long = wide.melt(
        id_vars=[
            "signal_date",
            "symbol",
            "index_code",
            "forward_return",
            "entry_date",
            "exit_date",
        ],
        value_vars=list(factor_names),
        var_name="factor_name",
        value_name="raw_value",
    )
    long["raw_value"] = long.groupby(
        ["signal_date", "index_code", "factor_name"], observed=True
    )["raw_value"].transform(lambda values: zscore(mad_winsorize(values)))
    long["neutralized_value"] = np.nan
    long["available_date"] = long["signal_date"]
    long["period"] = assign_period(long["signal_date"])
    return long


def build_v0_factor_panel(paths: LocalSourcePaths) -> pd.DataFrame:
    """Build 39-factor monthly panel for HS300/CSI500 with open-to-open labels."""
    members = storage_io.read_frame(paths.equity_root / "index_members.parquet")
    symbols = set(members["symbol"])
    daily = _load_filtered_daily(paths, symbols)
    wide = _attach_labels_and_eligibility(paths, daily, _daily_factor_panel(paths, daily))
    wide = _add_fundamental_factors(paths, wide)

    long = _melt_factor_panel(wide, PRICE_VOLUME_FACTORS + FUNDAMENTAL_FACTORS)
    output = paths.artifact_root / "factor_panel_39.parquet"
    storage_io.write_frame(long, output, index=False)
    return long


def build_short_horizon_panel(paths: LocalSourcePaths) -> pd.DataFrame:
    """Build the pre-registered 1D--252D parameter-sensitivity panel."""
    members = storage_io.read_frame(paths.equity_root / "index_members.parquet")
    symbols = set(members["symbol"])
    daily = _load_filtered_daily(paths, symbols)
    wide = _attach_labels_and_eligibility(paths, daily, _short_horizon_wide(paths, daily))
    long = _melt_factor_panel(wide, SHORT_HORIZON_FACTORS)
    storage_io.write_frame(long, paths.artifact_root / "short_horizon_panel.parquet", index=False)
    return long


def _technical_wide(paths: LocalSourcePaths, daily: pd.DataFrame) -> pd.DataFrame:
    calendar = storage_io.read_frame(paths.equity_root / "calendar.parquet")
    dates = pd.DatetimeIndex(calendar["trade_date"])
    dates = dates[(dates >= "2014-01-01") & (dates <= "2025-12-31")]
    signal_dates = pd.DatetimeIndex(pd.Series(dates).groupby(dates.to_period("M")).max())
    members = _signal_members(paths, signal_dates)
    signal_index = pd.MultiIndex.from_frame(members[["symbol", "signal_date"]])
    values = compute_technical_features(daily, signal_index)
    return pd.concat([members.reset_index(drop=True), values.reset_index(drop=True)], axis=1)


def _write_technical_catalog(paths: LocalSourcePaths) -> None:
    rows: list[dict[str, object]] = []
    for name, (hypothesis, inputs, lookback, minimum) in TECHNICAL_METADATA.items():
        rows.append(
            {
                "source_name": name,
                "canonical_name": name,
                "status": "implemented",
                "hypothesis_id": hypothesis,
                "input_fields": ",".join(inputs),
                "lookback_days": lookback,
                "min_observations": minimum,
                "note": "backward-adjusted; raw price-level output normalized",
            }
        )
    for source_name, canonical_name in SOURCE_ALIASES.items():
        rows.append(
            {
                "source_name": source_name,
                "canonical_name": canonical_name,
                "status": "exact_alias",
                "hypothesis_id": TECHNICAL_METADATA[canonical_name][0],
                "input_fields": "",
                "lookback_days": TECHNICAL_METADATA[canonical_name][2],
                "min_observations": TECHNICAL_METADATA[canonical_name][3],
                "note": "not tested as an independent hypothesis",
            }
        )
    for source_name, note in BLOCKED_SOURCE_FACTORS.items():
        rows.append(
            {
                "source_name": source_name,
                "canonical_name": "",
                "status": "blocked",
                "hypothesis_id": "technical_xsii",
                "input_fields": "",
                "lookback_days": np.nan,
                "min_observations": np.nan,
                "note": note,
            }
        )
    storage_io.write_csv(pd.DataFrame(rows), paths.artifact_root / "technical_factor_catalog.csv", index=False)


def build_technical_factor_panel(paths: LocalSourcePaths) -> pd.DataFrame:
    """Build all independently calculable price-volume factors from the workbook."""
    members = storage_io.read_frame(paths.equity_root / "index_members.parquet")
    symbols = set(members["symbol"])
    daily = _load_filtered_daily(paths, symbols)
    wide = _attach_labels_and_eligibility(paths, daily, _technical_wide(paths, daily))
    long = _melt_factor_panel(wide, TECHNICAL_FACTOR_NAMES)
    storage_io.write_frame(long, paths.artifact_root / "technical_factor_panel.parquet", index=False)
    _write_technical_catalog(paths)
    return long


def _evaluate_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    for index_code, part in panel.groupby("index_code", observed=True):
        monthly, summary = evaluate_factor_batch(part)
        monthly["index_code"] = index_code
        summary["index_code"] = index_code
        monthly_parts.append(monthly)
        summary_parts.append(summary)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    summary = pd.concat(summary_parts, ignore_index=True)
    # One correction family across both universes, all concrete factors, directions and periods.
    summary["bh_q_value"] = benjamini_hochberg(summary["p_value"])
    # Model selection must not let validation/test results alter development q-values.
    # Reverse orientation is an exact mirror, not an independent concrete parameter.
    summary["development_selection_q_value"] = np.nan
    development = (
        (summary["period"] == "development")
        & (summary["value_type"] == "raw")
        & (summary["orientation"] == "original")
    )
    summary.loc[development, "development_selection_q_value"] = benjamini_hochberg(
        summary.loc[development, "p_value"]
    )
    return monthly, summary


def evaluate_v0(
    paths: LocalSourcePaths, panel: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate both indices separately and write BH-adjusted single-factor results."""
    if panel is None:
        panel = storage_io.read_frame(paths.artifact_root / "factor_panel_39.parquet")
    monthly, summary = _evaluate_panel(panel)
    paths.artifact_root.mkdir(parents=True, exist_ok=True)
    storage_io.write_frame(monthly, paths.artifact_root / "single_factor_monthly.parquet", index=False)
    storage_io.write_csv(summary, paths.artifact_root / "single_factor_summary.csv", index=False)
    return monthly, summary


def evaluate_short_horizon(
    paths: LocalSourcePaths, panel: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the short-window batch with a single batch-wide BH correction."""
    if panel is None:
        panel = storage_io.read_frame(paths.artifact_root / "short_horizon_panel.parquet")
    monthly, summary = _evaluate_panel(panel)
    storage_io.write_frame(monthly, paths.artifact_root / "short_horizon_monthly.parquet", index=False)
    storage_io.write_csv(summary, paths.artifact_root / "short_horizon_summary.csv", index=False)
    return monthly, summary


def build_expanded_factor_panel(paths: LocalSourcePaths) -> pd.DataFrame:
    """Merge the baseline and short-window panels without duplicating shared factors."""
    baseline = storage_io.read_frame(paths.artifact_root / "factor_panel_39.parquet")
    short = storage_io.read_frame(paths.artifact_root / "short_horizon_panel.parquet")
    baseline_names = set(baseline["factor_name"].unique())
    short = short[~short["factor_name"].isin(baseline_names)]
    expanded = pd.concat([baseline, short], ignore_index=True)
    storage_io.write_frame(expanded, paths.artifact_root / "expanded_factor_panel.parquet", index=False)
    return expanded


def evaluate_expanded_factors(
    paths: LocalSourcePaths, panel: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-evaluate all available factors as one multiple-testing family."""
    if panel is None:
        panel = storage_io.read_frame(paths.artifact_root / "expanded_factor_panel.parquet")
    monthly, summary = _evaluate_panel(panel)
    storage_io.write_frame(monthly, paths.artifact_root / "expanded_factor_monthly.parquet", index=False)
    storage_io.write_csv(summary, paths.artifact_root / "expanded_factor_summary.csv", index=False)
    return monthly, summary


def evaluate_technical_factors(
    paths: LocalSourcePaths, panel: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate workbook technical factors as one multiple-testing family."""
    if panel is None:
        panel = storage_io.read_frame(paths.artifact_root / "technical_factor_panel.parquet")
    monthly, summary = _evaluate_panel(panel)
    storage_io.write_frame(monthly, paths.artifact_root / "technical_factor_monthly.parquet", index=False)
    storage_io.write_csv(summary, paths.artifact_root / "technical_factor_summary.csv", index=False)
    return monthly, summary


def build_full_factor_panel(paths: LocalSourcePaths) -> pd.DataFrame:
    """Merge classic, short-window and workbook factors without duplicate names."""
    expanded = storage_io.read_frame(paths.artifact_root / "expanded_factor_panel.parquet")
    technical = storage_io.read_frame(paths.artifact_root / "technical_factor_panel.parquet")
    known = set(expanded["factor_name"].unique())
    technical = technical[~technical["factor_name"].isin(known)]
    full = pd.concat([expanded, technical], ignore_index=True)
    storage_io.write_frame(full, paths.artifact_root / "full_factor_panel.parquet", index=False)
    return full


def evaluate_full_factors(
    paths: LocalSourcePaths, panel: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the complete available pool under one batch-wide BH correction."""
    if panel is None:
        panel = storage_io.read_frame(paths.artifact_root / "full_factor_panel.parquet")
    monthly, summary = _evaluate_panel(panel)
    storage_io.write_frame(monthly, paths.artifact_root / "full_factor_monthly.parquet", index=False)
    storage_io.write_csv(summary, paths.artifact_root / "full_factor_summary.csv", index=False)
    return monthly, summary


def _trading_costs(date: pd.Timestamp, buys: float, sells: float) -> tuple[float, float]:
    commission = 0.0003
    slippage = 0.0005
    stress_slippage = 0.0010
    stamp = 0.0005 if date >= pd.Timestamp("2023-08-28") else 0.001
    transfer = 0.00001 if date >= pd.Timestamp("2022-04-29") else 0.00002
    regular = (buys + sells) * (commission + slippage + transfer) + sells * stamp
    stress = (buys + sells) * (commission + stress_slippage + transfer) + sells * stamp
    return regular, stress


def _performance_metrics(returns: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return {}
    count = len(values)
    annual_return = float((1 + values).prod() ** (12 / count) - 1)
    annual_volatility = float(values.std(ddof=1) * np.sqrt(12))
    downside = values[values < 0].std(ddof=1) * np.sqrt(12)
    wealth = (1 + values).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    maximum_drawdown = float(drawdown.min())
    quarterly = (1 + values).groupby(values.index.to_period("Q")).prod() - 1
    yearly = (1 + values).groupby(values.index.to_period("Y")).prod() - 1
    positive = values.clip(lower=0).sum()
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": float(values.mean() * 12 / annual_volatility) if annual_volatility else np.nan,
        "sortino": float(values.mean() * 12 / downside) if downside else np.nan,
        "max_drawdown": maximum_drawdown,
        "calmar": annual_return / abs(maximum_drawdown) if maximum_drawdown else np.nan,
        "monthly_win_rate": float((values > 0).mean()),
        "quarterly_win_rate": float((quarterly > 0).mean()),
        "yearly_win_rate": float((yearly > 0).mean()),
        "best_5_month_concentration": float(values.nlargest(5).clip(lower=0).sum() / positive)
        if positive
        else np.nan,
    }


def _benchmark_forward_returns(paths: LocalSourcePaths, panel: pd.DataFrame) -> pd.Series:
    index_daily = storage_io.read_frame(paths.meta_root / "index_daily.parquet")
    index_daily = index_daily[index_daily["code"].isin(["sh.000300", "sh.000905"])].copy()
    index_daily["index_code"] = index_daily["code"].map(
        {"sh.000300": "000300.SH", "sh.000905": "000905.SH"}
    )
    index_daily["trade_date"] = pd.to_datetime(index_daily["date"])
    index_daily["open"] = pd.to_numeric(index_daily["open"], errors="coerce")
    opens = index_daily.set_index(["index_code", "trade_date"])["open"]
    dates = panel[["index_code", "signal_date", "entry_date", "exit_date"]].drop_duplicates()
    entry_key = pd.MultiIndex.from_frame(
        dates[["index_code", "entry_date"]].rename(columns={"entry_date": "trade_date"})
    )
    exit_key = pd.MultiIndex.from_frame(
        dates[["index_code", "exit_date"]].rename(columns={"exit_date": "trade_date"})
    )
    dates["benchmark_return"] = (
        opens.reindex(exit_key).to_numpy() / opens.reindex(entry_key).to_numpy() - 1
    )
    return dates.set_index(["index_code", "signal_date"])["benchmark_return"]


def build_v0_portfolios(
    paths: LocalSourcePaths,
    panel: pd.DataFrame | None = None,
    *,
    output_prefix: str = "single_factor",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build direction-aligned equal-weight quintiles and costed top-layer diagnostics."""
    if panel is None:
        panel = storage_io.read_frame(paths.artifact_root / "factor_panel_39.parquet")
    direction = {spec.name: spec.expected_direction for spec in REGISTRY.list()}
    benchmark = _benchmark_forward_returns(paths, panel)
    detail_rows: list[dict[str, object]] = []
    for (index_code, factor_name), factor in panel.groupby(
        ["index_code", "factor_name"], observed=True
    ):
        sign = -1.0 if direction[factor_name] == "negative" else 1.0
        previous: dict[str, float] = {}
        for date, cross_section in factor.groupby("signal_date", observed=True):
            sample = cross_section[["symbol", "raw_value", "forward_return"]].dropna()
            if len(sample) < 25:
                continue
            sample = sample.assign(score=sample["raw_value"] * sign)
            sample["layer"] = 5 - pd.qcut(
                sample["score"].rank(method="first"), 5, labels=False
            )
            group_returns = sample.groupby("layer", observed=True)["forward_return"].mean()
            top = sample[sample["layer"] == 1]
            current = dict.fromkeys(top["symbol"], 1.0 / len(top))
            names = set(previous) | set(current)
            buys = sum(max(current.get(name, 0.0) - previous.get(name, 0.0), 0.0) for name in names)
            sells = sum(max(previous.get(name, 0.0) - current.get(name, 0.0), 0.0) for name in names)
            cost, stress_cost = _trading_costs(pd.Timestamp(date), buys, sells)
            row: dict[str, object] = {
                "signal_date": date,
                "index_code": index_code,
                "factor_name": factor_name,
                "constituents": len(sample),
                "turnover": 0.5 * (buys + sells),
                "buy_turnover": buys,
                "sell_turnover": sells,
                "cost": cost,
                "stress_cost": stress_cost,
                "benchmark_return": benchmark.get((index_code, pd.Timestamp(date)), np.nan),
            }
            row.update({f"group_{layer}": group_returns.get(layer, np.nan) for layer in range(1, 6)})
            row["long_short"] = row["group_1"] - row["group_5"]
            row["top_net_return"] = row["group_1"] - cost
            row["top_stress_return"] = row["group_1"] - stress_cost
            row["top_net_excess_return"] = row["top_net_return"] - row["benchmark_return"]
            detail_rows.append(row)
            previous = current
    detail = pd.DataFrame(detail_rows).sort_values(["index_code", "factor_name", "signal_date"])
    detail["period"] = assign_period(detail["signal_date"])
    metric_rows: list[dict[str, object]] = []
    for (index_code, factor_name), group in detail.groupby(
        ["index_code", "factor_name"], observed=True
    ):
        slices = [("all", group)] + list(group.groupby("period", observed=True))
        for period, sample in slices:
            indexed = sample.set_index("signal_date")["top_net_return"]
            metrics = _performance_metrics(indexed)
            excess = sample.set_index("signal_date")["top_net_excess_return"]
            excess_volatility = excess.std(ddof=1) * np.sqrt(12)
            metric_rows.append(
                {
                    "index_code": index_code,
                    "factor_name": factor_name,
                    "period": period,
                    "months": len(sample),
                    "mean_turnover": sample["turnover"].mean(),
                    "mean_gross_return": sample["group_1"].mean(),
                    "mean_net_return": sample["top_net_return"].mean(),
                    "mean_stress_return": sample["top_stress_return"].mean(),
                    "mean_long_short": sample["long_short"].mean(),
                    "mean_net_excess_return": excess.mean(),
                    "net_information_ratio": excess.mean() * 12 / excess_volatility
                    if excess_volatility
                    else np.nan,
                    **metrics,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    storage_io.write_frame(detail, paths.artifact_root / f"{output_prefix}_portfolios.parquet", index=False)
    storage_io.write_csv(metrics, paths.artifact_root / f"{output_prefix}_portfolio_metrics.csv", index=False)
    return detail, metrics


def _selected_development_factors(
    paths: LocalSourcePaths,
    index_code: str,
    *,
    summary: pd.DataFrame | None = None,
    collapse_hypotheses: bool = False,
    require_bh_significance: bool = True,
) -> list[str]:
    if summary is None:
        summary = storage_io.read_csv(paths.artifact_root / "single_factor_summary.csv")
    sample = summary[
        (summary["index_code"] == index_code)
        & (summary["period"] == "development")
        & (summary["value_type"] == "raw")
        & (summary["orientation"] == "original")
    ].copy()
    signs = _locked_direction_signs(summary, index_code)
    sample["aligned_ic"] = sample["rank_ic"] * sample["factor_name"].map(signs)
    eligible = (sample["aligned_ic"] > 0) & (sample["coverage"] >= 0.50)
    if require_bh_significance:
        eligible &= (
            sample.get("development_selection_q_value", sample["bh_q_value"]) < 0.10
        )
    selected = sample[eligible].copy()
    if collapse_hypotheses:
        hypothesis = {spec.name: spec.hypothesis_id for spec in REGISTRY.list()}
        selected["hypothesis_id"] = selected["factor_name"].map(hypothesis)
        selected = selected.sort_values(
            ["aligned_ic", "coverage", "factor_name"],
            ascending=[False, False, True],
        ).drop_duplicates("hypothesis_id", keep="first")
    return sorted(selected["factor_name"].tolist())


def _locked_direction_signs(
    summary: pd.DataFrame, index_code: str
) -> dict[str, float]:
    """Lock unknown directions from development IC only, independently by universe."""
    registered = {spec.name: spec.expected_direction for spec in REGISTRY.list()}
    development = summary[
        (summary["index_code"] == index_code)
        & (summary["period"] == "development")
        & (summary["value_type"] == "raw")
        & (summary["orientation"] == "original")
    ].set_index("factor_name")["rank_ic"]
    signs: dict[str, float] = {}
    for name in summary.loc[summary["index_code"] == index_code, "factor_name"].unique():
        direction = registered[name]
        if direction == "negative":
            signs[name] = -1.0
        elif direction == "positive":
            signs[name] = 1.0
        else:
            value = development.get(name, np.nan)
            signs[name] = -1.0 if np.isfinite(value) and value < 0 else 1.0
    return signs


def _method_weight(
    method: str,
    ic_history: pd.DataFrame,
    return_history: pd.DataFrame,
    factors: list[str],
) -> pd.Series:
    if method == "equal" or len(ic_history) < 2:
        return pd.Series(1 / len(factors), index=factors, name=method)
    history = return_history if method == "factor_return_decay" else ic_history
    history = history.reindex(columns=factors)
    return factor_weights(history, method).reindex(factors).fillna(0)


def build_v0_combinations(
    paths: LocalSourcePaths,
    panel: pd.DataFrame | None = None,
    *,
    factor_monthly: pd.DataFrame | None = None,
    factor_summary: pd.DataFrame | None = None,
    output_prefix: str = "multi_factor",
    collapse_hypotheses: bool = False,
    require_bh_significance: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Compare six development-selected combinations; freeze weights before test."""
    if panel is None:
        panel = storage_io.read_frame(paths.artifact_root / "factor_panel_39.parquet")
    if factor_monthly is None:
        factor_monthly = storage_io.read_frame(
            paths.artifact_root / "single_factor_monthly.parquet"
        )
    if factor_summary is None:
        factor_summary = storage_io.read_csv(paths.artifact_root / "single_factor_summary.csv")
    factor_monthly = factor_monthly[
        (factor_monthly["value_type"] == "raw")
        & (factor_monthly["orientation"] == "original")
    ].copy()
    benchmark = _benchmark_forward_returns(paths, panel)
    portfolio_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    selected_by_index: dict[str, list[str]] = {}
    locked_signs_by_index: dict[str, dict[str, float]] = {}
    for index_code, index_panel in panel.groupby("index_code", observed=True):
        sign = _locked_direction_signs(factor_summary, index_code)
        locked_signs_by_index[index_code] = sign
        factors = _selected_development_factors(
            paths,
            index_code,
            summary=factor_summary,
            collapse_hypotheses=collapse_hypotheses,
            require_bh_significance=require_bh_significance,
        )
        if not factors:
            raise ValueError(f"no development factors selected for {index_code}")
        selected_by_index[index_code] = factors
        history = factor_monthly[factor_monthly["index_code"] == index_code].copy()
        history["aligned_ic"] = history["rank_ic"] * history["factor_name"].map(sign)
        history["aligned_factor_return"] = (
            history["factor_return"] * history["factor_name"].map(sign)
        )
        ic = history.pivot(index="signal_date", columns="factor_name", values="aligned_ic")
        returns = history.pivot(
            index="signal_date", columns="factor_name", values="aligned_factor_return"
        )
        scores = index_panel[index_panel["factor_name"].isin(factors)].copy()
        scores["aligned_value"] = scores["raw_value"] * scores["factor_name"].map(sign)
        frozen: dict[str, pd.Series] = {}
        previous: dict[str, dict[str, float]] = {method: {} for method in METHODS}
        for date, cross_section in scores.groupby("signal_date", observed=True):
            matrix = cross_section.pivot(index="symbol", columns="factor_name", values="aligned_value")
            stock_returns = cross_section.drop_duplicates("symbol").set_index("symbol")["forward_return"]
            historical_ic = ic[ic.index < date].tail(12)
            historical_return = returns[returns.index < date].tail(12)
            for method in METHODS:
                if date >= pd.Timestamp("2024-01-01") and method in frozen:
                    weights = frozen[method]
                else:
                    weights = _method_weight(method, historical_ic, historical_return, factors)
                    if date >= pd.Timestamp("2023-12-01"):
                        frozen[method] = weights
                for factor_name, value in weights.items():
                    weight_rows.append(
                        {
                            "signal_date": date,
                            "index_code": index_code,
                            "method": method,
                            "factor_name": factor_name,
                            "weight": value,
                            "frozen_for_test": date >= pd.Timestamp("2024-01-01"),
                        }
                    )
                combined = combine_scores(matrix, weights).dropna()
                if len(combined) < 25:
                    continue
                layer = 5 - pd.qcut(combined.rank(method="first"), 5, labels=False)
                top_names = layer[layer == 1].index
                current = dict.fromkeys(top_names, 1.0 / len(top_names))
                prior = previous[method]
                names = set(prior) | set(current)
                buys = sum(max(current.get(name, 0.0) - prior.get(name, 0.0), 0.0) for name in names)
                sells = sum(max(prior.get(name, 0.0) - current.get(name, 0.0), 0.0) for name in names)
                cost, stress_cost = _trading_costs(pd.Timestamp(date), buys, sells)
                group_return = stock_returns.groupby(layer, observed=True).mean()
                gross = group_return.get(1, np.nan)
                if not np.isfinite(gross):
                    continue
                benchmark_return = benchmark.get((index_code, pd.Timestamp(date)), np.nan)
                portfolio_rows.append(
                    {
                        "signal_date": date,
                        "index_code": index_code,
                        "method": method,
                        "selected_factor_count": len(factors),
                        "group_1": gross,
                        "group_5": group_return.get(5, np.nan),
                        "long_short": gross - group_return.get(5, np.nan),
                        "turnover": 0.5 * (buys + sells),
                        "cost": cost,
                        "stress_cost": stress_cost,
                        "benchmark_return": benchmark_return,
                        "net_return": gross - cost,
                        "stress_return": gross - stress_cost,
                        "net_excess_return": gross - cost - benchmark_return,
                    }
                )
                previous[method] = current
    portfolios = pd.DataFrame(portfolio_rows)
    portfolios["period"] = assign_period(portfolios["signal_date"])
    metrics_rows: list[dict[str, object]] = []
    for (index_code, method), group in portfolios.groupby(["index_code", "method"], observed=True):
        slices = [("all", group)] + list(group.groupby("period", observed=True))
        for period, sample in slices:
            net = sample.set_index("signal_date")["net_return"]
            excess = sample.set_index("signal_date")["net_excess_return"]
            excess_volatility = excess.std(ddof=1) * np.sqrt(12)
            metrics_rows.append(
                {
                    "index_code": index_code,
                    "method": method,
                    "period": period,
                    "months": len(sample),
                    "mean_turnover": sample["turnover"].mean(),
                    "mean_long_short": sample["long_short"].mean(),
                    "mean_net_excess_return": excess.mean(),
                    "net_information_ratio": excess.mean() * 12 / excess_volatility
                    if excess_volatility
                    else np.nan,
                    **_performance_metrics(net),
                }
            )
    metrics = pd.DataFrame(metrics_rows)
    selection: dict[str, str] = {}
    for index_code, sample in metrics[
        (metrics["period"] == "validation")
    ].groupby("index_code", observed=True):
        # December 2023's forward return enters 2024, so exclude it from method selection.
        realized = portfolios[
            (portfolios["index_code"] == index_code)
            & portfolios["signal_date"].between("2021-01-01", "2023-11-30")
        ]
        candidates: list[dict[str, object]] = []
        for method, method_returns in realized.groupby("method", observed=True):
            excess = method_returns["net_excess_return"].dropna()
            volatility = excess.std(ddof=1) * np.sqrt(12)
            candidates.append(
                {
                    "method": method,
                    "net_information_ratio": excess.mean() * 12 / volatility
                    if volatility
                    else np.nan,
                    "turnover": method_returns["turnover"].mean(),
                }
            )
        selection[index_code] = select_validation_method(pd.DataFrame(candidates))
    storage_io.write_frame(portfolios,
        paths.artifact_root / f"{output_prefix}_portfolios.parquet", index=False
    )
    storage_io.write_csv(metrics, paths.artifact_root / f"{output_prefix}_metrics.csv", index=False)
    storage_io.write_frame(pd.DataFrame(weight_rows),
        paths.artifact_root / f"{output_prefix}_weights.parquet", index=False
    )
    storage_io.write_text(paths.artifact_root / f"{output_prefix}_selection.json",
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    storage_io.write_text(paths.artifact_root / f"{output_prefix}_selected_factors.json",
        json.dumps(selected_by_index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    storage_io.write_text(paths.artifact_root / f"{output_prefix}_locked_signs.json",
        json.dumps(locked_signs_by_index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    storage_io.write_text(paths.artifact_root / f"{output_prefix}_selection_policy.json",
        json.dumps(
            {
                "development_only": True,
                "minimum_coverage": 0.50,
                "require_development_bh_q_below_0_10": require_bh_significance,
                "one_concrete_variant_per_hypothesis": collapse_hypotheses,
                "exploratory_weak_signal_combination": not require_bh_significance,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return portfolios, metrics, selection


def build_expanded_combinations(
    paths: LocalSourcePaths, panel: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Combine baseline plus short-window factors, one variant per hypothesis."""
    if panel is None:
        panel = storage_io.read_frame(paths.artifact_root / "expanded_factor_panel.parquet")
    monthly = storage_io.read_frame(paths.artifact_root / "expanded_factor_monthly.parquet")
    summary = storage_io.read_csv(paths.artifact_root / "expanded_factor_summary.csv")
    return build_v0_combinations(
        paths,
        panel,
        factor_monthly=monthly,
        factor_summary=summary,
        output_prefix="expanded_multi_factor",
        collapse_hypotheses=True,
    )


def build_expanded_baseline_combinations(
    paths: LocalSourcePaths,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Build a hypothesis-collapsed 39-factor comparator under the 70-factor q-family."""
    panel = storage_io.read_frame(paths.artifact_root / "factor_panel_39.parquet")
    names = set(panel["factor_name"].unique())
    monthly = storage_io.read_frame(paths.artifact_root / "expanded_factor_monthly.parquet")
    summary = storage_io.read_csv(paths.artifact_root / "expanded_factor_summary.csv")
    return build_v0_combinations(
        paths,
        panel,
        factor_monthly=monthly[monthly["factor_name"].isin(names)],
        factor_summary=summary[summary["factor_name"].isin(names)],
        output_prefix="expanded_baseline_multi_factor",
        collapse_hypotheses=True,
    )


def build_technical_combinations(
    paths: LocalSourcePaths, panel: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Combine only the workbook technical factors, one variant per hypothesis."""
    if panel is None:
        panel = storage_io.read_frame(paths.artifact_root / "technical_factor_panel.parquet")
    monthly = storage_io.read_frame(paths.artifact_root / "technical_factor_monthly.parquet")
    summary = storage_io.read_csv(paths.artifact_root / "technical_factor_summary.csv")
    return build_v0_combinations(
        paths,
        panel,
        factor_monthly=monthly,
        factor_summary=summary,
        output_prefix="technical_multi_factor",
        collapse_hypotheses=True,
        require_bh_significance=False,
    )


def build_full_combinations(
    paths: LocalSourcePaths, panel: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Combine the complete available pool, one concrete variant per hypothesis."""
    if panel is None:
        panel = storage_io.read_frame(paths.artifact_root / "full_factor_panel.parquet")
    monthly = storage_io.read_frame(paths.artifact_root / "full_factor_monthly.parquet")
    summary = storage_io.read_csv(paths.artifact_root / "full_factor_summary.csv")
    return build_v0_combinations(
        paths,
        panel,
        factor_monthly=monthly,
        factor_summary=summary,
        output_prefix="full_multi_factor",
        collapse_hypotheses=True,
        require_bh_significance=True,
    )


def build_v0_report(paths: LocalSourcePaths, report_path: Path) -> Path:
    """Render the concise Markdown report, two diagnostic charts and a hash manifest."""
    summary = storage_io.read_csv(paths.artifact_root / "single_factor_summary.csv")
    multi = storage_io.read_csv(paths.artifact_root / "multi_factor_metrics.csv")
    selection = json.loads(
        storage_io.read_text(paths.artifact_root / "multi_factor_selection.json", encoding="utf-8")
    )
    panel = storage_io.read_frame(
        paths.artifact_root / "factor_panel_39.parquet",
        columns=["signal_date", "symbol", "index_code", "factor_name"],
    )
    direction = {spec.name: spec.expected_direction for spec in REGISTRY.list()}
    expected = summary[
        (summary["value_type"] == "raw")
        & (summary["orientation"] == "original")
        & summary["period"].isin(["development", "validation", "test"])
    ].copy()
    expected["aligned_ic"] = expected.apply(
        lambda row: -row["rank_ic"]
        if direction[row["factor_name"]] == "negative"
        else row["rank_ic"],
        axis=1,
    )

    ic_chart = paths.artifact_root / "v0_ic_stability.html"
    options = []
    for index_code, sample in expected.groupby("index_code", observed=True):
        pivot = sample.pivot(index="factor_name", columns="period", values="aligned_ic")
        pivot["stable"] = pivot[["development", "validation"]].min(axis=1)
        pivot = pivot.nlargest(10, "stable")
        options.append({
            "title": {"text": str(index_code)}, "tooltip": {"trigger": "axis"},
            "legend": {"top": 30}, "grid": {"left": 180, "top": 80},
            "xAxis": {"type": "value", "name": "Direction-aligned Rank IC"},
            "yAxis": {"type": "category", "data": pivot.index.tolist(), "inverse": True},
            "series": [{"type": "bar", "name": period, "data": pivot[period].tolist()}
                       for period in ("development", "validation", "test")],
        })
    write_charts(ic_chart, "IC stability", options, smoke=True)
    multi_chart = paths.artifact_root / "v0_multi_factor_ir.html"
    chart_data = multi[multi["period"].isin(["validation", "test"])].pivot_table(
        index=["index_code", "method"], columns="period", values="net_information_ratio"
    )
    options = []
    for index_code in sorted(chart_data.index.get_level_values(0).unique()):
        sample = chart_data.loc[index_code]
        options.append({
            "title": {"text": str(index_code)}, "tooltip": {"trigger": "axis"},
            "legend": {"top": 30}, "grid": {"top": 80, "bottom": 100},
            "xAxis": {"type": "category", "data": sample.index.tolist(),
                      "axisLabel": {"rotate": 45}},
            "yAxis": {"type": "value", "name": "Net excess information ratio"},
            "series": [{"type": "bar", "name": period, "data": sample[period].tolist()}
                       for period in sample.columns],
        })
    write_charts(multi_chart, "Multi-factor IR", options, smoke=True)

    lines = [
        "# A股华泰式因子研究 v0（非正式）",
        "",
        "> **NON-FORMAL**：缺少历史申万一级行业、官方宽基权重和历史涨跌停状态；",
        "> 本报告只用于验证研究管线与提出下一轮假设，不能称为正式华泰框架复刻。",
        "",
        "## 数据与口径",
        "",
        "- 研究域：沪深300、中证500历史成分快照；成分权重暂以同期成员等权替代。",
        "- 信号：月末收盘后；标签：下一交易日开盘至下月对应开盘的后复权收益。",
        "- 复权：QMT原始不复权价 × DividData累计后复权因子。2026年1–2月只用于结清",
        "  2025年末冻结测试仓位，不参与方向、入选或权重估计。",
        "- 状态未知即剔除，开发可用段为2015–2020；验证2021–2023；测试信号2024–2025。",
        "- 39个因子：23个量价/换手因子与16个财务/估值因子；`SP_TTM`因缺流通市值未计算。",
        "- 5×MAD、截面标准化；不做行业/市值中性化。最高五分位是等权纯多头诊断组合。",
        "- 成本含佣金3bp、滑点5bp（压力10bp）、按日期切换的印花税和过户费。",
        "",
        "## 样本覆盖",
        "",
        "| 宽基 | 月均严格可用股票数 | 最少 | 最多 |",
        "|---|---:|---:|---:|",
    ]
    counts = panel.drop_duplicates(["signal_date", "symbol", "index_code"]).groupby(
        ["index_code", "signal_date"], observed=True
    )["symbol"].nunique()
    for index_code, values in counts.groupby(level=0):
        lines.append(
            f"| {index_code} | {values.mean():.1f} | {int(values.min())} | {int(values.max())} |"
        )

    lines.extend(
        [
            "",
            "## 单因子主要结果",
            "",
            "IC按预注册方向统一符号，正值代表符合经济假设。q值是在两宽基、全部具体参数、",
            "方向和分期上的统一BH校正。下表按开发期与验证期中较弱的一段排序。",
            "",
        ]
    )
    for index_code, sample in expected.groupby("index_code", observed=True):
        pivot = sample.pivot(index="factor_name", columns="period", values="aligned_ic")
        pivot["dev_q"] = sample[sample["period"] == "development"].set_index("factor_name")[
            "bh_q_value"
        ]
        pivot["stable"] = pivot[["development", "validation"]].min(axis=1)
        chosen = pivot.nlargest(10, "stable")
        lines.extend(
            [
                f"### {index_code}",
                "",
                "| 因子 | 开发IC | 验证IC | 测试IC | 开发q |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for factor_name, row in chosen.iterrows():
            lines.append(
                f"| {factor_name} | {row['development']:.4f} | {row['validation']:.4f} | "
                f"{row['test']:.4f} | {row['dev_q']:.4f} |"
            )
        lines.append("")

    momentum_factors = [
        "REVERSAL_5D",
        "REVERSAL_20D",
        "MOMENTUM_60D",
        "MOMENTUM_120D",
        "MOMENTUM_120D_SKIP20",
        "MOMENTUM_252D_SKIP20",
        "HIGH_252D_PROXIMITY",
    ]
    lines.extend(
        [
            "## 动量与反转因子完整结果",
            "",
            "动量族已经完整计算；它们未进入上面的稳定性前十，是因为验证期方向翻转或",
            "显著性不足，而不是因子遗漏。反转因子已在公式中乘以负号，因此表中正IC仍表示",
            "符合预注册方向。",
            "",
        ]
    )
    for index_code, sample in expected[
        expected["factor_name"].isin(momentum_factors)
    ].groupby("index_code", observed=True):
        pivot = sample.pivot(index="factor_name", columns="period", values="aligned_ic")
        pivot["dev_q"] = sample[sample["period"] == "development"].set_index("factor_name")[
            "bh_q_value"
        ]
        lines.extend(
            [
                f"### {index_code}",
                "",
                "| 因子 | 开发IC | 验证IC | 测试IC | 开发q |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for factor_name in momentum_factors:
            row = pivot.loc[factor_name]
            lines.append(
                f"| {factor_name} | {row['development']:.4f} | {row['validation']:.4f} | "
                f"{row['test']:.4f} | {row['dev_q']:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 多因子验证选择与冻结测试",
            "",
            "入选只看开发期：预期方向一致、BH q<10%、覆盖率≥50%。六种方法只用过去12个月，",
            "半衰期6个月；验证选择不读取2023年12月尚未实现的跨年收益。",
            "",
            "| 宽基 | 选中方法 | 分期 | 年化收益 | Sharpe | 成本后超额IR | 月均换手 | 高减低月均 |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for index_code, method in selection.items():
        rows = multi[
            (multi["index_code"] == index_code)
            & (multi["method"] == method)
            & multi["period"].isin(["development", "validation", "test"])
        ].copy()
        rows["period_order"] = rows["period"].map(
            {"development": 0, "validation": 1, "test": 2}
        )
        rows = rows.sort_values("period_order")
        for row in rows.itertuples(index=False):
            lines.append(
                f"| {index_code} | {method} | {row.period} | {row.annual_return:.2%} | "
                f"{row.sharpe:.3f} | {row.net_information_ratio:.3f} | "
                f"{row.mean_turnover:.2%} | {row.mean_long_short:.2%} |"
            )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "1. 低换手波动、低换手水平、低特质波动和低总波动在开发/验证段最稳定；",
            "   测试期Rank IC多数仍同向，但幅度明显衰减。",
            "2. IC持续不等于极端分组持续：两套冻结主组合在测试期都有正绝对收益，",
            "   但成本后均跑输对应宽基，高减低收益也转负，因此不晋级正式候选。",
            "3. 下一轮应先补历史行业和官方权重，再判断行业中性与权重匹配能否修复超额。",
            "",
            "## 仍未满足的正式门槛",
            "",
            "- 无点时申万一级行业，无法做行业+流通市值WLS中性化和行业内五层。",
            "- 无官方历史权重；当前等权会产生基准结构偏差。",
            "- 无历史涨跌停、公司行动现金流和流通市值；这是向量化诊断，不是完整现金账本。",
            "- 状态源只覆盖部分成分，严格池月均约为完整指数六成，存在覆盖偏差。",
            "",
            "## 产物",
            "",
            f"- IC稳定性图：`{ic_chart}`",
            f"- 多因子超额IR图：`{multi_chart}`",
            f"- 明细目录：`{paths.artifact_root}`",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    storage_io.write_text(report_path, "\n".join(lines), encoding="utf-8")

    import hashlib

    manifest: dict[str, dict[str, object]] = {}
    for artifact in sorted(storage_io.iterdir(paths.artifact_root)):
        if (
            storage_io.exists(artifact)
            and artifact.name != "manifest.json"
            and not artifact.name.startswith("daily_2013_")
        ):
            digest = hashlib.sha256(storage_io.read_bytes(artifact))
            manifest[artifact.name] = {
                "bytes": storage_io.stat(artifact).st_size,
                "sha256": digest.hexdigest(),
            }
    storage_io.write_text(paths.artifact_root / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report_path
