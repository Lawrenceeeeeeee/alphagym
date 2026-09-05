"""Auditable price-volume technical factors sourced from ``factor_list.xlsx``.

The workbook contains raw price-level indicators, exact aliases and four XSII
fields without a public formula.  Price-level indicators are normalized by the
current backward-adjusted close; exact aliases are represented once; XSII is
blocked pending vendor calibration.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from mlquant.factors.base import FactorContext, FactorRegistry, FactorSpec


def _metadata() -> dict[str, tuple[str, tuple[str, ...], int, int]]:
    result: dict[str, tuple[str, tuple[str, ...], int, int]] = {}

    def add(
        name: str,
        hypothesis: str,
        inputs: tuple[str, ...],
        lookback: int,
        minimum: int | None = None,
    ) -> None:
        result[name] = (hypothesis, inputs, lookback, minimum or max(1, int(lookback * 0.8)))

    for window in (5, 20, 60, 120, 6, 12, 24):
        add(f"BIAS_{window}D", "technical_price_ma_bias", ("adj_close",), window)
    for price_window, signal_window in ((5, 3), (12, 9), (20, 12), (60, 20)):
        add(
            f"TRIX_SIGNAL_{price_window}_{signal_window}",
            "technical_trix",
            ("adj_close",),
            price_window * 3 + signal_window,
        )
    add("TRIX_12D", "technical_trix", ("adj_close",), 37, 30)
    add("TRIX_SIGNAL_12_20", "technical_trix", ("adj_close",), 56, 45)
    for window in (6, 12, 24, 60):
        add(f"RSI_{window}D", "technical_rsi", ("adj_close",), window + 1)
    for window in (5, 9, 14, 20):
        for output in ("RSV", "J"):
            add(
                f"KDJ_{output}_{window}D",
                "technical_kdj",
                ("adj_high", "adj_low", "adj_close"),
                window + 6,
            )
    add("KDJ_K_9D", "technical_kdj", ("adj_high", "adj_low", "adj_close"), 15)
    add("KDJ_D_9D", "technical_kdj", ("adj_high", "adj_low", "adj_close"), 15)
    for window in (5, 14, 20, 60):
        add(f"CCI_{window}D", "technical_cci", ("adj_high", "adj_low", "adj_close"), window)
    add("OBV", "technical_obv", ("adj_close", "volume"), 2, 2)
    for window in (6, 20, 60):
        add(f"MAOBV_{window}D", "technical_obv", ("adj_close", "volume"), window + 1)
    add("BBI_PCT", "technical_bbi", ("adj_close",), 20)
    add("DPO_20D_PCT", "technical_dpo", ("adj_close",), 31)
    add("ROC_12D", "price_momentum", ("adj_close",), 13)
    add("WR_14D", "technical_williams_r", ("adj_high", "adj_low", "adj_close"), 14)
    add("WR_6D", "technical_williams_r", ("adj_high", "adj_low", "adj_close"), 6)
    add("PSY_12D", "technical_psychological_line", ("adj_close",), 13)
    add("PSYMA_12_6", "technical_psychological_line", ("adj_close",), 18)
    add("UP_STREAK", "technical_streak", ("adj_close",), 2, 2)
    add("DOWN_STREAK", "technical_streak", ("adj_close",), 2, 2)
    add("DAYS_SINCE_HIGH_20D", "technical_extreme_recency", ("adj_high",), 20)
    add("DAYS_SINCE_LOW_20D", "technical_extreme_recency", ("adj_low",), 20)
    add("MASS_9_25", "technical_mass", ("adj_high", "adj_low"), 43)
    add("MASS_MA_9_25_6", "technical_mass", ("adj_high", "adj_low"), 48)
    add("MFI_14D", "technical_money_flow", ("adj_high", "adj_low", "adj_close", "volume"), 15)
    add("VR_26D", "technical_volume_ratio", ("adj_close", "volume"), 27)
    add("EMV_14D", "technical_emv", ("adj_high", "adj_low", "volume"), 15)
    add("MAEMV_14_9", "technical_emv", ("adj_high", "adj_low", "volume"), 23)
    add("AR_26D", "technical_sentiment_ar", ("adj_open", "adj_high", "adj_low"), 26)
    add("BR_26D", "technical_sentiment_br", ("adj_high", "adj_low", "adj_close"), 27)
    add("CR_20D", "technical_cr", ("adj_high", "adj_low"), 21)
    for output in ("PDI", "MDI", "ADX", "ADXR"):
        add(
            f"DMI_{output}_14_6",
            "technical_dmi",
            ("adj_high", "adj_low", "adj_close"),
            35 if output == "ADXR" else 29,
        )
    for fast, slow, signal in ((5, 20, 5), (12, 26, 9), (10, 50, 10)):
        for output in ("DIF", "DEA", "HIST"):
            add(
                f"MACD_{output}_{fast}_{slow}_{signal}_PCT",
                "technical_macd",
                ("adj_close",),
                slow + signal,
            )
    for window in (6, 14, 20, 60):
        add(
            f"ATR_{window}D_PCT",
            "technical_atr",
            ("adj_high", "adj_low", "adj_close"),
            window + 1,
        )
    for window in (10, 20):
        add(f"BBANDS_WIDTH_{window}_2", "technical_bollinger", ("adj_close",), window)
        add(f"BBANDS_POSITION_{window}_2", "technical_bollinger", ("adj_close",), window)
    for window in (20, 60):
        add(
            f"DONCHIAN_POSITION_{window}D",
            "technical_donchian",
            ("adj_high", "adj_low", "adj_close"),
            window,
        )
        add(
            f"DONCHIAN_WIDTH_{window}D",
            "technical_donchian",
            ("adj_high", "adj_low", "adj_close"),
            window,
        )
    add("ASI_26D", "technical_asi", ("adj_open", "adj_high", "adj_low", "adj_close"), 27)
    add("ASIT_26_10", "technical_asi", ("adj_open", "adj_high", "adj_low", "adj_close"), 36)
    for output in ("LOWER", "MID", "UPPER"):
        add(f"BOLL_{output}_20D_PCT", "technical_bollinger_level", ("adj_close",), 20)
    add("DFMA_DIF_10_50_PCT", "technical_dfma", ("adj_close",), 50)
    add("DFMA_SIGNAL_10_50_10_PCT", "technical_dfma", ("adj_close",), 59)
    add("MADPO_20_10_6_PCT", "technical_dpo", ("adj_close",), 36)
    add("EXPMA_12D_PCT", "technical_expma", ("adj_close",), 12)
    add("EXPMA_50D_PCT", "technical_expma", ("adj_close",), 50)
    for output in ("LOWER", "MID", "UPPER"):
        add(
            f"KELTNER_{output}_20_10_PCT",
            "technical_keltner",
            ("adj_high", "adj_low", "adj_close"),
            21,
        )
    add("MTMMA_12_6_PCT", "technical_momentum_average", ("adj_close",), 18)
    add("MAROC_12_6", "technical_momentum_average", ("adj_close",), 18)
    for output in ("LOWER", "MID", "UPPER"):
        add(
            f"DONCHIAN_{output}_20D_PCT",
            "technical_donchian_level",
            ("adj_high", "adj_low", "adj_close"),
            20,
        )
    if len(result) != 98:
        raise AssertionError(f"expected 98 technical factors, got {len(result)}")
    return result


TECHNICAL_METADATA = _metadata()
TECHNICAL_FACTOR_NAMES = tuple(TECHNICAL_METADATA)
BLOCKED_SOURCE_FACTORS = {
    "xsii_td1_bfq": "vendor formula unavailable; calibrate against Tushare",
    "xsii_td2_bfq": "vendor formula unavailable; calibrate against Tushare",
    "xsii_td3_bfq": "vendor formula unavailable; calibrate against Tushare",
    "xsii_td4_bfq": "vendor formula unavailable; calibrate against Tushare",
}
SOURCE_ALIASES = {
    "mtm_bfq": "ROC_12D",
    "atr_bfq": "ATR_20D_PCT",
    "cci_bfq": "CCI_14D",
    "kdj_bfq": "KDJ_J_9D",
    "obv_bfq": "OBV",
    "macd_dif_bfq": "MACD_DIF_12_26_9_PCT",
    "macd_dea_bfq": "MACD_DEA_12_26_9_PCT",
    "macd_bfq": "MACD_HIST_12_26_9_PCT",
}


def _rolling(series: pd.Series, window: int, minimum: int, operation: str) -> pd.Series:
    grouped = series.groupby(level=0, observed=True)
    result = getattr(grouped.rolling(window, min_periods=minimum), operation)()
    return result.reset_index(level=0, drop=True).reindex(series.index)


def _rolling_apply(
    series: pd.Series, window: int, minimum: int, function: object
) -> pd.Series:
    grouped = series.groupby(level=0, observed=True)
    result = grouped.rolling(window, min_periods=minimum).apply(function, raw=True)
    return result.reset_index(level=0, drop=True).reindex(series.index)


def _ewm(
    series: pd.Series,
    *,
    span: int | None = None,
    alpha: float | None = None,
    minimum: int,
) -> pd.Series:
    grouped = series.groupby(level=0, observed=True)
    kwargs = {"adjust": False, "min_periods": minimum}
    if span is not None:
        kwargs["span"] = span
    else:
        kwargs["alpha"] = alpha
    result = grouped.ewm(**kwargs).mean()
    return result.reset_index(level=0, drop=True).reindex(series.index)


def _shift(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.groupby(level=0, observed=True).shift(periods)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _streak(series: pd.Series, positive: bool) -> pd.Series:
    def one(values: pd.Series) -> pd.Series:
        condition = values.diff().gt(0) if positive else values.diff().lt(0)
        groups = (~condition).cumsum()
        return condition.groupby(groups).cumsum().astype(float)

    return series.groupby(level=0, group_keys=False, observed=True).apply(one).reindex(series.index)


def compute_technical_features(
    daily: pd.DataFrame,
    signal_index: pd.MultiIndex,
    names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute requested factors on daily history and sample them at signal dates."""
    requested = set(names or TECHNICAL_FACTOR_NAMES)
    unknown = requested - set(TECHNICAL_FACTOR_NAMES)
    if unknown:
        raise KeyError(f"unknown technical factors: {sorted(unknown)}")
    indexed = daily.set_index(["symbol", "trade_date"]).sort_index()
    close = indexed["adj_close"]
    if "adjust_factor" in indexed:
        factor = indexed["adjust_factor"]
    elif "close" in indexed:
        factor = _safe_ratio(close, indexed["close"])
    else:
        factor = pd.Series(1.0, index=indexed.index)
    high = indexed["adj_high"] if "adj_high" in indexed else indexed["high"] * factor
    low = indexed["adj_low"] if "adj_low" in indexed else indexed["low"] * factor
    open_ = indexed["adj_open"] if "adj_open" in indexed else indexed["open"] * factor
    volume = indexed["volume"].replace(0, np.nan)
    target = pd.MultiIndex.from_arrays(
        [signal_index.get_level_values(0), signal_index.get_level_values(1)],
        names=["symbol", "trade_date"],
    )
    output: dict[str, np.ndarray] = {}

    def put(name: str, values: pd.Series) -> None:
        if name in requested:
            output[name] = values.reindex(target).to_numpy()

    bias_names = {name for name in requested if name.startswith("BIAS_")}
    for window in (5, 20, 60, 120, 6, 12, 24):
        name = f"BIAS_{window}D"
        if name in bias_names:
            put(name, _safe_ratio(close, _rolling(close, window, max(1, int(window * 0.8)), "mean")) - 1)

    trix_names = {name for name in requested if name.startswith("TRIX")}
    if trix_names:
        cache: dict[int, pd.Series] = {}
        for window in (5, 12, 20, 60):
            first = _ewm(close, span=window, minimum=window)
            second = _ewm(first, span=window, minimum=window)
            third = _ewm(second, span=window, minimum=window)
            cache[window] = _safe_ratio(third, _shift(third)) - 1
        for window, signal in ((5, 3), (12, 9), (20, 12), (60, 20)):
            put(
                f"TRIX_SIGNAL_{window}_{signal}",
                _rolling(cache[window], signal, max(2, int(signal * 0.8)), "mean"),
            )
        put("TRIX_12D", cache[12])
        put("TRIX_SIGNAL_12_20", _rolling(cache[12], 20, 16, "mean"))

    rsi_names = {name for name in requested if name.startswith("RSI_")}
    if rsi_names:
        delta = close - _shift(close)
        for window in (6, 12, 24, 60):
            gain = _ewm(delta.clip(lower=0), span=window, minimum=window)
            loss = _ewm((-delta).clip(lower=0), span=window, minimum=window)
            put(f"RSI_{window}D", 100 * _safe_ratio(gain, gain + loss))

    kdj_names = {name for name in requested if name.startswith("KDJ_")}
    if kdj_names:
        for window in (5, 9, 14, 20):
            lower = _rolling(low, window, max(3, int(window * 0.8)), "min")
            upper = _rolling(high, window, max(3, int(window * 0.8)), "max")
            rsv = 100 * _safe_ratio(close - lower, upper - lower)
            k = _ewm(rsv, alpha=1 / 3, minimum=3)
            d = _ewm(k, alpha=1 / 3, minimum=3)
            put(f"KDJ_RSV_{window}D", rsv)
            put(f"KDJ_J_{window}D", 3 * k - 2 * d)
            if window == 9:
                put("KDJ_K_9D", k)
                put("KDJ_D_9D", d)

    cci_names = {name for name in requested if name.startswith("CCI_")}
    if cci_names:
        typical = (high + low + close) / 3
        for window in (5, 14, 20, 60):
            mean = _rolling(typical, window, max(3, int(window * 0.8)), "mean")
            deviation = _rolling_apply(
                typical,
                window,
                max(3, int(window * 0.8)),
                lambda values: np.mean(np.abs(values - values.mean())),
            )
            put(f"CCI_{window}D", _safe_ratio(typical - mean, 0.015 * deviation))

    obv_names = {name for name in requested if "OBV" in name}
    if obv_names:
        sign = np.sign(close - _shift(close)).fillna(0)
        obv = (sign * volume.fillna(0)).groupby(level=0, observed=True).cumsum()
        put("OBV", obv)
        for window in (6, 20, 60):
            put(f"MAOBV_{window}D", _rolling(obv, window, max(3, int(window * 0.8)), "mean"))

    if "BBI_PCT" in requested:
        bbi = sum(_rolling(close, window, window, "mean") for window in (3, 6, 12, 20)) / 4
        put("BBI_PCT", _safe_ratio(bbi, close) - 1)
    if "DPO_20D_PCT" in requested or "MADPO_20_10_6_PCT" in requested:
        dpo = _safe_ratio(_shift(close, 11) - _rolling(close, 20, 16, "mean"), close)
        put("DPO_20D_PCT", dpo)
        put("MADPO_20_10_6_PCT", _rolling(dpo, 6, 5, "mean"))
    if "ROC_12D" in requested or "MAROC_12_6" in requested:
        roc = _safe_ratio(close, _shift(close, 12)) - 1
        put("ROC_12D", roc)
        put("MAROC_12_6", _rolling(roc, 6, 5, "mean"))
    if "MTMMA_12_6_PCT" in requested:
        mtm_pct = _safe_ratio(close - _shift(close, 12), close)
        put("MTMMA_12_6_PCT", _rolling(mtm_pct, 6, 5, "mean"))

    wr_names = {name for name in requested if name.startswith("WR_")}
    for window in (6, 14):
        if f"WR_{window}D" in wr_names:
            upper = _rolling(high, window, max(3, int(window * 0.8)), "max")
            lower = _rolling(low, window, max(3, int(window * 0.8)), "min")
            put(f"WR_{window}D", -100 * _safe_ratio(upper - close, upper - lower))

    if {"PSY_12D", "PSYMA_12_6"} & requested:
        up = (close > _shift(close)).astype(float)
        psy = 100 * _rolling(up, 12, 10, "mean")
        put("PSY_12D", psy)
        put("PSYMA_12_6", _rolling(psy, 6, 5, "mean"))
    if {"UP_STREAK", "DOWN_STREAK"} & requested:
        put("UP_STREAK", _streak(close, True))
        put("DOWN_STREAK", _streak(close, False))
    if "DAYS_SINCE_HIGH_20D" in requested:
        put(
            "DAYS_SINCE_HIGH_20D",
            _rolling_apply(high, 20, 16, lambda values: len(values) - 1 - np.argmax(values)),
        )
    if "DAYS_SINCE_LOW_20D" in requested:
        put(
            "DAYS_SINCE_LOW_20D",
            _rolling_apply(low, 20, 16, lambda values: len(values) - 1 - np.argmin(values)),
        )

    range_ = high - low
    if {"MASS_9_25", "MASS_MA_9_25_6"} & requested:
        ema1 = _ewm(range_, span=9, minimum=9)
        ema2 = _ewm(ema1, span=9, minimum=9)
        mass = _rolling(_safe_ratio(ema1, ema2), 25, 20, "sum")
        put("MASS_9_25", mass)
        put("MASS_MA_9_25_6", _rolling(mass, 6, 5, "mean"))
    if "MFI_14D" in requested:
        typical = (high + low + close) / 3
        flow = typical * volume
        direction = typical - _shift(typical)
        positive = flow.where(direction > 0, 0.0)
        negative = flow.where(direction < 0, 0.0)
        ratio = _safe_ratio(_rolling(positive, 14, 11, "sum"), _rolling(negative, 14, 11, "sum"))
        put("MFI_14D", 100 - 100 / (1 + ratio))
    if "VR_26D" in requested:
        change = close - _shift(close)
        av = _rolling(volume.where(change > 0, 0.0), 26, 21, "sum")
        bv = _rolling(volume.where(change < 0, 0.0), 26, 21, "sum")
        cv = _rolling(volume.where(change == 0, 0.0), 26, 21, "sum")
        put("VR_26D", 100 * _safe_ratio(av + 0.5 * cv, bv + 0.5 * cv))
    if {"EMV_14D", "MAEMV_14_9"} & requested:
        midpoint = (high + low) / 2
        emv_raw = (midpoint - _shift(midpoint)) * range_ / volume
        emv = _rolling(emv_raw, 14, 11, "mean")
        put("EMV_14D", emv)
        put("MAEMV_14_9", _rolling(emv, 9, 7, "mean"))
    if "AR_26D" in requested:
        put(
            "AR_26D",
            100
            * _safe_ratio(
                _rolling(high - open_, 26, 21, "sum"),
                _rolling(open_ - low, 26, 21, "sum"),
            ),
        )
    previous_close = _shift(close)
    if "BR_26D" in requested:
        put(
            "BR_26D",
            100
            * _safe_ratio(
                _rolling((high - previous_close).clip(lower=0), 26, 21, "sum"),
                _rolling((previous_close - low).clip(lower=0), 26, 21, "sum"),
            ),
        )
    if "CR_20D" in requested:
        previous_mid = _shift((high + low) / 2)
        put(
            "CR_20D",
            100
            * _safe_ratio(
                _rolling((high - previous_mid).clip(lower=0), 20, 16, "sum"),
                _rolling((previous_mid - low).clip(lower=0), 20, 16, "sum"),
            ),
        )

    dmi_names = {name for name in requested if name.startswith("DMI_")}
    true_range = pd.concat(
        [range_, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    if dmi_names:
        high_change = high - _shift(high)
        low_change = _shift(low) - low
        plus_dm = high_change.where((high_change > low_change) & (high_change > 0), 0.0)
        minus_dm = low_change.where((low_change > high_change) & (low_change > 0), 0.0)
        tr_mean = _rolling(true_range, 14, 11, "mean")
        pdi = 100 * _safe_ratio(_rolling(plus_dm, 14, 11, "mean"), tr_mean)
        mdi = 100 * _safe_ratio(_rolling(minus_dm, 14, 11, "mean"), tr_mean)
        dx = 100 * _safe_ratio((pdi - mdi).abs(), pdi + mdi)
        adx = _rolling(dx, 6, 5, "mean")
        put("DMI_PDI_14_6", pdi)
        put("DMI_MDI_14_6", mdi)
        put("DMI_ADX_14_6", adx)
        put("DMI_ADXR_14_6", (adx + _shift(adx, 6)) / 2)

    macd_names = {name for name in requested if name.startswith("MACD_")}
    if macd_names:
        for fast, slow, signal in ((5, 20, 5), (12, 26, 9), (10, 50, 10)):
            dif = _ewm(close, span=fast, minimum=fast) - _ewm(
                close, span=slow, minimum=slow
            )
            dea = _ewm(dif, span=signal, minimum=signal)
            put(f"MACD_DIF_{fast}_{slow}_{signal}_PCT", _safe_ratio(dif, close))
            put(f"MACD_DEA_{fast}_{slow}_{signal}_PCT", _safe_ratio(dea, close))
            put(
                f"MACD_HIST_{fast}_{slow}_{signal}_PCT",
                _safe_ratio(2 * (dif - dea), close),
            )
    for window in (6, 14, 20, 60):
        put(
            f"ATR_{window}D_PCT",
            _safe_ratio(
                _rolling(true_range, window, max(3, int(window * 0.8)), "mean"), close
            ),
        )

    bollinger_names = {name for name in requested if "BAND" in name or name.startswith("BOLL_")}
    if bollinger_names:
        for window in (10, 20):
            mid = _rolling(close, window, max(3, int(window * 0.8)), "mean")
            std = _rolling(close, window, max(3, int(window * 0.8)), "std")
            upper = mid + 2 * std
            lower = mid - 2 * std
            put(f"BBANDS_WIDTH_{window}_2", _safe_ratio(upper - lower, mid))
            put(f"BBANDS_POSITION_{window}_2", _safe_ratio(close - lower, upper - lower))
            if window == 20:
                put("BOLL_LOWER_20D_PCT", _safe_ratio(lower, close) - 1)
                put("BOLL_MID_20D_PCT", _safe_ratio(mid, close) - 1)
                put("BOLL_UPPER_20D_PCT", _safe_ratio(upper, close) - 1)

    donchian_names = {name for name in requested if name.startswith("DONCHIAN_")}
    if donchian_names:
        for window in (20, 60):
            upper = _rolling(high, window, max(3, int(window * 0.8)), "max")
            lower = _rolling(low, window, max(3, int(window * 0.8)), "min")
            put(f"DONCHIAN_POSITION_{window}D", _safe_ratio(close - lower, upper - lower))
            put(f"DONCHIAN_WIDTH_{window}D", _safe_ratio(upper - lower, close))
            if window == 20:
                mid = (upper + lower) / 2
                put("DONCHIAN_LOWER_20D_PCT", _safe_ratio(lower, close) - 1)
                put("DONCHIAN_MID_20D_PCT", _safe_ratio(mid, close) - 1)
                put("DONCHIAN_UPPER_20D_PCT", _safe_ratio(upper, close) - 1)

    if {"ASI_26D", "ASIT_26_10"} & requested:
        previous_open = _shift(open_)
        previous_low = _shift(low)
        a = (high - previous_close).abs()
        b = (low - previous_close).abs()
        c = (high - previous_low).abs()
        d = (previous_close - previous_open).abs()
        x = close - previous_close + 0.5 * (close - open_) + previous_close - previous_open
        k = pd.concat([a, b], axis=1).max(axis=1)
        r = pd.Series(
            np.select(
                [(a >= b) & (a >= c), (b > a) & (b >= c)],
                [a + 0.5 * b + 0.25 * d, b + 0.5 * a + 0.25 * d],
                default=c + 0.25 * d,
            ),
            index=close.index,
        )
        si = 50 * _safe_ratio(x * k, r) / 26
        asi = _rolling(si, 26, 21, "sum")
        put("ASI_26D", asi)
        put("ASIT_26_10", _rolling(asi, 10, 8, "mean"))

    if {"DFMA_DIF_10_50_PCT", "DFMA_SIGNAL_10_50_10_PCT"} & requested:
        dif = _rolling(close, 10, 8, "mean") - _rolling(close, 50, 40, "mean")
        put("DFMA_DIF_10_50_PCT", _safe_ratio(dif, close))
        put(
            "DFMA_SIGNAL_10_50_10_PCT",
            _safe_ratio(_rolling(dif, 10, 8, "mean"), close),
        )
    for window in (12, 50):
        put(
            f"EXPMA_{window}D_PCT",
            _safe_ratio(_ewm(close, span=window, minimum=window), close) - 1,
        )
    keltner_names = {name for name in requested if name.startswith("KELTNER_")}
    if keltner_names:
        mid = _ewm(close, span=20, minimum=20)
        atr = _rolling(true_range, 10, 8, "mean")
        put("KELTNER_LOWER_20_10_PCT", _safe_ratio(mid - atr, close) - 1)
        put("KELTNER_MID_20_10_PCT", _safe_ratio(mid, close) - 1)
        put("KELTNER_UPPER_20_10_PCT", _safe_ratio(mid + atr, close) - 1)

    return pd.DataFrame(output)


def _calculator(name: str):
    def calculate(context: FactorContext) -> pd.Series:
        visible = context.daily[context.daily["trade_date"] <= context.signal_date]
        symbols = visible["symbol"].drop_duplicates().sort_values()
        signal_index = pd.MultiIndex.from_arrays(
            [symbols, np.repeat(context.signal_date, len(symbols))],
            names=["symbol", "signal_date"],
        )
        values = compute_technical_features(visible, signal_index, [name])[name]
        return pd.Series(values.to_numpy(), index=symbols, name=name).rename_axis("symbol")

    return calculate


def register_technical_factors(registry: FactorRegistry) -> None:
    for name, (hypothesis, inputs, lookback, minimum) in TECHNICAL_METADATA.items():
        registry.register(
            FactorSpec(
                name=name,
                hypothesis_id=hypothesis,
                family="technical",
                formula_version="factor_list_xlsx_v1_adjusted_dimensionless",
                input_fields=inputs,
                lookback_days=lookback,
                min_observations=minimum,
                availability_rule="market data <= signal close; backward-adjusted OHLC",
                expected_direction="unknown",
                calculator=_calculator(name),
            )
        )
