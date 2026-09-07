from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from alphagym.factor_dsl import (
    FieldRef,
    FieldRegistry,
    ModelRef,
    OperatorRegistry,
    OperatorSpec,
)
from alphagym.factors.base import FactorContext, FieldDefinition


def _visible(context: FactorContext, ref: FieldRef) -> pd.DataFrame:
    return context.visible(ref.definition).sort_values(
        [ref.definition.entity_key, ref.definition.event_time]
    )


def _latest(context: FactorContext, ref: FieldRef) -> pd.Series:
    definition = ref.definition
    frame = _visible(context, ref)
    if definition.dataset == "financial":
        frame = frame.sort_values(["symbol", "stat_date", "available_date"])
        frame = frame.drop_duplicates(["symbol", "stat_date"], keep="last")
    result = frame.groupby(definition.entity_key, observed=True).tail(1)
    return pd.to_numeric(result.set_index(definition.entity_key)[definition.column], errors="coerce")


def op_asof(context: FactorContext, field: FieldRef) -> pd.Series:
    return _latest(context, field)


def _rolling(
    context: FactorContext, field: FieldRef, window: int, operation: str,
    min_observations: int | None = None,
) -> pd.Series:
    minimum = min_observations if min_observations is not None else max(1, int(window * 0.8))
    definition = field.definition

    def one(group: pd.DataFrame) -> float:
        values = pd.to_numeric(group[definition.column], errors="coerce").dropna().tail(window)
        if len(values) < minimum:
            return np.nan
        return float(getattr(values, operation)())

    return _visible(context, field).groupby(definition.entity_key, observed=True).apply(
        one, include_groups=False
    )


def op_rolling_mean(
    context: FactorContext, field: FieldRef, window: int, min_observations: int | None = None
) -> pd.Series:
    return _rolling(context, field, window, "mean", min_observations)


def op_std(
    context: FactorContext, field: FieldRef, window: int, min_observations: int | None = None
) -> pd.Series:
    return _rolling(context, field, window, "std", min_observations)


def op_sum(
    context: FactorContext, field: FieldRef, window: int, min_observations: int | None = None
) -> pd.Series:
    return _rolling(context, field, window, "sum", min_observations)


def op_min(
    context: FactorContext, field: FieldRef, window: int, min_observations: int | None = None
) -> pd.Series:
    return _rolling(context, field, window, "min", min_observations)


def op_max(
    context: FactorContext, field: FieldRef, window: int, min_observations: int | None = None
) -> pd.Series:
    return _rolling(context, field, window, "max", min_observations)


def op_return(
    context: FactorContext, field: FieldRef, window: int, skip: int = 0
) -> pd.Series:
    definition = field.definition

    def one(group: pd.DataFrame) -> float:
        values = pd.to_numeric(group[definition.column], errors="coerce").dropna()
        end = len(values) - skip
        start = end - window - 1
        if start < 0 or end <= 0:
            return np.nan
        return float(values.iloc[end - 1] / values.iloc[start] - 1.0)

    return _visible(context, field).groupby(definition.entity_key, observed=True).apply(
        one, include_groups=False
    )


def op_diff(context: FactorContext, field: FieldRef, periods: int = 1) -> pd.Series:
    definition = field.definition

    def one(group: pd.DataFrame) -> float:
        values = pd.to_numeric(group[definition.column], errors="coerce").dropna()
        return float(values.iloc[-1] - values.iloc[-periods - 1]) if len(values) > periods else np.nan

    return _visible(context, field).groupby(definition.entity_key, observed=True).apply(
        one, include_groups=False
    )


def op_ewm(context: FactorContext, field: FieldRef, span: int) -> pd.Series:
    definition = field.definition

    def one(group: pd.DataFrame) -> float:
        values = pd.to_numeric(group[definition.column], errors="coerce").dropna()
        if len(values) < span:
            return np.nan
        return float(values.ewm(span=span, adjust=False, min_periods=span).mean().iloc[-1])

    return _visible(context, field).groupby(definition.entity_key, observed=True).apply(
        one, include_groups=False
    )


def op_safe_div(_context: FactorContext, numerator: object, denominator: object) -> object:
    if isinstance(denominator, pd.Series):
        denominator = denominator.replace(0, np.nan)
    elif denominator == 0:
        return np.nan
    return numerator / denominator


def op_log(_context: FactorContext, value: object) -> object:
    return np.log(value)


def op_abs(_context: FactorContext, value: object) -> object:
    return np.abs(value)


def op_clip(
    _context: FactorContext, value: object, lower: float | None = None,
    upper: float | None = None,
) -> object:
    return value.clip(lower=lower, upper=upper) if isinstance(value, pd.Series) else np.clip(value, lower, upper)


def op_rank(_context: FactorContext, value: pd.Series, pct: bool = True) -> pd.Series:
    return value.rank(method="average", pct=pct)


def op_zscore(_context: FactorContext, value: pd.Series) -> pd.Series:
    standard_deviation = value.std(ddof=0)
    return (value - value.mean()) / standard_deviation if standard_deviation else value * np.nan


def op_winsorize(_context: FactorContext, value: pd.Series, scale: float = 5.0) -> pd.Series:
    median = value.median()
    mad = (value - median).abs().median()
    return value.clip(median - scale * mad, median + scale * mad) if mad else value


def op_features(_context: FactorContext, *values: object) -> list[object]:
    return list(values)


def _model_entry(context: FactorContext, model: ModelRef) -> tuple[object, pd.Timestamp]:
    try:
        entry = context.models.get(model.version) if model.version else None
        if entry is None:
            entry = context.models[model.model_id]
    except KeyError as error:
        raise ValueError(f"model unavailable in context: {model.model_id}") from error
    if isinstance(entry, dict):
        estimator = entry.get("estimator")
        training_end = pd.Timestamp(entry.get("training_end"))
    else:
        estimator = entry
        training_end = pd.Timestamp.min
    if training_end > context.signal_date:
        raise ValueError(
            f"model {model.model_id} training_end {training_end.date()} exceeds signal date"
        )
    return estimator, training_end


def op_model_predict(
    context: FactorContext, model: ModelRef, features: list[object]
) -> pd.Series:
    estimator, _training_end = _model_entry(context, model)
    series = [value for value in features if isinstance(value, pd.Series)]
    if len(series) != len(features) or not series:
        raise ValueError("MODEL_PREDICT features must be cross-sectional Series")
    matrix = pd.concat(series, axis=1, join="inner")
    prediction = estimator.predict(matrix.to_numpy())
    return pd.Series(prediction, index=matrix.index, dtype=float)


def op_ensemble(
    _context: FactorContext, predictions: list[pd.Series],
    weights: list[float] | None = None,
) -> pd.Series:
    if not predictions:
        raise ValueError("ENSEMBLE requires at least one prediction")
    frame = pd.concat(predictions, axis=1, join="inner")
    weight = np.asarray(weights if weights is not None else np.ones(len(predictions)), dtype=float)
    if len(weight) != len(predictions) or weight.sum() == 0:
        raise ValueError("ENSEMBLE weights must match predictions and have non-zero sum")
    return frame.mul(weight / weight.sum(), axis=1).sum(axis=1)


def _latest_raw(context: FactorContext, field: FieldRef) -> pd.Series:
    definition = field.definition
    frame = _visible(context, field).groupby(definition.entity_key, observed=True).tail(1)
    return frame.set_index(definition.entity_key)[definition.column]


def op_text_score(context: FactorContext, field: FieldRef, model: ModelRef) -> pd.Series:
    estimator, _training_end = _model_entry(context, model)
    documents = _latest_raw(context, field).dropna().astype(str)
    prediction = estimator.predict(documents.tolist())
    return pd.Series(prediction, index=documents.index, dtype=float)


def op_embedding(context: FactorContext, field: FieldRef, model: ModelRef) -> pd.Series:
    estimator, _training_end = _model_entry(context, model)
    documents = _latest_raw(context, field).dropna().astype(str)
    vectors = estimator.transform(documents.tolist())
    return pd.Series(list(vectors), index=documents.index, dtype=object)


def op_document_agg(
    _context: FactorContext, values: pd.Series, method: str = "mean"
) -> pd.Series:
    if method not in {"mean", "sum", "max", "min"}:
        raise ValueError(f"unsupported document aggregation: {method}")
    return values


def _financial_history(context: FactorContext, field: FieldRef) -> pd.DataFrame:
    frame = _visible(context, field)
    frame = frame.sort_values(["symbol", "stat_date", "available_date"])
    return frame.drop_duplicates(["symbol", "stat_date"], keep="last")


def _ttm_one(group: pd.DataFrame, column: str) -> float:
    values = group.set_index("stat_date")[column].dropna().sort_index()
    if values.empty:
        return np.nan
    date = pd.Timestamp(values.index[-1])
    if date.month == 12:
        return float(values.iloc[-1])
    prior_fy = pd.Timestamp(date.year - 1, 12, 31)
    prior_same = date - pd.DateOffset(years=1)
    if prior_fy not in values.index or prior_same not in values.index:
        return np.nan
    return float(values.iloc[-1] + values.loc[prior_fy] - values.loc[prior_same])


def op_ttm(context: FactorContext, field: FieldRef) -> pd.Series:
    frame = _financial_history(context, field)
    return frame.groupby("symbol", observed=True).apply(
        lambda group: _ttm_one(group, field.definition.column), include_groups=False
    )


def op_mrq(context: FactorContext, field: FieldRef) -> pd.Series:
    return _latest(context, field)


def op_yoy(context: FactorContext, field: FieldRef, difference: bool = False) -> pd.Series:
    frame = _financial_history(context, field)
    column = field.definition.column

    def one(group: pd.DataFrame) -> float:
        values = group.set_index("stat_date")[column].dropna().sort_index()
        if values.empty:
            return np.nan
        prior_date = pd.Timestamp(values.index[-1]) - pd.DateOffset(years=1)
        if prior_date not in values.index:
            return np.nan
        current, prior = float(values.iloc[-1]), float(values.loc[prior_date])
        return current - prior if difference else (current / abs(prior) - 1 if prior else np.nan)

    return frame.groupby("symbol", observed=True).apply(one, include_groups=False)


def op_qoq(context: FactorContext, field: FieldRef, difference: bool = False) -> pd.Series:
    frame = _financial_history(context, field)
    column = field.definition.column

    def one(group: pd.DataFrame) -> float:
        values = group.set_index("stat_date")[column].dropna().sort_index()
        if len(values) < 2:
            return np.nan
        current, prior = float(values.iloc[-1]), float(values.iloc[-2])
        return current - prior if difference else (current / abs(prior) - 1 if prior else np.nan)

    return frame.groupby("symbol", observed=True).apply(one, include_groups=False)


def op_stability(
    context: FactorContext, field: FieldRef, observations: int,
    minimum: int, growth: bool = False,
) -> pd.Series:
    frame = _financial_history(context, field)
    column = field.definition.column

    def one(group: pd.DataFrame) -> float:
        values = group.sort_values("stat_date")[column].dropna().tail(observations)
        if growth:
            values = values.pct_change(fill_method=None).dropna()
        return float(-values.std(ddof=1)) if len(values) >= minimum else np.nan

    return frame.groupby("symbol", observed=True).apply(one, include_groups=False)


def op_turnover_bias(
    context: FactorContext, field: FieldRef, short: int, long: int
) -> pd.Series:
    definition = field.definition

    def one(group: pd.DataFrame) -> float:
        values = group[definition.column].dropna()
        if len(values) < int(long * 0.8):
            return np.nan
        denominator = values.tail(long).mean()
        return float(values.tail(short).mean() / denominator - 1) if denominator else np.nan

    return _visible(context, field).groupby("symbol", observed=True).apply(one, include_groups=False)


def op_amihud(
    context: FactorContext, close: FieldRef, amount: FieldRef, window: int,
    minimum: int | None = None,
) -> pd.Series:
    frame = _visible(context, close)
    minimum = minimum or max(1, int(window * 0.8))

    def one(group: pd.DataFrame) -> float:
        group = group.tail(window + 1)
        values = (
            group[close.definition.column].pct_change(fill_method=None).abs()
            / group[amount.definition.column].replace(0, np.nan)
        ).dropna()
        return float(values.mean() * 1e8) if len(values) >= minimum else np.nan

    return frame.groupby("symbol", observed=True).apply(one, include_groups=False)


def op_high_proximity(context: FactorContext, field: FieldRef, window: int) -> pd.Series:
    definition = field.definition

    def one(group: pd.DataFrame) -> float:
        values = group[definition.column].dropna().tail(window)
        minimum = int(window * 0.8)
        return float(values.iloc[-1] / values.max() - 1) if len(values) >= minimum else np.nan

    return _visible(context, field).groupby("symbol", observed=True).apply(one, include_groups=False)


def _return_stat(context: FactorContext, field: FieldRef, window: int, mode: str) -> pd.Series:
    definition = field.definition

    def one(group: pd.DataFrame) -> float:
        returns = group[definition.column].pct_change(fill_method=None).dropna().tail(window)
        if len(returns) < max(1, int(window * 0.8)):
            return np.nan
        if mode == "std":
            return float(returns.std(ddof=1))
        if mode == "upside":
            return float(returns.clip(lower=0).std(ddof=1))
        if mode == "downside":
            return float(returns.clip(upper=0).std(ddof=1))
        if mode == "max":
            return float(returns.max())
        return float(abs(returns.iloc[-1]))

    return _visible(context, field).groupby("symbol", observed=True).apply(one, include_groups=False)


def op_volatility(context: FactorContext, field: FieldRef, window: int) -> pd.Series:
    return _return_stat(context, field, window, "std")


def op_upside_vol(context: FactorContext, field: FieldRef, window: int) -> pd.Series:
    return _return_stat(context, field, window, "upside")


def op_downside_vol(context: FactorContext, field: FieldRef, window: int) -> pd.Series:
    return _return_stat(context, field, window, "downside")


def op_max_return(context: FactorContext, field: FieldRef, window: int) -> pd.Series:
    return _return_stat(context, field, window, "max")


def op_abs_return(context: FactorContext, field: FieldRef, window: int = 1) -> pd.Series:
    return _return_stat(context, field, window, "abs")


def op_idio(context: FactorContext, field: FieldRef, window: int, mode: str) -> pd.Series:
    visible = _visible(context, field).copy()
    visible["ret"] = visible.groupby("symbol", observed=True)[field.definition.column].pct_change(
        fill_method=None
    )
    if "market_return" not in visible:
        visible["market_return"] = visible.groupby("trade_date", observed=True)["ret"].transform("mean")

    def one(group: pd.DataFrame) -> float:
        sample = group[["ret", "market_return"]].dropna().tail(window)
        if len(sample) < int(window * 0.8) or sample["market_return"].var() == 0:
            return np.nan
        design = np.column_stack([np.ones(len(sample)), sample["market_return"]])
        residual = sample["ret"].to_numpy() - design @ np.linalg.lstsq(
            design, sample["ret"], rcond=None
        )[0]
        series = pd.Series(residual)
        return float(series.std(ddof=1) if mode == "vol" else series.skew())

    return visible.groupby("symbol", observed=True).apply(one, include_groups=False)


def op_idio_vol(context: FactorContext, field: FieldRef, window: int) -> pd.Series:
    return op_idio(context, field, window, "vol")


def op_idio_skew(context: FactorContext, field: FieldRef, window: int) -> pd.Series:
    return op_idio(context, field, window, "skew")


def _technical(context: FactorContext, output: str) -> pd.Series:
    from alphagym.factors.technical import compute_technical_features

    symbols = context.daily.loc[
        context.daily["trade_date"] <= context.signal_date, "symbol"
    ].drop_duplicates().sort_values()
    index = pd.MultiIndex.from_arrays(
        [symbols, np.repeat(context.signal_date, len(symbols))],
        names=["symbol", "signal_date"],
    )
    values = compute_technical_features(context.daily, index, [output])[output]
    return pd.Series(values.to_numpy(), index=symbols, name=output).rename_axis("symbol")


def _technical_named(name_builder: Callable[..., str]) -> Callable[..., pd.Series]:
    def operator(context: FactorContext, *args: object) -> pd.Series:
        scalar_args = [arg for arg in args if not isinstance(arg, FieldRef)]
        return _technical(context, name_builder(*scalar_args))

    return operator


def build_field_registry() -> FieldRegistry:
    registry = FieldRegistry()
    market_fields = {
        "open": "open", "high": "high", "low": "low", "close": "close",
        "adj_open": "adj_open", "adj_high": "adj_high", "adj_low": "adj_low",
        "adj_close": "adj_close", "volume": "volume", "amount": "amount",
        "turnover": "turnover", "float_market_cap": "float_market_cap",
        "index_return": "market_return",
    }
    for name, column in market_fields.items():
        registry.register(FieldDefinition(
            name=f"market.{name}", dataset="market", column=column,
            unit="CNY" if "price" in name or name in {"open", "high", "low", "close"} else None,
        ))
    for name in (
        "bps", "eps", "revenue", "ocfps", "net_profit", "roe", "gross_margin",
    ):
        registry.register(FieldDefinition(
            name=f"financial.{name}", dataset="financial", column=name,
            frequency="quarterly", event_time="stat_date", available_time="available_date",
        ))
    return registry


def build_operator_registry() -> OperatorRegistry:
    registry = OperatorRegistry()

    def add(name: str, function: Callable[..., object], description: str = "") -> None:
        registry.register(OperatorSpec(name, "1.0.0", function, description))

    core = {
        "ASOF": op_asof, "MRQ": op_mrq, "TTM": op_ttm, "YOY": op_yoy, "QOQ": op_qoq,
        "RETURN": op_return, "DIFF": op_diff, "ROLLING_MEAN": op_rolling_mean,
        "STD": op_std, "SUM": op_sum, "MIN": op_min, "MAX": op_max, "EWM": op_ewm,
        "SAFE_DIV": op_safe_div, "LOG": op_log, "ABS": op_abs, "CLIP": op_clip,
        "RANK": op_rank, "ZSCORE": op_zscore, "WINSORIZE": op_winsorize,
        "FEATURES": op_features, "MODEL_PREDICT": op_model_predict,
        "ENSEMBLE": op_ensemble, "TEXT_SCORE": op_text_score,
        "EMBEDDING": op_embedding, "DOCUMENT_AGG": op_document_agg,
        "STABILITY": op_stability, "TURNOVER_BIAS": op_turnover_bias,
        "AMIHUD": op_amihud, "HIGH_PROXIMITY": op_high_proximity,
        "VOLATILITY": op_volatility, "UPSIDE_VOL": op_upside_vol,
        "DOWNSIDE_VOL": op_downside_vol, "MAX_RETURN": op_max_return,
        "ABS_RETURN": op_abs_return, "IDIO_VOL": op_idio_vol, "IDIO_SKEW": op_idio_skew,
    }
    for name, function in core.items():
        add(name, function)

    technical: dict[str, Callable[..., pd.Series]] = {
        "BIAS": _technical_named(lambda window: f"BIAS_{window}D"),
        "TRIX": _technical_named(lambda window: f"TRIX_{window}D"),
        "TRIX_SIGNAL": _technical_named(lambda window, signal: f"TRIX_SIGNAL_{window}_{signal}"),
        "RSI": _technical_named(lambda window: f"RSI_{window}D"),
        "KDJ": _technical_named(lambda output, window: f"KDJ_{output}_{window}D"),
        "CCI": _technical_named(lambda window: f"CCI_{window}D"),
        "OBV": _technical_named(lambda: "OBV"),
        "MAOBV": _technical_named(lambda window: f"MAOBV_{window}D"),
        "BBI": _technical_named(lambda: "BBI_PCT"),
        "DPO": _technical_named(lambda window: f"DPO_{window}D_PCT"),
        "ROC": _technical_named(lambda window: f"ROC_{window}D"),
        "WR": _technical_named(lambda window: f"WR_{window}D"),
        "PSY": _technical_named(lambda window: f"PSY_{window}D"),
        "PSYMA": _technical_named(lambda window, signal: f"PSYMA_{window}_{signal}"),
        "STREAK": _technical_named(lambda direction: f"{direction}_STREAK"),
        "DAYS_SINCE": _technical_named(lambda side, window: f"DAYS_SINCE_{side}_{window}D"),
        "MASS": _technical_named(lambda fast, window: f"MASS_{fast}_{window}"),
        "MASS_MA": _technical_named(lambda fast, window, signal: f"MASS_MA_{fast}_{window}_{signal}"),
        "MFI": _technical_named(lambda window: f"MFI_{window}D"),
        "VR": _technical_named(lambda window: f"VR_{window}D"),
        "EMV": _technical_named(lambda window: f"EMV_{window}D"),
        "MAEMV": _technical_named(lambda window, signal: f"MAEMV_{window}_{signal}"),
        "AR": _technical_named(lambda window: f"AR_{window}D"),
        "BR": _technical_named(lambda window: f"BR_{window}D"),
        "CR": _technical_named(lambda window: f"CR_{window}D"),
        "DMI": _technical_named(lambda output, window, signal: f"DMI_{output}_{window}_{signal}"),
        "MACD": _technical_named(
            lambda output, fast, slow, signal: f"MACD_{output}_{fast}_{slow}_{signal}_PCT"
        ),
        "ATR": _technical_named(lambda window: f"ATR_{window}D_PCT"),
        "BBANDS": _technical_named(lambda output, window, width: f"BBANDS_{output}_{window}_{width}"),
        "DONCHIAN": _technical_named(
            lambda output, window: f"DONCHIAN_{output}_{window}D"
            if output in {"POSITION", "WIDTH"} else f"DONCHIAN_{output}_{window}D_PCT"
        ),
        "ASI": _technical_named(lambda window: f"ASI_{window}D"),
        "ASIT": _technical_named(lambda window, signal: f"ASIT_{window}_{signal}"),
        "BOLL": _technical_named(lambda output, window: f"BOLL_{output}_{window}D_PCT"),
        "DFMA": _technical_named(
            lambda output, fast, slow, signal=0: f"DFMA_{output}_{fast}_{slow}_PCT"
            if output == "DIF" else f"DFMA_{output}_{fast}_{slow}_{signal}_PCT"
        ),
        "MADPO": _technical_named(lambda window, shift, signal: f"MADPO_{window}_{shift}_{signal}_PCT"),
        "EXPMA": _technical_named(lambda window: f"EXPMA_{window}D_PCT"),
        "KELTNER": _technical_named(lambda output, window, atr: f"KELTNER_{output}_{window}_{atr}_PCT"),
        "MTMMA": _technical_named(lambda window, signal: f"MTMMA_{window}_{signal}_PCT"),
        "MAROC": _technical_named(lambda window, signal: f"MAROC_{window}_{signal}"),
    }
    for name, function in technical.items():
        add(name, function)
    return registry
