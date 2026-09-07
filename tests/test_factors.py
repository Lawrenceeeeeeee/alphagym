from __future__ import annotations

import numpy as np
import pandas as pd

from alphagym.factors import REGISTRY, FactorContext
from alphagym.factors.compute import compute_factors
from alphagym.factors.technical import (
    BLOCKED_SOURCE_FACTORS,
    SOURCE_ALIASES,
    TECHNICAL_FACTOR_NAMES,
    compute_technical_features,
)


def test_exactly_40_preregistered_factors() -> None:
    assert len(REGISTRY) == 169
    assert len({spec.name for spec in REGISTRY.list()}) == 169
    assert all(spec.hypothesis_id and spec.formula_version for spec in REGISTRY.list())


def test_all_40_calculators_preserve_symbol_index(bundle) -> None:
    context = FactorContext(pd.Timestamp("2025-02-24"), bundle.daily, bundle.fundamentals)
    for spec in REGISTRY.list():
        result = spec.calculator(context)
        assert isinstance(result, pd.Series), spec.name
        assert result.index.name == "symbol", spec.name


def test_return_factor_gold_value(bundle) -> None:
    context = FactorContext(bundle.daily["trade_date"].max(), bundle.daily, bundle.fundamentals)
    symbol = "000001.SZ"
    close = bundle.daily[bundle.daily["symbol"] == symbol]["adj_close"]
    expected = close.iloc[-1] / close.iloc[-61] - 1
    assert np.isclose(REGISTRY.get("MOMENTUM_60D").calculator(context).loc[symbol], expected)


def test_reversal_has_locked_positive_orientation(bundle) -> None:
    context = FactorContext(bundle.daily["trade_date"].max(), bundle.daily, bundle.fundamentals)
    momentum = REGISTRY.get("MOMENTUM_60D").calculator(context)
    reversal = REGISTRY.get("REVERSAL_5D").calculator(context)
    assert (momentum > 0).all()
    assert (reversal < 0).all()


def test_future_prices_do_not_change_historical_factors(bundle) -> None:
    signal = pd.Timestamp("2024-10-01")
    before = REGISTRY.get("VOLATILITY_60D").calculator(FactorContext(signal, bundle.daily, bundle.fundamentals))
    future = bundle.daily.iloc[[0]].copy()
    future["trade_date"] = pd.Timestamp("2026-01-05")
    future["adj_close"] = 1_000_000
    modified = pd.concat([bundle.daily, future], ignore_index=True)
    after = REGISTRY.get("VOLATILITY_60D").calculator(FactorContext(signal, modified, bundle.fundamentals))
    pd.testing.assert_series_equal(before, after)


def test_compute_output_contract(bundle) -> None:
    result = compute_factors(bundle.daily, bundle.fundamentals, [bundle.daily["trade_date"].max()], ["EP_TTM"])
    assert list(result) == ["signal_date", "symbol", "factor_name", "raw_value", "neutralized_value", "available_date"]
    assert result["neutralized_value"].isna().all()


def test_workbook_factor_inventory_is_auditable() -> None:
    assert len(TECHNICAL_FACTOR_NAMES) == 98
    assert len(SOURCE_ALIASES) == 8
    assert len(BLOCKED_SOURCE_FACTORS) == 4
    assert set(TECHNICAL_FACTOR_NAMES) <= {spec.name for spec in REGISTRY.list()}
    assert set(BLOCKED_SOURCE_FACTORS).isdisjoint(TECHNICAL_FACTOR_NAMES)


def test_technical_price_factors_match_gold_values(bundle) -> None:
    signal_date = bundle.daily["trade_date"].max()
    symbols = bundle.daily["symbol"].drop_duplicates().sort_values()
    signal_index = pd.MultiIndex.from_arrays(
        [symbols, np.repeat(signal_date, len(symbols))],
        names=["symbol", "signal_date"],
    )
    result = compute_technical_features(
        bundle.daily, signal_index, ["BIAS_5D", "ROC_12D"]
    ).set_axis(symbols)
    sample = bundle.daily[bundle.daily["symbol"] == symbols.iloc[0]]["adj_close"]
    assert np.isclose(result.loc[symbols.iloc[0], "BIAS_5D"], sample.iloc[-1] / sample.iloc[-5:].mean() - 1)
    assert np.isclose(result.loc[symbols.iloc[0], "ROC_12D"], sample.iloc[-1] / sample.iloc[-13] - 1)


def test_future_prices_do_not_change_historical_technical_factor(bundle) -> None:
    signal = pd.Timestamp("2024-10-01")
    context = FactorContext(signal, bundle.daily, bundle.fundamentals)
    before = REGISTRY.get("RSI_12D").calculator(context)
    future = bundle.daily.iloc[[0]].copy()
    future["trade_date"] = pd.Timestamp("2026-01-05")
    future["adj_close"] = 1_000_000
    modified = pd.concat([bundle.daily, future], ignore_index=True)
    after = REGISTRY.get("RSI_12D").calculator(
        FactorContext(signal, modified, bundle.fundamentals)
    )
    pd.testing.assert_series_equal(before, after)
