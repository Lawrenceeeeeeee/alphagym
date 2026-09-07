from __future__ import annotations

import re

import numpy as np
import pandas as pd

from alphagym.factor_dsl import FormulaCompiler, FormulaEngine
from alphagym.factor_operators import build_field_registry, build_operator_registry
from alphagym.factors.base import FactorContext, FactorDefinition, FactorRegistry, FactorSpec
from alphagym.factors.technical import TECHNICAL_METADATA


def _technical_formula(name: str) -> str:
    price = "market.adj_close"
    high_low_close = "market.adj_high, market.adj_low, market.adj_close"
    if match := re.fullmatch(r"BIAS_(\d+)D", name):
        return f"=BIAS({price}, {match[1]})"
    if match := re.fullmatch(r"TRIX_(\d+)D", name):
        return f"=TRIX({price}, {match[1]})"
    if match := re.fullmatch(r"TRIX_SIGNAL_(\d+)_(\d+)", name):
        return f"=TRIX_SIGNAL({price}, {match[1]}, {match[2]})"
    if match := re.fullmatch(r"RSI_(\d+)D", name):
        return f"=RSI({price}, {match[1]})"
    if match := re.fullmatch(r"KDJ_(RSV|J|K|D)_(\d+)D", name):
        return f'=KDJ({high_low_close}, "{match[1]}", {match[2]})'
    if match := re.fullmatch(r"CCI_(\d+)D", name):
        return f"=CCI({high_low_close}, {match[1]})"
    if name == "OBV":
        return f"=OBV({price}, market.volume)"
    if match := re.fullmatch(r"MAOBV_(\d+)D", name):
        return f"=MAOBV({price}, market.volume, {match[1]})"
    if name == "BBI_PCT":
        return f"=BBI({price})"
    if match := re.fullmatch(r"DPO_(\d+)D_PCT", name):
        return f"=DPO({price}, {match[1]})"
    if match := re.fullmatch(r"ROC_(\d+)D", name):
        return f"=ROC({price}, {match[1]})"
    if match := re.fullmatch(r"WR_(\d+)D", name):
        return f"=WR({high_low_close}, {match[1]})"
    if match := re.fullmatch(r"PSY_(\d+)D", name):
        return f"=PSY({price}, {match[1]})"
    if match := re.fullmatch(r"PSYMA_(\d+)_(\d+)", name):
        return f"=PSYMA({price}, {match[1]}, {match[2]})"
    if match := re.fullmatch(r"(UP|DOWN)_STREAK", name):
        return f'=STREAK({price}, "{match[1]}")'
    if match := re.fullmatch(r"DAYS_SINCE_(HIGH|LOW)_(\d+)D", name):
        field = "market.adj_high" if match[1] == "HIGH" else "market.adj_low"
        return f'=DAYS_SINCE({field}, "{match[1]}", {match[2]})'
    if match := re.fullmatch(r"MASS_(\d+)_(\d+)", name):
        return f"=MASS(market.adj_high, market.adj_low, {match[1]}, {match[2]})"
    if match := re.fullmatch(r"MASS_MA_(\d+)_(\d+)_(\d+)", name):
        return (
            f"=MASS_MA(market.adj_high, market.adj_low, "
            f"{match[1]}, {match[2]}, {match[3]})"
        )
    if match := re.fullmatch(r"MFI_(\d+)D", name):
        return f"=MFI({high_low_close}, market.volume, {match[1]})"
    if match := re.fullmatch(r"VR_(\d+)D", name):
        return f"=VR({price}, market.volume, {match[1]})"
    if match := re.fullmatch(r"EMV_(\d+)D", name):
        return f"=EMV(market.adj_high, market.adj_low, market.volume, {match[1]})"
    if match := re.fullmatch(r"MAEMV_(\d+)_(\d+)", name):
        return (
            f"=MAEMV(market.adj_high, market.adj_low, market.volume, "
            f"{match[1]}, {match[2]})"
        )
    if match := re.fullmatch(r"(AR|BR|CR)_(\d+)D", name):
        fields = "market.adj_open, market.adj_high, market.adj_low, market.adj_close"
        return f"={match[1]}({fields}, {match[2]})"
    if match := re.fullmatch(r"DMI_(PDI|MDI|ADX|ADXR)_(\d+)_(\d+)", name):
        return f'=DMI({high_low_close}, "{match[1]}", {match[2]}, {match[3]})'
    if match := re.fullmatch(r"MACD_(DIF|DEA|HIST)_(\d+)_(\d+)_(\d+)_PCT", name):
        return (
            f'=MACD({price}, "{match[1]}", {match[2]}, {match[3]}, {match[4]})'
        )
    if match := re.fullmatch(r"ATR_(\d+)D_PCT", name):
        return f"=ATR({high_low_close}, {match[1]})"
    if match := re.fullmatch(r"BBANDS_(WIDTH|POSITION)_(\d+)_(\d+)", name):
        return f'=BBANDS({price}, "{match[1]}", {match[2]}, {match[3]})'
    if match := re.fullmatch(r"DONCHIAN_(POSITION|WIDTH)_(\d+)D", name):
        return f'=DONCHIAN({high_low_close}, "{match[1]}", {match[2]})'
    if match := re.fullmatch(r"DONCHIAN_(LOWER|MID|UPPER)_(\d+)D_PCT", name):
        return f'=DONCHIAN({high_low_close}, "{match[1]}", {match[2]})'
    if match := re.fullmatch(r"ASI_(\d+)D", name):
        fields = "market.adj_open, market.adj_high, market.adj_low, market.adj_close"
        return f"=ASI({fields}, {match[1]})"
    if match := re.fullmatch(r"ASIT_(\d+)_(\d+)", name):
        fields = "market.adj_open, market.adj_high, market.adj_low, market.adj_close"
        return f"=ASIT({fields}, {match[1]}, {match[2]})"
    if match := re.fullmatch(r"BOLL_(LOWER|MID|UPPER)_(\d+)D_PCT", name):
        return f'=BOLL({price}, "{match[1]}", {match[2]})'
    if match := re.fullmatch(r"DFMA_(DIF)_(\d+)_(\d+)_PCT", name):
        return f'=DFMA({price}, "{match[1]}", {match[2]}, {match[3]})'
    if match := re.fullmatch(r"DFMA_(SIGNAL)_(\d+)_(\d+)_(\d+)_PCT", name):
        return (
            f'=DFMA({price}, "{match[1]}", {match[2]}, {match[3]}, {match[4]})'
        )
    if match := re.fullmatch(r"MADPO_(\d+)_(\d+)_(\d+)_PCT", name):
        return f"=MADPO({price}, {match[1]}, {match[2]}, {match[3]})"
    if match := re.fullmatch(r"EXPMA_(\d+)D_PCT", name):
        return f"=EXPMA({price}, {match[1]})"
    if match := re.fullmatch(r"KELTNER_(LOWER|MID|UPPER)_(\d+)_(\d+)_PCT", name):
        return f'=KELTNER({high_low_close}, "{match[1]}", {match[2]}, {match[3]})'
    if match := re.fullmatch(r"MTMMA_(\d+)_(\d+)_PCT", name):
        return f"=MTMMA({price}, {match[1]}, {match[2]})"
    if match := re.fullmatch(r"MAROC_(\d+)_(\d+)", name):
        return f"=MAROC({price}, {match[1]}, {match[2]})"
    raise KeyError(f"no DSL formula for technical factor: {name}")


def seed_definitions() -> tuple[FactorDefinition, ...]:
    definitions: list[FactorDefinition] = []

    def add(
        name: str, hypothesis: str, family: str, formula: str,
        direction: str = "unknown", description: str = "",
        formula_version: str = "1.0",
    ) -> None:
        definitions.append(FactorDefinition(
            factor_id=name, name=name, formula=formula, hypothesis_id=hypothesis,
            family=family, formula_version=formula_version,
            expected_direction=direction, description=description or hypothesis.replace("_", " "),
        ))

    valuation = {
        "EP_TTM": '=SAFE_DIV(TTM(financial.eps), ASOF(market.close))',
        "BP_MRQ": '=SAFE_DIV(MRQ(financial.bps), ASOF(market.close))',
        "SP_TTM": '=SAFE_DIV(TTM(financial.revenue), ASOF(market.float_market_cap))',
        "CFP_TTM": '=SAFE_DIV(TTM(financial.ocfps), ASOF(market.close))',
    }
    for name, formula in valuation.items():
        add(name, f"value_{name.lower().split('_')[0]}", "value", formula, "positive")
    for name, field, difference in [
        ("REVENUE_YOY", "revenue", False), ("NET_PROFIT_YOY", "net_profit", False),
        ("EPS_YOY", "eps", False), ("OCFPS_YOY", "ocfps", False),
        ("ROE_YOY_CHANGE", "roe", True), ("GROSS_MARGIN_YOY_CHANGE", "gross_margin", True),
    ]:
        add(name, f"growth_{field}", "growth",
            f"=YOY(financial.{field}, difference={difference})", "positive")
    for name, window, skip, reverse in [
        ("REVERSAL_5D", 5, 0, True), ("REVERSAL_20D", 20, 0, True),
        ("MOMENTUM_60D", 60, 0, False), ("MOMENTUM_120D", 120, 0, False),
        ("MOMENTUM_120D_SKIP20", 120, 20, False),
        ("MOMENTUM_252D_SKIP20", 252, 20, False),
    ]:
        sign = "-" if reverse else ""
        add(name, f"price_{'reversal' if reverse else 'momentum'}", "momentum",
            f"={sign}RETURN(market.adj_close, {window}, skip={skip})", "positive")
    add("HIGH_252D_PROXIMITY", "price_high_proximity", "momentum",
        "=HIGH_PROXIMITY(market.adj_close, 252)", "positive")
    for window in (20, 60, 120):
        add(f"TURNOVER_MEAN_{window}D", "turnover_level", "liquidity",
            f"=ROLLING_MEAN(market.turnover, {window})", "negative")
    for short in (20, 60):
        add(f"TURNOVER_BIAS_{short}_480D", "turnover_bias", "liquidity",
            f"=TURNOVER_BIAS(market.turnover, {short}, 480)", "negative")
        add(f"TURNOVER_VOL_{short}D", "turnover_volatility", "liquidity",
            f"=STD(market.turnover, {short})", "negative")
    add("AMIHUD_20D", "illiquidity_amihud", "liquidity",
        "=AMIHUD(market.adj_close, market.amount, 20, minimum=15)", "negative")
    for window in (20, 60, 120):
        add(f"VOLATILITY_{window}D", "return_volatility", "risk",
            f"=VOLATILITY(market.adj_close, {window})", "negative")
    add("UPSIDE_VOL_60D", "asymmetric_volatility", "risk",
        "=UPSIDE_VOL(market.adj_close, 60)", "negative")
    add("DOWNSIDE_VOL_60D", "asymmetric_volatility", "risk",
        "=DOWNSIDE_VOL(market.adj_close, 60)", "negative")
    add("IDIO_VOL_60D", "capm_idiosyncratic_risk", "risk",
        "=IDIO_VOL(market.adj_close, 60)", "negative")
    add("IDIO_SKEW_60D", "capm_idiosyncratic_skew", "risk",
        "=IDIO_SKEW(market.adj_close, 60)", "positive")
    add("MAX_RETURN_20D", "lottery_max_return", "risk",
        "=MAX_RETURN(market.adj_close, 20)", "negative")
    for window in (1, 2, 3, 7, 10):
        add(f"REVERSAL_{window}D", "price_reversal", "momentum",
            f"=-RETURN(market.adj_close, {window})", "positive")
    for window in (20, 60, 120):
        add(f"HIGH_{window}D_PROXIMITY", "price_high_proximity", "momentum",
            f"=HIGH_PROXIMITY(market.adj_close, {window})", "positive")
    for window in (1, 5, 7, 10):
        add(f"TURNOVER_MEAN_{window}D", "turnover_level", "liquidity",
            f"=ROLLING_MEAN(market.turnover, {window})", "negative")
    for window in (5, 7, 10):
        add(f"TURNOVER_VOL_{window}D", "turnover_volatility", "liquidity",
            f"=STD(market.turnover, {window})", "negative")
        minimum = max(3, int(np.ceil(window * 0.8)))
        add(f"AMIHUD_{window}D", "illiquidity_amihud", "liquidity",
            f"=AMIHUD(market.adj_close, market.amount, {window}, minimum={minimum})",
            "negative")
    add("ABS_RETURN_1D", "return_volatility", "risk",
        "=ABS_RETURN(market.adj_close, 1)", "negative")
    for window in (2, 3, 5, 7, 10):
        add(f"VOLATILITY_{window}D", "return_volatility", "risk",
            f"=VOLATILITY(market.adj_close, {window})", "negative")
    for window in (10, 20):
        add(f"UPSIDE_VOL_{window}D", "asymmetric_volatility", "risk",
            f"=UPSIDE_VOL(market.adj_close, {window})", "negative")
        add(f"DOWNSIDE_VOL_{window}D", "asymmetric_volatility", "risk",
            f"=DOWNSIDE_VOL(market.adj_close, {window})", "negative")
    for window in (5, 7, 10):
        add(f"MAX_RETURN_{window}D", "lottery_max_return", "risk",
            f"=MAX_RETURN(market.adj_close, {window})", "negative")
    add("ROE", "quality_roe", "quality", "=ASOF(financial.roe)", "positive")
    add("GROSS_MARGIN", "quality_gross_margin", "quality",
        "=ASOF(financial.gross_margin)", "positive")
    add("NET_MARGIN_TTM", "quality_net_margin", "quality",
        "=SAFE_DIV(TTM(financial.net_profit), TTM(financial.revenue))", "positive")
    add("OCF_TO_EARNINGS", "quality_cash_conversion", "quality",
        "=SAFE_DIV(TTM(financial.ocfps), TTM(financial.eps))", "positive")
    add("ROE_STABILITY_8Q", "quality_roe_stability", "quality",
        "=STABILITY(financial.roe, 8, 6)", "positive")
    add("GROSS_MARGIN_STABILITY_8Q", "quality_margin_stability", "quality",
        "=STABILITY(financial.gross_margin, 8, 6)", "positive")
    add("EARNINGS_GROWTH_STABILITY_8Q", "quality_growth_stability", "quality",
        "=STABILITY(financial.net_profit, 9, 7, growth=True)", "positive")
    for name, (hypothesis, _inputs, _lookback, _minimum) in TECHNICAL_METADATA.items():
        add(name, hypothesis, "technical", _technical_formula(name), "unknown",
            formula_version="factor_list_xlsx_v1_adjusted_dimensionless")
    if len(definitions) != 169:
        raise AssertionError(f"expected 169 formula definitions, got {len(definitions)}")
    return tuple(definitions)


def build_registry() -> FactorRegistry:
    fields = build_field_registry()
    operators = build_operator_registry()
    engine = FormulaEngine(fields, operators)
    registry = FactorRegistry(engine)

    def current_revision(factor_id: str) -> str:
        return registry.get(factor_id).revision_id

    compiler = FormulaCompiler(fields, operators, factor_resolver=current_revision)
    for definition in seed_definitions():
        compiled = compiler.compile(definition.formula)
        revision_id = f"{definition.factor_id}:{definition.formula_version}:{compiled.definition_hash[:16]}"
        dependencies = dict(compiled.factor_dependencies)

        def calculate(
            context: FactorContext, *, item: object = compiled,
            locked: dict[str, str] = dependencies,
        ) -> pd.Series:
            return engine.evaluate(
                item, context,
                factor_resolver=lambda factor_id, child_context: registry.compute_revision(
                    locked[factor_id], child_context
                ),
            )

        registry.register(FactorSpec(
            name=definition.name, factor_id=definition.factor_id,
            revision_id=revision_id, hypothesis_id=definition.hypothesis_id,
            family=definition.family, formula_version=definition.formula_version,
            input_fields=compiled.fields, lookback_days=compiled.lookback_days,
            min_observations=compiled.min_observations,
            availability_rule="all inputs available_at/event_time <= signal_date",
            expected_direction=definition.expected_direction, calculator=calculate,
            formula=compiled.source, description=definition.description,
            tags=definition.tags, status=definition.status,
            definition_hash=compiled.definition_hash,
        ))
    return registry


REGISTRY = build_registry()
