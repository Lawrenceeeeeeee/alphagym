from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlquant.factors import REGISTRY
from mlquant.local_v0 import (
    FUNDAMENTAL_FACTORS,
    PRICE_VOLUME_FACTORS,
    SHORT_HORIZON_FACTORS,
    LocalSourcePaths,
    _locked_direction_signs,
    _selected_development_factors,
    apply_backward_adjustments,
    vendor_symbol,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [("sh.600000", "600000.SH"), ("000001.SZ", "000001.SZ"), ("600519", "600519.SH")],
)
def test_vendor_symbol(source: str, expected: str) -> None:
    assert vendor_symbol(source) == expected


def test_local_v0_registers_39_available_factors() -> None:
    names = PRICE_VOLUME_FACTORS + FUNDAMENTAL_FACTORS
    assert len(names) == 39
    assert len(set(names)) == 39
    assert "SP_TTM" not in names


def test_short_horizon_batch_is_registered_and_unique() -> None:
    assert len(SHORT_HORIZON_FACTORS) == 41
    assert len(set(SHORT_HORIZON_FACTORS)) == 41
    assert set(SHORT_HORIZON_FACTORS) <= {spec.name for spec in REGISTRY.list()}


def test_expanded_selection_keeps_one_window_per_hypothesis(tmp_path) -> None:
    summary = pd.DataFrame(
        {
            "index_code": "000300.SH",
            "period": "development",
            "value_type": "raw",
            "orientation": "original",
            "factor_name": [
                "REVERSAL_1D",
                "REVERSAL_5D",
                "REVERSAL_7D",
                "MOMENTUM_60D",
            ],
            "rank_ic": [0.04, 0.03, 0.05, 0.02],
            "bh_q_value": [0.01, 0.01, 0.001, 0.02],
            "development_selection_q_value": [0.01, 0.01, 0.20, 0.02],
            "coverage": [0.99, 0.99, 0.99, 0.99],
        }
    )
    paths = LocalSourcePaths(tmp_path, tmp_path, tmp_path)
    selected = _selected_development_factors(
        paths,
        "000300.SH",
        summary=summary,
        collapse_hypotheses=True,
    )
    assert selected == ["MOMENTUM_60D", "REVERSAL_1D"]


def test_adjustment_asof_includes_events_before_daily_slice() -> None:
    daily = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "symbol": "000001.SZ",
            "close": [10.0, 10.1, 9.8],
        }
    )
    adjustments = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2019-06-01", "2020-01-05"]),
            "symbol": "000001.SZ",
            "adjust_factor": [2.0, 2.2],
        }
    )
    result = apply_backward_adjustments(daily, adjustments)
    assert result["adjust_factor"].tolist() == [2.0, 2.0, 2.2]


def test_unknown_technical_direction_is_locked_from_development_only() -> None:
    summary = pd.DataFrame(
        {
            "index_code": ["000300.SH", "000300.SH", "000905.SH"],
            "period": ["development", "test", "development"],
            "value_type": "raw",
            "orientation": "original",
            "factor_name": "RSI_12D",
            "rank_ic": [-0.03, 0.20, 0.04],
        }
    )
    assert _locked_direction_signs(summary, "000300.SH")["RSI_12D"] == -1.0
    assert _locked_direction_signs(summary, "000905.SH")["RSI_12D"] == 1.0


def test_technical_report_contains_strategy_and_complete_backtest_metrics() -> None:
    report = (
        Path(__file__).parents[1] / "research" / "series" / "equity_v0_technical_factors.md"
    ).read_text(encoding="utf-8")
    required = {
        "纯多头",
        "不实际做空",
        "年化收益",
        "年化波动",
        "Sharpe",
        "Sortino",
        "最大回撤",
        "Calmar",
        "净超额IR",
        "月均换手",
        "月胜率",
        "季胜率",
        "年胜率",
    }
    missing = {item for item in required if item not in report}
    assert not missing
