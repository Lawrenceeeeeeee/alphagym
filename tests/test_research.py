from __future__ import annotations

import numpy as np
import pandas as pd

from alphagym.research import (
    benjamini_hochberg,
    evaluate_factor_batch,
    evaluate_factor_suite,
    huatai_industry_layers,
    neutralize_cross_section,
)


def test_neutralization_removes_industry_and_size_exposure() -> None:
    rng = np.random.default_rng(7)
    count = 120
    industry = np.repeat(["A", "B", "C"], count // 3)
    size = np.exp(rng.normal(20, 1, count))
    raw = np.log(size) * .7 + pd.Series(industry).map({"A": -2, "B": 0, "C": 2}) + rng.normal(0, .1, count)
    result = neutralize_cross_section(pd.DataFrame({"raw_value": raw, "industry_code": industry, "float_market_cap": size}))
    residual = result["neutralized_value"]
    weights = size / np.median(size)
    weighted_size_exposure = np.sum(weights * residual * np.log(size)) / np.sum(weights)
    assert abs(weighted_size_exposure) < 1e-8
    weighted_industry = result.assign(residual=residual, weight=weights).groupby("industry_code").apply(
        lambda group: np.average(group["residual"], weights=group["weight"]),
        include_groups=False,
    )
    assert weighted_industry.abs().max() < 1e-8


def test_huatai_layers_match_industry_weights_and_split_boundaries() -> None:
    frame = pd.DataFrame({
        "symbol": [f"A{i}" for i in range(3)] + [f"B{i}" for i in range(7)],
        "industry_code": ["A"] * 3 + ["B"] * 7,
        "factor_value": range(10),
        "benchmark_weight": [0.6 / 3] * 3 + [0.4 / 7] * 7,
    })
    layers = huatai_industry_layers(frame)
    assert np.allclose(layers.groupby("layer")["target_weight"].sum(), 1)
    exposure = layers.groupby(["layer", "industry_code"])["target_weight"].sum().unstack()
    assert np.allclose(exposure["A"], .6)
    assert np.allclose(exposure["B"], .4)
    assert (layers["boundary_fraction"] < 1).any()
    assert (layers["target_weight"] >= 0).all()
    assert layers.loc[layers["symbol"] == "A2", "layer"].min() == 1
    assert layers.loc[layers["symbol"] == "A0", "layer"].max() == 5


def test_bh_is_monotone_and_bounded() -> None:
    adjusted = benjamini_hochberg(pd.Series([.01, .04, .03, .8]))
    assert adjusted.between(0, 1).all()
    ordered = pd.DataFrame({"p": [.01, .04, .03, .8], "q": adjusted}).sort_values("p")
    assert ordered["q"].is_monotonic_increasing


def test_factor_suite_discloses_raw_neutralized_and_both_directions() -> None:
    panel = pd.DataFrame(
        [
            {"signal_date": date, "symbol": symbol, "raw_value": value,
             "neutralized_value": value / 2, "forward_return": value / 100}
            for date in pd.to_datetime(["2020-01-31", "2020-02-28", "2021-01-29", "2024-01-31"])
            for symbol, value in zip(list("ABCDE"), range(1, 6), strict=True)
        ]
    )
    monthly, summary = evaluate_factor_suite(panel)
    assert set(monthly["value_type"]) == {"raw", "neutralized"}
    assert set(monthly["orientation"]) == {"original", "reversed"}
    assert {"development", "validation", "test"}.issubset(set(summary["period"]))
    assert "rank_turnover" in monthly
    original = monthly[
        (monthly["value_type"] == "raw") & (monthly["orientation"] == "original")
    ]
    assert (original["group_order"] == "descending").all()
    assert (original["group_1"] > original["group_5"]).all()


def test_batch_applies_one_bh_correction() -> None:
    panel = pd.DataFrame(
        [
            {"factor_name": factor, "signal_date": date, "symbol": symbol,
             "raw_value": value, "neutralized_value": value,
             "forward_return": (value if date.month % 2 else 6 - value) / 100}
            for factor in ["A", "B"]
            for date in pd.date_range("2019-01-31", periods=6, freq="ME")
            for symbol, value in zip(list("ABCDE"), range(1, 6), strict=True)
        ]
    )
    _, summary = evaluate_factor_batch(panel)
    assert summary["bh_q_value"].between(0, 1).all()
