from __future__ import annotations

import numpy as np
import pandas as pd

from alphagym.cli import build_parser
from alphagym.crypto_style_rotation import (
    RotationSpec,
    UniverseSpec,
    _allocator_weights,
    build_point_in_time_universe,
    build_style_positions,
    compute_style_features,
    evaluate_rotation,
    freeze_style_directions,
    market_state_attribution,
    style_portfolio_returns,
)


def rotation_spec(**overrides):
    values = {
        "universe": UniverseSpec(
            size=8, minimum_listing_age_days=30, liquidity_lookback_days=30,
            minimum_coverage=0.8, instruments=tuple(f"C{i}-USDT-SWAP" for i in range(8)),
        ),
        "candidate_windows_weeks": (4, 8, 12),
        "fee_bps_per_side": (2.0, 5.0, 10.0),
        "primary_fee_bps_per_side": 5.0,
        "splits": {
            "development": ("2024-01-01", "2024-03-31"),
            "validation": ("2024-04-01", "2024-06-30"),
            "test": ("2024-07-01", "2024-09-30"),
            "monitoring": ("2024-10-01", None),
        },
        "bootstrap_samples": 100,
    }
    values.update(overrides)
    return RotationSpec(**values)


def sample_panel(end="2024-11-30", seed=9):
    times = pd.date_range("2023-10-01", end, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    rows = []
    for asset in range(8):
        innovations = rng.normal(asset * 1e-6, 0.006 + asset * 0.0002, len(times))
        close = 100 * np.exp(np.cumsum(innovations))
        volume = (2_000_000 + asset * 400_000) * np.exp(rng.normal(0, 0.15, len(times)))
        funding = np.zeros(len(times))
        funding[::2] = (asset - 3.5) * 1e-6
        for number, ts in enumerate(times):
            rows.append({
                "ts": ts, "instrument": f"C{asset}-USDT-SWAP",
                "open": close[number] * np.exp(-innovations[number]),
                "high": close[number] * 1.004, "low": close[number] * 0.996,
                "close": close[number], "volume_quote": volume[number],
                "funding_event": funding[number], "funding_rate": (asset - 3.5) * 1e-6,
                "listing_time": pd.Timestamp("2020-01-01", tz="UTC"),
            })
    return pd.DataFrame(rows)


def test_cli_exposes_style_rotation_spec_entrypoint():
    args = build_parser().parse_args([
        "crypto-style-rotation", "--root", "workspace", "--spec", "rotation.yaml"
    ])
    assert args.spec == "rotation.yaml"
    assert args.func.__name__ == "_cmd_crypto_style_rotation"


def test_weekly_features_use_next_open_and_point_in_time_universe():
    panel = sample_panel(end="2024-04-30")
    spec = rotation_spec()
    features = compute_style_features(panel, spec)
    selected, audit = build_point_in_time_universe(features, spec)
    assert not selected.empty
    assert selected.groupby("ts")["instrument"].nunique().max() == 8
    assert set(audit["exclusion_reason"]) <= {
        "eligible", "listing_age", "coverage", "liquidity", "style_history",
        "outside_liquid_top_n",
    }
    row = features[features["instrument"].eq("C0-USDT-SWAP")].dropna(
        subset=["forward_price_return"]
    ).iloc[-1]
    instrument = panel[panel["instrument"].eq("C0-USDT-SWAP")].sort_values("ts")
    location = instrument.index[instrument["ts"].eq(row["ts"])][0]
    ordinal = instrument.index.get_loc(location)
    expected = np.log(
        instrument.iloc[ordinal + 43]["open"] / instrument.iloc[ordinal + 1]["open"]
    )
    assert row["forward_price_return"] == expected


def test_style_books_are_market_neutral_and_gross_one():
    spec = rotation_spec()
    features = compute_style_features(sample_panel(end="2024-07-31"), spec)
    selected, _ = build_point_in_time_universe(features, spec)
    directions = freeze_style_directions(selected, spec)
    positions = build_style_positions(selected, directions, spec)
    totals = positions.groupby(["ts", "style"])["position"].agg(
        net="sum", gross=lambda x: x.abs().sum()
    )
    assert np.allclose(totals["net"], 0)
    assert np.allclose(totals["gross"], 1)


def test_positive_funding_is_paid_by_longs_and_received_by_shorts():
    positions = pd.DataFrame([
        {"ts": pd.Timestamp("2024-01-01", tz="UTC"), "style": "carry",
         "instrument": "LONG", "position": 0.5, "forward_price_return": 0.0,
         "forward_funding": 0.01},
        {"ts": pd.Timestamp("2024-01-01", tz="UTC"), "style": "carry",
         "instrument": "SHORT", "position": -0.5, "forward_price_return": 0.0,
         "forward_funding": 0.02},
    ])
    returns, _ = style_portfolio_returns(positions, fee_bps=0)
    assert returns.iloc[0]["funding_return"] == 0.005


def test_window_and_directions_ignore_test_mutation():
    spec = rotation_spec()
    panel = sample_panel()
    original = evaluate_rotation(panel, spec)
    changed = panel.copy()
    test = changed["ts"].between(
        pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp("2024-09-30 23:59", tz="UTC")
    )
    changed.loc[test, ["open", "high", "low", "close"]] *= np.exp(
        np.random.default_rng(123).normal(0, 0.25, test.sum())[:, None]
    )
    mutated = evaluate_rotation(changed, spec)
    assert original["directions"] == mutated["directions"]
    assert original["selected_window"] == mutated["selected_window"]


def test_factor_momentum_allocator_rotates_and_holds_cash_when_all_scores_negative():
    times = pd.date_range("2024-01-01", periods=20, freq="7D", tz="UTC")
    rows = []
    for style in ("momentum", "reversal"):
        values = np.full(20, 0.02 if style == "momentum" else -0.01)
        for ts, value in zip(times, values, strict=True):
            rows.append({"ts": ts, "style": style, "net_return": value})
    allocations = _allocator_weights(pd.DataFrame(rows), 4, ("momentum", "reversal"))
    assert allocations.iloc[-1]["momentum"] == 1
    assert allocations.iloc[-1]["reversal"] == 0

    negative = pd.DataFrame(rows)
    negative["net_return"] = -negative["net_return"].abs()
    cash = _allocator_weights(negative, 4, ("momentum", "reversal"))
    assert cash.iloc[-1].sum() == 0


def test_random_smoke_produces_all_outputs_without_false_deployment_pass():
    result = evaluate_rotation(sample_panel(), rotation_spec())
    assert result["selected_window"] in {4, 8, 12}
    assert set(result["directions"]) == {
        "momentum", "reversal", "low_volatility", "liquidity", "carry"
    }
    assert {"static_equal", "dynamic_selected", "dynamic_selected_ex_momentum"} <= set(
        result["summary"]["strategy"]
    )
    assert result["bootstrap"]["observations"] > 8
    assert result["acceptance"]["passed"] is False


def test_missing_funding_history_disables_carry_without_emptying_universe():
    panel = sample_panel().drop(columns=["funding_rate"])
    result = evaluate_rotation(panel, rotation_spec())
    assert "carry" in result["inactive_styles"]
    assert result["active_styles"] == (
        "momentum", "reversal", "low_volatility", "liquidity"
    )
    assert not result["universe"].query("selected").empty
    assert "style_carry" not in set(result["summary"]["strategy"])


def test_market_state_is_joined_only_after_daily_observation_closes():
    result = evaluate_rotation(sample_panel(), rotation_spec())
    states = pd.DataFrame({
        "ts": pd.date_range("2024-07-01", periods=7, freq="D", tz="UTC"),
        "market_state": ["risk_on"] * 7,
    })
    joined, summary = market_state_attribution(result["returns"], states, rotation_spec())
    same_day = joined["ts"].eq(pd.Timestamp("2024-07-01", tz="UTC"))
    assert joined.loc[same_day, "market_state"].isna().all()
    assert not summary.empty
