from __future__ import annotations

import numpy as np
import pandas as pd

from alphagym.crypto_hourly import HourlySpec, compute_factors, evaluate, liquid_usdt_swaps


def sample_panel(n=360, assets=8):
    times = pd.date_range('2024-01-01', periods=n, freq='4h', tz='UTC')
    rows = []
    rng = np.random.default_rng(7)
    for asset in range(assets):
        returns = rng.normal(asset * 1e-5, .01, n)
        close = 100 * np.exp(np.cumsum(returns))
        for i, ts in enumerate(times):
            rows.append({'ts': ts, 'instrument': f'C{asset}-USDT-SWAP',
                         'open': close[i] * np.exp(-returns[i]), 'high': close[i] * 1.005,
                         'low': close[i] * .995, 'close': close[i],
                         'volume_quote': 1_000_000 * (1 + abs(returns[i]))})
    return pd.DataFrame(rows)


def test_hourly_factors_are_computed_per_instrument():
    result = compute_factors(sample_panel())
    assert result.groupby('instrument')['momentum_1'].apply(lambda x: x.isna().sum()).eq(1).all()
    assert {'range_position_24', 'residual_momentum_12', 'funding_z_42'} <= set(result)


def test_evaluation_freezes_direction_on_development_and_keeps_test_separate():
    spec = HourlySpec(history_days=60, minimum_bars=100, horizons=(1,))
    features, summary, curves, selected = evaluate(sample_panel(), spec)
    assert set(features['split']) == {'development', 'validation', 'test'}
    assert set(summary['style']) == {'cross_sectional', 'time_series'}
    assert set(summary['split']) == {'development', 'validation', 'test'}
    directions = summary.groupby(['factor', 'style', 'horizon_bars'])['direction'].nunique()
    assert directions.eq(1).all()
    assert not curves.empty
    assert 'selection_score' in selected


def test_liquid_universe_uses_comparable_quote_notional():
    class Client:
        def list_instruments(self, _kind):
            return [{'instId': name, 'state': 'live', 'settleCcy': 'USDT', 'ctType': 'linear',
                     'listTime': '1'}
                    for name in ('BTC-USDT-SWAP', 'MEME-USDT-SWAP')]

        def tickers(self, _kind):
            return [
                {'instId': 'BTC-USDT-SWAP', 'volCcy24h': '100', 'last': '80000'},
                {'instId': 'MEME-USDT-SWAP', 'volCcy24h': '1000000000', 'last': '.000001'},
            ]

    assert liquid_usdt_swaps(Client(), 1) == ['BTC-USDT-SWAP']


def test_spec_rejects_unsafe_worker_count():
    import pytest

    with pytest.raises(ValueError, match='workers'):
        HourlySpec(workers=20).validate()
