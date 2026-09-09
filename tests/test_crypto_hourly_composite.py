from __future__ import annotations

import numpy as np
from test_crypto_hourly import sample_panel

from alphagym.crypto_hourly import HourlySpec, evaluate
from alphagym.crypto_hourly_composite import CompositeSpec, evaluate_composites


def test_composite_selection_uses_development_and_validation_only():
    features, _, _, _ = evaluate(
        sample_panel(n=720), HourlySpec(history_days=120, minimum_bars=100, horizons=(6,)))
    spec = CompositeSpec(source_report_id='crypto-source', features=(
        'momentum_1', 'momentum_3', 'volatility_24', 'candle_strength'))
    summary, curves, predictions, selected, metadata = evaluate_composites(features, spec)
    assert set(summary['method']) == {'equal_signed', 'ic_weighted', 'ridge', 'elastic_net'}
    assert set(summary['split']) == {'development', 'validation', 'test'}
    assert not curves.empty and not predictions.empty
    assert selected.iloc[0]['selection_score'] == max(selected['selection_score'])
    assert set(metadata) == set(summary['method'])

    changed = features.copy()
    test = changed['split'] == 'test'
    changed.loc[test, 'forward_open_return_6'] = np.random.default_rng(3).normal(0, 10, test.sum())
    _, _, _, selected_changed, metadata_changed = evaluate_composites(changed, spec)
    assert selected['method'].tolist() == selected_changed['method'].tolist()
    assert metadata == metadata_changed
