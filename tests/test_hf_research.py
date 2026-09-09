from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphagym.equity_data import DataContractError
from alphagym.hf_book import LEVELS, BookSampler
from alphagym.hf_factors import (
    PRESSURE_LEVELS,
    PRESSURE_WEIGHTS,
    HFFactorContext,
    compute_features,
    factor_catalog,
    registry,
)
from alphagym.hf_research import evaluate, load_hf_spec, outcomes


def message(row, ts, *, action='update', bids=(), asks=()):
    return {'source_row': row, 'ts': ts, 'action': action, 'bids': bids, 'asks': asks}


def initial(ts=10):
    return message(0, ts, action='snapshot', bids=[['100', '3']], asks=[['102', '1']])


def samples():
    rows = []
    for day in pd.date_range('2026-08-29', periods=3, tz='UTC'):
        for second in range(2000):
            ts = day.value//1_000_000+second*1000
            mid = 100+np.sin(second/30)
            row = {'ts': ts, 'book_ts': ts, 'segment': 1, 'bid': mid-.01,
                   'ask': mid+.01, 'mid': mid, 'bid_size': 2., 'ask_size': 1.,
                   'ofi': np.cos(second/30), 'events': 5}
            for n in LEVELS:
                row[f'bdepth_{n}'] = n*(2+np.cos(second/30))
                row[f'adepth_{n}'] = float(n)
                row[f'bnotional_{n}'] = row[f'bdepth_{n}']*(mid-.02)
                row[f'anotional_{n}'] = row[f'adepth_{n}']*(mid+.02)
                row[f'bid_distance_{n}'] = max(.01, n*.05)
                row[f'ask_distance_{n}'] = max(.01, n*.06)
            for n in PRESSURE_LEVELS:
                for weight in PRESSURE_WEIGHTS:
                    row[f'bpressure_{weight}_{n}'] = row[f'bdepth_{n}']/(1+n/10)
                    row[f'apressure_{weight}_{n}'] = row[f'adepth_{n}']/(1+n/10)
            rows.append(row)
    return pd.DataFrame(rows)


def spec():
    return {'splits': {'development': ['2026-08-29', '2026-08-29'],
                       'validation': ['2026-08-30', '2026-08-30'],
                       'test': ['2026-08-31', '2026-08-31']},
            'horizons_seconds': [5], 'fees_bps_per_side': [0, 5]}


def test_grid_is_causal_and_equal_timestamp_updates_are_atomic():
    sampler = BookSampler()
    sampler.consume([initial(), message(1, 1000, bids=[['100', '5']]),
                     message(2, 1000, asks=[['102', '2']]),
                     message(3, 1001, bids=[['100', '90']]), message(4, 2001)])
    frame, _ = sampler.finish()
    assert frame.ts.tolist() == [1000, 2000]
    assert frame.bid_size.tolist() == [5, 90]
    assert frame.ask_size.tolist() == [2, 2]
    assert frame.ofi.tolist() == [1, 85]


def test_delete_snapshot_reset_and_missing_depth_remain_nan():
    sampler = BookSampler()
    sampler.consume([initial(), message(1, 200, bids=[['100', '0'], ['99', '2']]),
                     message(2, 1001),
                     message(3, 1200, action='snapshot', bids=[['98', '4']],
                             asks=[['101', '2']]), message(4, 2001)])
    frame, _ = sampler.finish()
    assert frame.bid.tolist() == [99, 98]
    assert frame.segment.tolist() == [1, 2]
    assert frame.bdepth_5.isna().all()


def test_stale_and_crossed_books_are_excluded():
    sampler = BookSampler(max_stale_ms=1100)
    sampler.consume([initial(), message(1, 4001, bids=[['103', '2']]), message(2, 5001)])
    frame, quality = sampler.finish()
    assert frame.ts.tolist() == [1000]
    assert quality['invalid_grid'] == 4
    assert quality['crossed_or_empty'] == 2


@pytest.mark.parametrize('bad', [message(2, 20), message(1, 0),
                                message(1, 20, bids=[['100', '-1']])])
def test_corrupt_messages_block_reconstruction(bad):
    sampler = BookSampler()
    with pytest.raises(DataContractError):
        sampler.consume([initial(), bad])


def test_features_do_not_change_when_future_changes_and_clock_gap_resets_lags():
    raw = samples().iloc[:100].copy()
    original = compute_features(raw)
    changed = raw.copy()
    changed.loc[80:, 'mid'] *= 2
    modified = compute_features(changed)
    pd.testing.assert_frame_equal(original.iloc[:80], modified.iloc[:80])
    gapped = compute_features(raw.drop(index=50))
    assert np.isnan(gapped.loc[50, 'momentum_1s'])
    assert np.isnan(gapped.loc[50, 'ofi_5s'])
    assert np.isnan(original.loc[0, 'momentum_1s'])


def test_registered_calculator_keeps_intraday_time():
    frame = compute_features(samples().iloc[:100])
    result = registry().get('book_imbalance_1').calculator(
        HFFactorContext(int(frame.ts.iloc[50]), frame))
    assert result.iloc[:51].notna().all()
    assert result.iloc[51:].isna().all()


def test_large_catalog_is_grouped_and_fully_computed():
    catalog = factor_catalog()
    assert len(catalog) >= 150
    assert len({item['factor_id'] for item in catalog}) == len(catalog)
    assert len({item['hypothesis_id'] for item in catalog}) < 20
    frame = compute_features(samples().iloc[:400])
    assert {item['name'] for item in catalog} <= set(frame)


def test_multiprocess_features_match_single_process():
    raw = samples().iloc[:1500]
    single = compute_features(raw, workers=1)
    parallel = compute_features(raw, workers=2)
    pd.testing.assert_frame_equal(single, parallel)


def test_labels_use_later_quotes_and_capacity_and_do_not_cross_boundaries():
    frame = compute_features(samples().iloc[:30])
    frame['mid'] = 100.
    frame['bid'], frame['ask'] = 99., 101.
    result = outcomes(frame, 5, 1, .1)
    valid = result.valid
    assert valid.any()
    assert np.allclose(result.loc[valid, 'mid_bps'], 0)
    assert (result.loc[valid, 'long_bps'] < -190).all()
    assert (result.loc[valid, 'short_bps'] < -200).all()
    assert np.allclose(result.loc[valid, 'direction_bps'],
                       (result.loc[valid, 'long_bps']-result.loc[valid, 'short_bps'])/2)
    assert not result.valid.iloc[-6:].any()
    assert not outcomes(frame, 5, 1, 100).long_ok.any()
    frame.loc[1:, 'block'] = 99
    assert not outcomes(frame, 5, 1, .1).valid.iloc[0]


def test_test_data_cannot_change_policies_or_selection():
    raw = samples()
    _, metrics, _, selected = evaluate(raw, spec())
    altered = raw.copy()
    altered.loc[4000:, ['mid', 'bid', 'ask']] *= 10
    _, other, _, selected_other = evaluate(altered, spec())
    assert selected == selected_other
    pd.testing.assert_frame_equal(metrics[metrics.split != 'test'], other[other.split != 'test'])
    free = metrics[metrics.fee_bps_per_side == 0].reset_index(drop=True)
    paid = metrics[metrics.fee_bps_per_side == 5].reset_index(drop=True)
    assert np.allclose(free.mean_net_bps-10, paid.mean_net_bps, equal_nan=True)


def test_hf_web_reads_stored_report_without_computation(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from alphagym import hf_research
    from alphagym.factor_web import create_app
    from alphagym.storage import ClickHouseStore

    def forbidden(*args, **kwargs):
        raise AssertionError('Viewing must not compute')

    monkeypatch.setattr(hf_research, 'evaluate', forbidden)
    database = ClickHouseStore(tmp_path, initialize=True)
    report_id = 'hf-'+'a'*16
    database.write_blob(f'factor_library/reports/{report_id}/report.html', b'<h1>Stored HF</h1>')
    client = TestClient(create_app(tmp_path))
    assert 'book_imbalance_1' in client.get('/hf').text
    assert 'Stored HF' in client.get('/hf/reports/'+report_id).text
    assert client.get('/hf/reports/not-an-id').status_code == 404


def test_spec_rejects_overlapping_splits(tmp_path):
    import yaml

    value = spec()
    value.update(mode='exploratory', instrument='BTC-USDT', exchange='OKX')
    value['splits']['validation'][0] = '2026-08-29'
    path = tmp_path/'spec.yaml'
    path.write_text(yaml.safe_dump(value), encoding='utf-8')
    with pytest.raises(DataContractError, match='non-overlapping'):
        load_hf_spec(path)
