"""Parameterized, hypothesis-grouped L2 factor library for intraday research."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphagym.factors.base import FactorRegistry, FactorSpec
from alphagym.hf_book import LEVELS, PRESSURE_LEVELS

FLOW_WINDOWS = (1, 5, 30)
ZSCORE_WINDOWS = (5, 30, 60)
MOMENTUM_WINDOWS = (1, 5, 10, 30, 60, 300)
PRESSURE_WEIGHTS = ('inverse_level', 'linear', 'exp2', 'inverse_distance')


@dataclass
class HFFactorContext:
    signal_ts: int
    features: pd.DataFrame


@dataclass(frozen=True)
class HFFactorDefinition:
    name: str
    hypothesis: str
    lookback_seconds: int
    formula: str


def _definitions():
    result = []
    for n in LEVELS:
        result += [
            HFFactorDefinition(f'book_imbalance_{n}', 'book_imbalance', 0,
                               f'(bid_depth_{n}-ask_depth_{n})/total_depth_{n}'),
            HFFactorDefinition(f'notional_imbalance_{n}', 'book_imbalance', 0,
                               f'(bid_notional_{n}-ask_notional_{n})/total_notional_{n}'),
            HFFactorDefinition(f'log_depth_ratio_{n}', 'book_imbalance', 0,
                               f'log1p(bid_depth_{n})-log1p(ask_depth_{n})'),
            HFFactorDefinition(
                'weighted_mid_dislocation' if n == 1 else f'fair_price_dislocation_{n}',
                'weighted_mid', 0, f'10000*(opposite_depth_weighted_vwap_{n}/mid-1)'),
        ]
        for window in FLOW_WINDOWS:
            result += [
                HFFactorDefinition(f'depth_flow_{n}_{window}s', 'aggregate_depth_flow', window,
                                   f'rolling_sum(delta_bid_depth_{n}-delta_ask_depth_{n},{window}s)'),
                HFFactorDefinition(
                    'imbalance_change_5s' if (n, window) == (5, 5)
                    else f'imbalance_change_{n}_{window}s',
                    'imbalance_change', window, f'book_imbalance_{n}-lag_{window}s'),
            ]
        for window in ZSCORE_WINDOWS:
            result.append(HFFactorDefinition(f'imbalance_zscore_{n}_{window}s',
                                             'imbalance_persistence', window,
                                             f'zscore(book_imbalance_{n},{window}s)'))
    for n in LEVELS[1:]:
        result += [
            HFFactorDefinition(f'distance_asymmetry_{n}', 'book_geometry', 0,
                               f'(ask_radius_{n}-bid_radius_{n})/total_radius_{n}'),
            HFFactorDefinition(f'radius_log_ratio_{n}', 'book_geometry', 0,
                               f'log1p(ask_radius_{n})-log1p(bid_radius_{n})'),
            HFFactorDefinition(f'density_asymmetry_{n}', 'book_slope', 0,
                               f'normalized(depth_{n}/price_radius_{n})'),
        ]
    for n in PRESSURE_LEVELS:
        for weight in PRESSURE_WEIGHTS:
            result.append(HFFactorDefinition(f'pressure_{weight}_{n}',
                                             'distance_weighted_pressure', 0,
                                             f'normalized_bid_ask_pressure({weight},{n})'))
        result.append(HFFactorDefinition(f'concentration_asymmetry_{n}',
                                         'depth_concentration', 0,
                                         f'normalized(top_depth_share_{n})'))
    result.append(HFFactorDefinition('depth_shape', 'book_geometry', 0,
                                     '(ask_radius_20-bid_radius_20)/total_radius_20'))
    for window in FLOW_WINDOWS:
        result.append(HFFactorDefinition(f'ofi_{window}s', 'order_flow_imbalance', window,
                                         f'sum(event_OFI,{window}s)/mean(BBO_depth,{window}s)'))
    for window in MOMENTUM_WINDOWS:
        result.append(HFFactorDefinition(f'momentum_{window}s', 'short_term_momentum', window,
                                         f'10000*(mid/mid_lag_{window}s-1)'))
    result += [
        HFFactorDefinition('spread_bps', 'spread_state', 0, '10000*(ask-bid)/mid'),
        HFFactorDefinition('spread_change_1s', 'spread_state', 1, 'spread_bps-lag_1s'),
        HFFactorDefinition('spread_change_5s', 'spread_state', 5, 'spread_bps-lag_5s'),
    ]
    for window in (1, 5, 30, 60):
        result.append(HFFactorDefinition(f'event_intensity_{window}s', 'quote_intensity', window,
                                         f'sum(order_book_messages,{window}s)/{window}'))
    names = [item.name for item in result]
    if len(names) != len(set(names)):
        raise RuntimeError('Duplicate high-frequency factor definition')
    return tuple(result)


DEFINITIONS = _definitions()


def registry():
    result = FactorRegistry()
    for definition in DEFINITIONS:
        def calculate(context, column=definition.name):
            if not isinstance(context, HFFactorContext):
                raise TypeError('High-frequency factors require HFFactorContext')
            return context.features[column].where(context.features['ts'] <= context.signal_ts)

        result.register(FactorSpec(
            name=definition.name, factor_id='hf.'+definition.name,
            hypothesis_id='hf.'+definition.hypothesis, family='high_frequency',
            formula_version='2.0', input_fields=('order_book',), lookback_days=0,
            min_observations=max(1, definition.lookback_seconds),
            availability_rule='exchange_ts <= signal_ts; execution strictly later',
            expected_direction='unknown', calculator=calculate, formula=definition.formula,
            description=(f'Experimental L2 factor; '
                         f'lookback_seconds={definition.lookback_seconds}'),
            tags=('crypto', 'experimental',
                  f'lookback_seconds:{definition.lookback_seconds}'),
        ))
    return result


def factor_catalog():
    specs = {item.name: item for item in registry().list()}
    return [{
        'factor_id': specs[item.name].factor_id, 'name': item.name,
        'hypothesis_id': specs[item.name].hypothesis_id, 'formula': item.formula,
        'lookback_seconds': item.lookback_seconds, 'status': 'experimental',
        'formula_version': specs[item.name].formula_version,
    } for item in DEFINITIONS]


def _rolling(series, blocks, window, operation):
    grouped = series.groupby(blocks)
    return grouped.transform(
        lambda x: getattr(x.rolling(window, min_periods=window), operation)())


def _normalized(left, right):
    return (left-right)/(left+right).replace(0, np.nan)


def _level_features(payload):
    """Compute one depth-level family in a spawn-safe worker process."""
    n, values = payload
    blocks = pd.Series(values.pop('blocks'))
    frame = {name: pd.Series(value) for name, value in values.items()}
    b, a = frame['bdepth'], frame['adepth']
    bn, an, mid = frame['bnotional'], frame['anotional'], frame['mid']
    result = {}
    imbalance = _normalized(b, a)
    result[f'book_imbalance_{n}'] = imbalance
    result[f'notional_imbalance_{n}'] = _normalized(bn, an)
    result[f'log_depth_ratio_{n}'] = np.log1p(b)-np.log1p(a)
    bid_vwap, ask_vwap = bn/b.replace(0, np.nan), an/a.replace(0, np.nan)
    fair = (ask_vwap*b+bid_vwap*a)/(a+b).replace(0, np.nan)
    name = 'weighted_mid_dislocation' if n == 1 else f'fair_price_dislocation_{n}'
    result[name] = (fair/mid-1)*10000
    delta = b.groupby(blocks).diff()-a.groupby(blocks).diff()
    for window in FLOW_WINDOWS:
        depth = _rolling(b+a, blocks, window, 'mean')
        result[f'depth_flow_{n}_{window}s'] = (
            _rolling(delta, blocks, window, 'sum')/depth.replace(0, np.nan))
        change_name = ('imbalance_change_5s' if (n, window) == (5, 5)
                       else f'imbalance_change_{n}_{window}s')
        result[change_name] = imbalance-imbalance.groupby(blocks).shift(window)
    for window in ZSCORE_WINDOWS:
        mean = _rolling(imbalance, blocks, window, 'mean')
        std = _rolling(imbalance, blocks, window, 'std')
        result[f'imbalance_zscore_{n}_{window}s'] = (
            (imbalance-mean)/std.replace(0, np.nan))
    if n > 1:
        bid_radius, ask_radius = frame['bid_distance'], frame['ask_distance']
        result[f'distance_asymmetry_{n}'] = _normalized(ask_radius, bid_radius)
        result[f'radius_log_ratio_{n}'] = np.log1p(ask_radius)-np.log1p(bid_radius)
        result[f'density_asymmetry_{n}'] = _normalized(
            b/bid_radius.replace(0, np.nan), a/ask_radius.replace(0, np.nan))
    return {name: value.to_numpy() for name, value in result.items()}


def _level_payload(frame, blocks, n):
    values = {
        'blocks': blocks.to_numpy(), 'mid': frame.mid.to_numpy(),
        'bdepth': frame[f'bdepth_{n}'].to_numpy(),
        'adepth': frame[f'adepth_{n}'].to_numpy(),
        'bnotional': frame[f'bnotional_{n}'].to_numpy(),
        'anotional': frame[f'anotional_{n}'].to_numpy(),
    }
    if n > 1:
        values['bid_distance'] = frame[f'bid_distance_{n}'].to_numpy()
        values['ask_distance'] = frame[f'ask_distance_{n}'].to_numpy()
    return n, values


def compute_features(samples, workers=1):
    """Compute every registered feature causally on complete one-second blocks."""
    frame = samples.sort_values('ts').copy().reset_index(drop=True)
    if frame.ts.duplicated().any():
        raise ValueError('Duplicate high-frequency sample timestamp')
    ts = frame.ts.to_numpy()
    blocks = ((frame.ts.diff() != 1000) | (frame.segment.diff() != 0)
              | (frame.ts//86400000).diff().ne(0)).cumsum()
    frame['ofi'] = frame.ofi.where(blocks.eq(blocks.shift()))
    features = {}
    spread = (frame.ask-frame.bid)/frame.mid*10000
    features['spread_bps'] = spread
    for window in (1, 5):
        features[f'spread_change_{window}s'] = spread-spread.groupby(blocks).shift(window)
    payloads = [_level_payload(frame, blocks, n) for n in LEVELS]
    if workers > 1 and len(frame) >= 1_000:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(payloads))) as executor:
            level_results = executor.map(_level_features, payloads)
            for result in level_results:
                features.update(result)
    else:
        for payload in payloads:
            features.update(_level_features(payload))
    features['depth_shape'] = features['distance_asymmetry_20']
    for n in PRESSURE_LEVELS:
        for weight in PRESSURE_WEIGHTS:
            features[f'pressure_{weight}_{n}'] = _normalized(
                frame[f'bpressure_{weight}_{n}'], frame[f'apressure_{weight}_{n}'])
        features[f'concentration_asymmetry_{n}'] = _normalized(
            frame.bdepth_1/frame[f'bdepth_{n}'], frame.adepth_1/frame[f'adepth_{n}'])
    for window in FLOW_WINDOWS:
        depth = _rolling(frame.bid_size+frame.ask_size, blocks, window, 'mean')
        features[f'ofi_{window}s'] = (
            _rolling(frame.ofi, blocks, window, 'sum')/depth.replace(0, np.nan))
    for window in MOMENTUM_WINDOWS:
        features[f'momentum_{window}s'] = (
            frame.mid/frame.mid.groupby(blocks).shift(window)-1)*10000
    for window in (1, 5, 30, 60):
        features[f'event_intensity_{window}s'] = (
            _rolling(frame.events, blocks, window, 'sum')/window)
    frame['block'] = blocks
    frame['date'] = pd.to_datetime(ts, unit='ms', utc=True).strftime('%Y-%m-%d')
    frame = pd.concat([frame, pd.DataFrame(features, index=frame.index)], axis=1)
    missing = {item.name for item in DEFINITIONS}-set(frame)
    if missing:
        raise RuntimeError(f'Missing computed high-frequency factors: {sorted(missing)}')
    return frame.replace([np.inf, -np.inf], np.nan)
