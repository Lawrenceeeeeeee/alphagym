"""Frozen multi-factor combinations for crypto hourly research.

The source report may already have had its test segment inspected.  This module
therefore fits and selects exclusively on development/validation and labels the
test output as a reused audit, not a fresh out-of-sample result.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alphagym import storage_io

FEATURES = (
    'momentum_1', 'momentum_3', 'momentum_12', 'volatility_24',
    'volume_surprise_24', 'range_position_24', 'candle_strength',
    'illiquidity_24', 'volume_confirmed_momentum_3', 'residual_momentum_12',
)
METHODS = ('equal_signed', 'ic_weighted', 'ridge', 'elastic_net')


@dataclass(frozen=True)
class CompositeSpec:
    source_report_id: str
    horizon_bars: int = 6
    fee_bps_per_side: float = 5.0
    features: tuple[str, ...] = FEATURES
    methods: tuple[str, ...] = METHODS
    seed: int = 42

    def validate(self) -> None:
        if not self.source_report_id.startswith('crypto-'):
            raise ValueError('source_report_id must identify a crypto report')
        if self.horizon_bars < 1:
            raise ValueError('horizon_bars must be positive')
        if self.fee_bps_per_side < 0:
            raise ValueError('fee_bps_per_side must be nonnegative')
        if len(self.features) < 2:
            raise ValueError('a multi-factor combination needs at least two features')
        if not self.methods or set(self.methods) - set(METHODS):
            raise ValueError('unsupported composite method')


def _causal_robust_features(data: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    result = pd.DataFrame(index=data.index)
    grouped = data.groupby('instrument', observed=True)
    for feature in features:
        values = data[feature].replace([np.inf, -np.inf], np.nan)
        median = grouped[feature].transform(
            lambda x: x.rolling(84, min_periods=24).median())
        q25 = grouped[feature].transform(
            lambda x: x.rolling(84, min_periods=24).quantile(.25))
        q75 = grouped[feature].transform(
            lambda x: x.rolling(84, min_periods=24).quantile(.75))
        result[feature] = ((values - median) / (q75 - q25).replace(0, np.nan)).clip(-10, 10)
    return result


def _mean_time_series_ic(frame: pd.DataFrame, feature: str, label: str) -> float:
    values = frame.groupby('instrument', observed=True).apply(
        lambda x: x[feature].corr(x[label], method='spearman'), include_groups=False)
    return float(values.mean()) if len(values) else np.nan


def _fit_predictions(method: str, x: pd.DataFrame, y: pd.Series,
                     train: pd.Series, seed: int) -> tuple[np.ndarray, dict]:
    train_x = x.loc[train]
    train_y = y.loc[train]
    medians = train_x.median()
    fit_x = train_x.fillna(medians).fillna(0)
    all_x = x.fillna(medians).fillna(0)
    ics = pd.Series({name: _mean_time_series_ic(
        pd.DataFrame({'instrument': train_x.index.get_level_values('instrument'),
                      name: train_x[name].to_numpy(), 'label': train_y.to_numpy()}),
        name, 'label') for name in x.columns})
    if method in {'equal_signed', 'ic_weighted'}:
        weights = np.sign(ics.fillna(0)) if method == 'equal_signed' else ics.fillna(0)
        denominator = float(weights.abs().sum())
        score = (all_x.mul(weights, axis=1).sum(axis=1) / denominator
                 if denominator else pd.Series(0.0, index=all_x.index))
        return score.to_numpy(), {'weights': weights.to_dict(), 'development_ic': ics.to_dict()}

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = (make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                               Ridge(alpha=10.0)) if method == 'ridge' else
                 make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                               ElasticNet(alpha=.001, l1_ratio=.1, max_iter=5000,
                                          random_state=seed)))
    estimator.fit(fit_x, train_y)
    score = estimator.predict(all_x)
    coefficients = estimator.steps[-1][1].coef_
    return score, {
        'coefficients': dict(zip(x.columns, map(float, coefficients), strict=True)),
        'development_ic': ics.to_dict(),
    }


def _portfolio(score: np.ndarray, data: pd.DataFrame, label: str,
               fee_bps: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    work = data[['ts', 'instrument', label]].copy()
    work['score'] = score
    work['position'] = np.sign(work['score']).astype(float)
    gross = work.groupby('ts', observed=True)['position'].transform(lambda x: x.abs().sum())
    work['weight'] = work['position'] / gross.replace(0, np.nan)
    weights = work.pivot(index='ts', columns='instrument', values='weight').fillna(0)
    turnover = weights.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = weights.iloc[0].abs().sum()
    gross_return = (work['weight'] * work[label]).groupby(work['ts']).sum().reindex(weights.index)
    return gross_return - turnover * fee_bps / 10_000, turnover, gross_return


def evaluate_composites(features: pd.DataFrame, spec: CompositeSpec):
    spec.validate()
    missing = set(spec.features) - set(features)
    if missing:
        raise ValueError(f'source report is missing features: {sorted(missing)}')
    data = features.sort_values(['ts', 'instrument']).copy()
    if 'split' not in data:
        times = pd.Index(sorted(data['ts'].unique()))
        dev_end, val_end = times[int(len(times) * .6)], times[int(len(times) * .8)]
        data['split'] = np.where(data['ts'] < dev_end, 'development',
                                 np.where(data['ts'] < val_end, 'validation', 'test'))
    label = f'forward_open_return_{spec.horizon_bars}'
    if label not in data:
        raise ValueError(f'source report is missing {label}')
    times = pd.Index(sorted(data['ts'].unique()))
    bar_number = {ts: number for number, ts in enumerate(times)}
    data = data[data['ts'].map(bar_number) % spec.horizon_bars == 0].copy()
    x = _causal_robust_features(data, spec.features)
    index = pd.MultiIndex.from_arrays(
        [data['instrument'].to_numpy(), data['ts'].to_numpy()],
        names=['instrument', 'ts'])
    x.index = index
    y = pd.Series(data[label].to_numpy(), index=index)
    valid = y.notna()
    train = pd.Series((data['split'].to_numpy() == 'development') & valid.to_numpy(), index=index)
    rows, curves, predictions, metadata = [], [], [], {}
    annual_periods = 365 * 24 / 4 / spec.horizon_bars
    for method in spec.methods:
        score, meta = _fit_predictions(method, x, y, train, spec.seed)
        metadata[method] = meta
        net, turnover, gross = _portfolio(score, data, label, spec.fee_bps_per_side)
        split_by_ts = data.drop_duplicates('ts').set_index('ts')['split'].reindex(net.index)
        for split in ('development', 'validation', 'test'):
            sample = net[split_by_ts == split].dropna()
            mean = float(sample.mean()) if len(sample) else np.nan
            std = float(sample.std(ddof=1)) if len(sample) > 1 else np.nan
            rows.append({
                'method': method, 'split': split, 'horizon_bars': spec.horizon_bars,
                'observations': len(sample), 'mean_return_bps': mean * 10_000,
                'sharpe': mean / std * np.sqrt(annual_periods) if std > 0 else np.nan,
                'average_turnover': float(turnover[split_by_ts == split].mean()),
            })
        curve = pd.DataFrame({'ts': net.index, 'method': method,
                              'net_return': net.values, 'gross_return': gross.values,
                              'turnover': turnover.values})
        curve['cumulative_return'] = np.exp(curve['net_return'].fillna(0).cumsum()) - 1
        curves.append(curve)
        prediction = data[['ts', 'instrument', 'split']].copy()
        prediction['method'], prediction['score'] = method, score
        predictions.append(prediction)
    summary = pd.DataFrame(rows)
    selection = summary[summary['split'].isin(['development', 'validation'])].pivot(
        index='method', columns='split', values='sharpe')
    selection['selection_score'] = selection[['development', 'validation']].min(axis=1)
    selection = selection.sort_values('selection_score', ascending=False).reset_index()
    return summary, pd.concat(curves, ignore_index=True), pd.concat(predictions, ignore_index=True), selection, metadata


def run_composite_research(root, spec: CompositeSpec) -> dict:
    root = Path(root).expanduser().resolve()
    spec.validate()
    source = root / 'factor_library' / 'reports' / spec.source_report_id
    source_manifest = json.loads(storage_io.read_text(source / 'manifest.json'))
    if source_manifest.get('bar') != '4H':
        raise ValueError('the current composite protocol is frozen to 4H bars')
    features = storage_io.read_frame(source / 'features.parquet')
    summary, curves, predictions, selection, metadata = evaluate_composites(features, spec)
    selected_method = str(selection.iloc[0]['method'])
    report_id = 'crypto-combo-' + hashlib.sha256(
        f'{datetime.now(UTC).isoformat()}:{spec}'.encode()).hexdigest()[:12]
    manifest = {
        'ok': True, 'report_id': report_id, 'report_type': 'multi_factor',
        'created_at': datetime.now(UTC).isoformat(), 'source_report_id': spec.source_report_id,
        'market': source_manifest['market'], 'bar': '4H',
        'horizon_bars': spec.horizon_bars, 'holding_hours': 4 * spec.horizon_bars,
        'universe': source_manifest['universe'], 'start': source_manifest['start'],
        'end': source_manifest['end'], 'factors': list(spec.features),
        'methods': list(spec.methods), 'fee_bps_per_side': spec.fee_bps_per_side,
        'selected_method': selected_method,
        'selection_rule': 'fit on development; choose max min(development, validation) Sharpe',
        'test_status': 'reused_audit_only_not_fresh_out_of_sample',
        'funding_note': 'funding factors excluded because source history coverage is insufficient',
        'method_metadata': metadata,
        'selected': json.loads(selection.to_json(orient='records')),
    }
    prefix = f'factor_library/reports/{report_id}'
    with storage_io.store_for(root, initialize=True).batch() as batch:
        batch.frame(f'{prefix}/summary.csv', summary)
        batch.frame(f'{prefix}/curves.parquet', curves)
        batch.frame(f'{prefix}/predictions.parquet', predictions)
        batch.frame(f'{prefix}/selected.csv', selection)
        batch.blob(f'{prefix}/manifest.json', json.dumps(
            manifest, ensure_ascii=False, indent=2, default=float).encode())
    return manifest
