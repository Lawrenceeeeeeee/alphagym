"""Frozen high-frequency factor combination research."""
from __future__ import annotations

import hashlib
import html
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alphagym import storage_io
from alphagym.config import resolve_root
from alphagym.equity_data import DataContractError
from alphagym.hf_research import _ic, outcomes
from alphagym.optional import require

BASELINES = ('equal_signed', 'ic_weighted')
SKLEARN_MODELS = ('ridge', 'elastic_net', 'pca_ridge', 'extra_trees', 'hist_gbdt')
METHODS = BASELINES + SKLEARN_MODELS


def load_ml_spec(path):
    spec = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(spec, dict):
        raise DataContractError('High-frequency ML spec must be a mapping')
    report_id = spec.get('source_report_id', '')
    if not isinstance(report_id, str) or not report_id.startswith('hf-'):
        raise DataContractError('source_report_id must identify a high-frequency factor report')
    methods = spec.get('methods', list(METHODS))
    unknown = set(methods)-set(METHODS)
    if not methods or unknown:
        raise DataContractError(f'Unsupported high-frequency ML methods: {sorted(unknown)}')
    workers = spec.get('workers', 8)
    if type(workers) is not int or not 1 <= workers <= 32:
        raise DataContractError('Workers must be an integer between 1 and 32')
    seed = spec.get('seed', 42)
    if type(seed) is not int:
        raise DataContractError('Seed must be an integer')
    fees = spec.get('fees_bps_per_side', [0, 1, 2, 5, 10])
    if not fees or any(not np.isfinite(value) or value < 0 for value in fees):
        raise DataContractError('Fees must be finite nonnegative bps')
    selection_fee = spec.get('selection_fee_bps_per_side', 5)
    if selection_fee not in fees:
        raise DataContractError('Selection fee must be included in fee scenarios')
    quantiles = spec.get('trade_edge_quantiles', [0, .5, .7, .8, .9, .95])
    if not quantiles or any(not np.isfinite(value) or not 0 <= value < 1 for value in quantiles):
        raise DataContractError('Trade edge quantiles must be between 0 and 1')
    if type(spec.get('min_policy_trades', 100)) is not int or spec.get('min_policy_trades', 100) < 1:
        raise DataContractError('Minimum policy trades must be a positive integer')
    spec['methods'] = list(dict.fromkeys(methods))
    return spec


def _estimator(method, workers, seed, feature_count):
    require('sklearn', 'ml')
    from sklearn.decomposition import PCA
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    if method == 'ridge':
        return make_pipeline(imputer, StandardScaler(), Ridge(alpha=10.0))
    if method == 'elastic_net':
        return make_pipeline(imputer, StandardScaler(), ElasticNet(
            alpha=.001, l1_ratio=.1, max_iter=3000, random_state=seed))
    if method == 'pca_ridge':
        components = min(32, feature_count)
        return make_pipeline(imputer, StandardScaler(), PCA(
            n_components=components, whiten=True, random_state=seed), Ridge(alpha=10.0))
    if method == 'extra_trees':
        return make_pipeline(imputer, ExtraTreesRegressor(
            n_estimators=160, max_depth=10, min_samples_leaf=50,
            max_features=.7, random_state=seed, n_jobs=workers))
    if method == 'hist_gbdt':
        return make_pipeline(imputer, HistGradientBoostingRegressor(
            max_iter=160, learning_rate=.05, max_leaf_nodes=15,
            min_samples_leaf=50, l2_regularization=1.0, random_state=seed))
    raise ValueError(f'Unknown model: {method}')


def _baseline_predictions(method, x_train, y_train, x_all, names):
    train = pd.DataFrame(x_train, columns=names)
    all_ = pd.DataFrame(x_all, columns=names)
    median = train.median()
    scale = train.quantile(.75)-train.quantile(.25)
    scale = scale.mask(scale.abs() < 1e-12, 1.0)
    train_z = (train-median)/scale
    all_z = ((all_-median)/scale).fillna(0).clip(-10, 10)
    ic = pd.Series({_name: _ic(train_z[_name], pd.Series(y_train)) for _name in names})
    signed = np.sign(ic.fillna(0))
    if method == 'equal_signed':
        weight = signed
    else:
        weight = ic.fillna(0)
    denominator = weight.abs().sum()
    if denominator == 0:
        return np.zeros(len(all_z)), ic
    return all_z.mul(weight, axis=1).sum(axis=1).to_numpy()/denominator, ic


def _importance(estimator, method, names):
    if method in BASELINES or method == 'pca_ridge':
        return []
    model = estimator.steps[-1][1]
    values = getattr(model, 'feature_importances_', getattr(model, 'coef_', None))
    if values is None:
        return []
    values = np.asarray(values).reshape(-1)
    order = np.argsort(np.abs(values))[::-1][:30]
    return [{'factor': names[index], 'importance': float(values[index])} for index in order]


def _fit_edge_calibration(score, target, train):
    valid = train & np.isfinite(score) & np.isfinite(target)
    if valid.sum() < 30 or np.std(score[valid]) < 1e-12:
        return np.full(len(score), np.nanmean(target[valid]))
    slope, intercept = np.polyfit(score[valid], target[valid], 1)
    return score*slope+intercept


def _metric_rows(method, horizon, long_prediction, short_prediction,
                 outcome, dates, split_masks, spec):
    finite = np.isfinite(long_prediction) & np.isfinite(short_prediction)
    dev = split_masks['development'] & outcome.valid.to_numpy() & finite
    prefer_long = long_prediction >= short_prediction
    predicted_edge = np.maximum(long_prediction, short_prediction)
    policy_candidates = []
    selection_fee = spec.get('selection_fee_bps_per_side', .1)
    for quantile in spec.get('trade_edge_quantiles', [0, .5, .7, .8, .9, .95]):
        threshold = float(np.quantile(predicted_edge[dev], quantile))
        buy = dev & prefer_long & (predicted_edge > threshold) & outcome.long_ok.to_numpy()
        sell = dev & ~prefer_long & (predicted_edge > threshold) & outcome.short_ok.to_numpy()
        gross = np.concatenate([outcome.long_bps.to_numpy()[buy],
                                outcome.short_bps.to_numpy()[sell]])
        if len(gross) >= spec.get('min_policy_trades', 100):
            policy_candidates.append((float(np.mean(gross)-2*selection_fee),
                                      quantile, threshold))
    if policy_candidates:
        _, quantile, threshold = max(policy_candidates, key=lambda item: item[0])
    else:
        quantile, threshold = .8, float(np.quantile(predicted_edge[dev], .8))
    rows, daily = [], []
    for split, mask in split_masks.items():
        ok = mask & outcome.valid.to_numpy() & finite
        buy = ok & prefer_long & (predicted_edge > threshold) & outcome.long_ok.to_numpy()
        sell = ok & ~prefer_long & (predicted_edge > threshold) & outcome.short_ok.to_numpy()
        gross = np.full(len(long_prediction), np.nan)
        gross[buy] = outcome.long_bps.to_numpy()[buy]
        gross[sell] = outcome.short_bps.to_numpy()[sell]
        for fee in spec.get('fees_bps_per_side', [0, 1, 2, 5, 10]):
            net = gross-2*fee
            traded = np.isfinite(net)
            row = {
                'method': method, 'horizon_seconds': horizon, 'split': split,
                'fee_bps_per_side': fee,
                'rank_ic': _ic(pd.Series((long_prediction-short_prediction)[ok]),
                               outcome.direction_bps[ok].reset_index(drop=True)),
                'observations': int(ok.sum()), 'trades': int(traded.sum()),
                'mean_net_bps': float(np.nanmean(net)) if traded.any() else np.nan,
                'win_rate': float(np.mean(net[traded] > 0)) if traded.any() else np.nan,
                'break_even_fee_bps_per_side': (
                    float(np.nanmean(gross)/2) if traded.any() else np.nan),
                'long_trades': int(buy.sum()), 'short_trades': int(sell.sum()),
            }
            rows.append(row)
            for date in sorted(np.unique(dates[mask])):
                day = mask & (dates == date)
                values = net[day & traded]
                daily.append({
                    'method': method, 'horizon_seconds': horizon, 'split': split,
                    'fee_bps_per_side': fee, 'date': date, 'trades': len(values),
                    'mean_net_bps': float(np.mean(values)) if len(values) else np.nan,
                })
    return rows, daily, float(quantile), float(threshold)


def _load_source(root, source_report_id):
    store = storage_io.store_for(root)
    prefix = f'factor_library/reports/{source_report_id}/'
    manifest = json.loads(store.read_blob(prefix+'manifest.json'))
    features = store.read_frame(prefix+'features.parquet')
    columns = ['ts', 'book_ts', 'mid', 'bid', 'ask', 'bid_size', 'ask_size', 'segment']
    raw = []
    for date in manifest['spec']['dates']:
        raw.append(store.read_frame(
            f'crypto/okx/hf_samples_v1/BTC-USDT/400/{date}.parquet', columns=columns))
    quotes = pd.concat(raw, ignore_index=True)
    frame = features.merge(quotes, on='ts', how='inner', validate='one_to_one')
    if len(frame) != len(features):
        raise DataContractError('Feature matrix and reconstructed quotes are not aligned')
    return manifest, frame.sort_values('ts').reset_index(drop=True)


def run_ml_research(root, spec_path):
    root = resolve_root(root)
    spec = load_ml_spec(spec_path)
    source, frame = _load_source(root, spec['source_report_id'])
    names = [item['name'] for item in source['factors']]
    x = frame[names].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    dates = frame.date.astype(str).to_numpy()
    split_masks = {split: frame.date.between(str(start), str(end)).to_numpy()
                   for split, (start, end) in source['spec']['splits'].items()}
    metrics, daily, policies, predictions, importances = [], [], [], [], []
    model_blobs = {}
    workers, seed = spec.get('workers', 8), spec.get('seed', 42)
    for horizon in source['spec']['horizons_seconds']:
        outcome = outcomes(frame, horizon, source['spec'].get('latency_seconds', 1),
                           source['spec'].get('quantity_btc', .0001))
        valid = outcome.valid.to_numpy()
        train = split_masks['development'] & valid
        x_train = x.loc[train]
        y_long = outcome.long_bps.to_numpy()
        y_short = outcome.short_bps.to_numpy()
        long_limits = np.quantile(y_long[train & np.isfinite(y_long)], [.01, .99])
        short_limits = np.quantile(y_short[train & np.isfinite(y_short)], [.01, .99])
        y_long_train = np.clip(y_long[train], *long_limits).astype(np.float32)
        y_short_train = np.clip(y_short[train], *short_limits).astype(np.float32)
        direction_train = outcome.direction_bps[train].to_numpy().astype(np.float32)
        for method in spec['methods']:
            if method in BASELINES:
                score, ic = _baseline_predictions(
                    method, x_train.to_numpy(), direction_train, x.to_numpy(), names)
                long_prediction = _fit_edge_calibration(score, y_long, train)
                short_prediction = _fit_edge_calibration(score, y_short, train)
                if method == 'ic_weighted':
                    importances.extend({'method': method, 'horizon_seconds': horizon,
                                        'factor': name, 'importance': float(value)}
                                       for name, value in ic.abs().nlargest(30).items())
            else:
                long_estimator = _estimator(method, workers, seed, len(names))
                short_estimator = _estimator(method, workers, seed+1, len(names))
                long_estimator.fit(x_train, y_long_train)
                short_estimator.fit(x_train, y_short_train)
                long_prediction = long_estimator.predict(x).astype(float)
                short_prediction = short_estimator.predict(x).astype(float)
                model_blobs[(horizon, method)] = pickle.dumps({
                    'format_version': 1, 'features': names, 'horizon_seconds': horizon,
                    'method': method, 'long_estimator': long_estimator,
                    'short_estimator': short_estimator,
                }, protocol=pickle.HIGHEST_PROTOCOL)
                importance = _importance(long_estimator, method, names)
                importances.extend({'method': method, 'horizon_seconds': horizon, **item}
                                   for item in importance)
            rows, day_rows, edge_quantile, edge_threshold = _metric_rows(
                method, horizon, long_prediction, short_prediction,
                outcome, dates, split_masks, spec)
            metrics.extend(rows)
            daily.extend(day_rows)
            policies.append({'method': method, 'horizon_seconds': horizon,
                             'edge_quantile': edge_quantile,
                             'predicted_gross_edge_threshold_bps': edge_threshold})
            observed = valid & np.isfinite(long_prediction) & np.isfinite(short_prediction)
            predictions.append(pd.DataFrame({
                'ts': frame.ts.to_numpy()[observed], 'date': dates[observed],
                'method': method, 'horizon_seconds': horizon,
                'predicted_long_bps': long_prediction[observed].astype(np.float32),
                'predicted_short_bps': short_prediction[observed].astype(np.float32),
            }))
    metrics, daily = pd.DataFrame(metrics), pd.DataFrame(daily)
    selection_fee = spec.get('selection_fee_bps_per_side', 5)
    validation = metrics[(metrics.split == 'validation')
                         & (metrics.fee_bps_per_side == selection_fee)].copy()
    chosen = []
    for horizon, group in validation.groupby('horizon_seconds'):
        row = group.sort_values(
            ['mean_net_bps', 'rank_ic'], ascending=False, key=lambda col: (
                col.abs() if col.name == 'rank_ic' else col)).iloc[0]
        chosen.append({'horizon_seconds': int(horizon), 'method': row.method,
                       'validation_mean_net_bps': float(row.mean_net_bps),
                       'validation_rank_ic': float(row.rank_ic)})
    identity = json.dumps({'source': source['report_id'], 'spec': spec,
                           'engine': Path(__file__).read_text(encoding='utf-8')}, sort_keys=True)
    report_id = 'hfml-'+hashlib.sha256(identity.encode()).hexdigest()[:16]
    manifest = {
        'report_id': report_id, 'report_type': 'hf_ml', 'mode': 'exploratory',
        'source_report_id': source['report_id'], 'spec': spec,
        'source_spec': source['spec'], 'samples': len(frame), 'feature_count': len(names),
        'prediction_target': 'executable long and short quote returns in bps',
        'selection': {'selection_split': 'validation', 'test_used_for_selection': False,
                      'fee_bps_per_side': selection_fee, 'chosen': chosen},
        'policies': policies,
        'model_resources': [f'models/{horizon}/{method}.pkl'
                            for horizon, method in sorted(model_blobs)],
        'limitations': [
            'One week of one instrument is insufficient to establish model stability.',
            'Hyperparameters are fixed for this first pass; method selection uses validation only.',
            'Execution diagnostics omit queue position, market impact and borrow costs.',
            'The primary fee threshold is deliberately demanding relative to observed gross edge.',
        ],
    }
    prefix = f'factor_library/reports/{report_id}/'
    report = render_ml_report(manifest, metrics)
    store = storage_io.store_for(root)
    with store.batch() as batch:
        batch.blob(prefix+'spec.yaml', yaml.safe_dump(spec).encode())
        batch.blob(prefix+'manifest.json', json.dumps(manifest).encode())
        batch.blob(prefix+'report.md', ml_summary(manifest, metrics).encode())
        batch.blob(prefix+'report.html', report.encode())
        batch.frame(prefix+'summary.csv', metrics)
        batch.frame(prefix+'daily.parquet', daily)
        batch.frame(prefix+'predictions.parquet', pd.concat(predictions, ignore_index=True),
                    keys=('ts', 'method', 'horizon_seconds'))
        batch.frame(prefix+'importance.parquet', pd.DataFrame(importances))
        for (horizon, method), blob in model_blobs.items():
            batch.blob(prefix+f'models/{horizon}/{method}.pkl', blob)
    return {'ok': True, 'report_id': report_id, 'source_report_id': source['report_id'],
            'samples': len(frame), 'models': len(spec['methods']), 'selected': chosen}


def ml_summary(manifest, metrics):
    source_line = (f"来源：{manifest['source_report_id']}；样本：{manifest['samples']:,}；"
                   f"特征：{manifest['feature_count']}。")
    lines = ['# AlphaGYM BTC-USDT 高频因子组合研究', '', source_line, '',
             '模型只使用开发段拟合，验证段选法，测试段只评估。', '',
             '|持有秒数|验证段选择|验证净收益 bps|验证 Rank IC|',
             '|---:|---|---:|---:|']
    for row in manifest['selection']['chosen']:
        lines.append(f"|{row['horizon_seconds']}|{row['method']}|"
                     f"{row['validation_mean_net_bps']:.4f}|{row['validation_rank_ic']:.4f}|")
    lines += ['', '## 限制', '']
    lines.extend('- '+item for item in manifest['limitations'])
    return '\n'.join(lines)+'\n'


def render_ml_report(manifest, metrics):
    fee = manifest['selection']['fee_bps_per_side']
    table = metrics[metrics.fee_bps_per_side == fee].copy()
    table = table[['method', 'horizon_seconds', 'split', 'rank_ic', 'trades',
                   'mean_net_bps', 'win_rate', 'break_even_fee_bps_per_side']]
    rendered = table.to_html(index=False, float_format=lambda value: f'{value:.4f}')
    chosen = html.escape(json.dumps(manifest['selection']['chosen'], ensure_ascii=False))
    limits = ''.join(f'<li>{html.escape(item)}</li>' for item in manifest['limitations'])
    return f'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>AlphaGYM 高频因子组合研究</title><style>
body{{font:15px system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#183047}}
table{{border-collapse:collapse;font-size:12px}}td,th{{padding:7px;border:1px solid #ddd}}
th{{background:#eef3f8}}h1{{color:#123b5d}}code{{word-break:break-all}}
</style><h1>AlphaGYM — BTC-USDT 高频因子组合研究</h1>
<p>来源报告 <code>{manifest['source_report_id']}</code> · {manifest['samples']:,} 个样本 ·
{manifest['feature_count']} 个特征。</p>
<p>开发段训练，验证段选择，测试段仅评估；每个预测周期独立建模。</p>
<p>验证段冻结选择：<code>{chosen}</code></p>
<h2>主成本情景：单边 {fee} bps</h2>{rendered}
<h2>边界与局限</h2><ul>{limits}</ul></html>'''
