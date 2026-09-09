"""Offline exploratory L2 research with chronological splits and frozen selection."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alphagym import storage_io
from alphagym.config import resolve_root
from alphagym.equity_data import DataContractError
from alphagym.hf_factors import compute_features, factor_catalog


def load_hf_spec(path):
    spec = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(spec, dict):
        raise DataContractError('High-frequency spec must be a mapping')
    if spec.get('mode') != 'exploratory':
        raise DataContractError('High-frequency research currently requires mode: exploratory')
    if spec.get('instrument') != 'BTC-USDT' or spec.get('exchange') != 'OKX':
        raise DataContractError('Current adapter supports OKX BTC-USDT spot only')
    dates = []
    for split in ('development', 'validation', 'test'):
        values = spec['splits'][split]
        start, end = (pd.Timestamp(str(x), tz='UTC') for x in values)
        if start > end:
            raise DataContractError('Invalid split dates')
        days = pd.date_range(start, end).strftime('%Y-%m-%d').tolist()
        dates.extend(days)
    if dates != pd.date_range(dates[0], dates[-1]).strftime('%Y-%m-%d').tolist():
        raise DataContractError('Splits must be ordered, adjacent and non-overlapping')
    if spec.get('sample_ms', 1000) != 1000:
        raise DataContractError('Initial research engine requires 1-second sampling')
    if any(type(n) is not int or n < 1 for n in spec.get('horizons_seconds', [1, 5, 30, 60])):
        raise DataContractError('Horizons must be positive integer seconds')
    if type(spec.get('latency_seconds', 1)) is not int or spec.get('latency_seconds', 1) < 1:
        raise DataContractError('Execution latency must be at least one second')
    workers = spec.get('workers', 1)
    if type(workers) is not int or not 1 <= workers <= 32:
        raise DataContractError('Workers must be an integer between 1 and 32')
    fees = spec.get('fees_bps_per_side', [0, 1, 2, 5, 10])
    if not fees or any(not np.isfinite(f) or f < 0 for f in fees):
        raise DataContractError('Fees must be finite nonnegative bps')
    if not np.isfinite(spec.get('quantity_btc', .0001)) or spec.get('quantity_btc', .0001) <= 0:
        raise DataContractError('Quantity must be positive')
    quantiles = spec.get('signal_upper_quantiles', [.8])
    if not quantiles or any(not np.isfinite(value) or not .5 < value < 1 for value in quantiles):
        raise DataContractError('Signal upper quantiles must be between 0.5 and 1')
    if type(spec.get('min_policy_trades', 100)) is not int or spec.get('min_policy_trades', 100) < 1:
        raise DataContractError('Minimum policy trades must be a positive integer')
    spec['dates'] = dates
    return spec


def outcomes(frame, horizon, latency, quantity):
    """Exact later quotes, no asof lookahead, full observed BBO capacity on both legs."""
    indexed = frame.set_index('ts')
    ts = frame.ts.to_numpy()
    entry = indexed.reindex(ts+latency*1000).reset_index(drop=True)
    exit_ = indexed.reindex(ts+(latency+horizon)*1000).reset_index(drop=True)
    valid = (entry.block.to_numpy() == frame.block.to_numpy()) & (
        exit_.block.to_numpy() == frame.block.to_numpy())
    valid &= (entry.book_ts.to_numpy() > ts)
    # Signal sampling phase is fixed in UTC, not chosen from profitable events.
    valid &= (ts//1000) % (horizon+latency) == 0
    long_ok = valid & (entry.ask_size >= quantity) & (exit_.bid_size >= quantity)
    short_ok = valid & (entry.bid_size >= quantity) & (exit_.ask_size >= quantity)
    result = pd.DataFrame({
        'mid_bps': (exit_.mid.to_numpy()/entry.mid.to_numpy()-1)*10000,
        'long_bps': (exit_.bid.to_numpy()/entry.ask.to_numpy()-1)*10000,
        'short_bps': (1-exit_.ask.to_numpy()/entry.bid.to_numpy())*10000,
        'long_ok': long_ok, 'short_ok': short_ok, 'valid': valid,
    })
    result['direction_bps'] = (result.long_bps-result.short_bps)/2
    result.loc[~valid, ['mid_bps', 'long_bps', 'short_bps']] = np.nan
    return result


def _ic(x, y):
    ok = x.notna() & y.notna()
    if ok.sum() < 30 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan
    return float(x[ok].corr(y[ok], method='spearman'))


def evaluate(frame, spec):
    """Return all splits for audit; select exclusively with development/validation."""
    frame = compute_features(frame, workers=spec.get('workers', 1))
    catalog = factor_catalog()
    names = [x['name'] for x in catalog]
    split_masks = {}
    for split, (start, end) in spec['splits'].items():
        split_masks[split] = frame.date.between(str(start), str(end))
    metrics, daily, policies = [], [], []
    fees = spec.get('fees_bps_per_side', [0, 1, 2, 5, 10])
    for horizon in spec.get('horizons_seconds', [1, 5, 30, 60]):
        outcome = outcomes(frame, horizon, spec.get('latency_seconds', 1),
                           spec.get('quantity_btc', .0001))
        for name in names:
            x = frame[name]
            dev = split_masks['development'] & outcome.valid
            ic_dev = _ic(x[dev], outcome.direction_bps[dev])
            direction = 1 if not np.isfinite(ic_dev) or ic_dev >= 0 else -1
            signed = x*direction
            policy_candidates = []
            selection_fee = spec.get('selection_fee_bps_per_side', 5)
            for upper_quantile in spec.get('signal_upper_quantiles', [.8]):
                low, high = signed[dev].quantile([1-upper_quantile, upper_quantile])
                buy = dev & (signed > high) & outcome.long_ok
                sell = dev & (signed < low) & outcome.short_ok
                gross = pd.concat([outcome.long_bps[buy], outcome.short_bps[sell]])
                if len(gross) >= spec.get('min_policy_trades', 100):
                    policy_candidates.append((gross.mean()-2*selection_fee,
                                              upper_quantile, low, high))
            if policy_candidates:
                _, upper_quantile, low, high = max(policy_candidates, key=lambda item: item[0])
            else:
                upper_quantile = .8
                low, high = signed[dev].quantile([.2, .8])
            policies.append({'factor': name, 'horizon_seconds': horizon,
                             'direction': direction, 'low': low, 'high': high,
                             'upper_quantile': upper_quantile})
            for split, mask in split_masks.items():
                ok = mask & outcome.valid & x.notna()
                buy = ok & (signed > high) & outcome.long_ok
                sell = ok & (signed < low) & outcome.short_ok
                gross = pd.Series(np.nan, index=frame.index)
                gross[buy] = outcome.long_bps[buy]
                gross[sell] = outcome.short_bps[sell]
                rank_ic = _ic(x[ok], outcome.direction_bps[ok])
                day_ics = {}
                for date in sorted(frame.loc[mask, 'date'].unique()):
                    day = frame.date == date
                    day_ics[date] = _ic(x[ok & day], outcome.direction_bps[ok & day])
                for fee in fees:
                    net = gross-2*fee
                    row = {'factor': name, 'horizon_seconds': horizon, 'split': split,
                           'fee_bps_per_side': fee, 'direction': direction,
                           'rank_ic': rank_ic,
                           'observations': int(ok.sum()), 'trades': int(net.notna().sum()),
                           'mean_net_bps': net.mean(), 'win_rate': (net.dropna() > 0).mean(),
                           'long_trades': int(buy.sum()),
                           'long_only_mean_net_bps': (outcome.long_bps[buy]-2*fee).mean(),
                           'break_even_fee_bps_per_side': gross.mean()/2}
                    metrics.append(row)
                    for date in sorted(frame.loc[mask, 'date'].unique()):
                        day = frame.date == date
                        values = net[day].dropna()
                        daily.append({**{k: row[k] for k in (
                            'factor', 'horizon_seconds', 'split', 'fee_bps_per_side')},
                            'date': date, 'trades': len(values),
                            'mean_net_bps': values.mean(),
                            'rank_ic': day_ics[date]})
    metrics, daily = pd.DataFrame(metrics), pd.DataFrame(daily)
    # Fixed primary fee scenario. No candidate is required to pass this gate.
    primary_fee = spec.get('selection_fee_bps_per_side', 5)
    if primary_fee not in fees:
        raise DataContractError('Selection fee must be included in fee scenarios')
    train = metrics[(metrics.fee_bps_per_side == primary_fee)
                    & metrics.split.isin(['development', 'validation'])]
    candidates = []
    for (name, horizon), group in train.groupby(['factor', 'horizon_seconds']):
        if len(group) != 2 or not (group.trades >= 100).all():
            continue
        score = group.mean_net_bps.min()
        if score > 0:
            candidates.append({'factor': name, 'horizon_seconds': int(horizon),
                               'score': float(score)})
    candidates.sort(key=lambda x: (-x['score'], x['factor'], x['horizon_seconds']))
    selection = {'primary_fee_bps_per_side': primary_fee,
                 'rule': 'positive mean net bps and >=100 trades in BOTH development/validation',
                 'selected': candidates[:1], 'passing_candidates': candidates,
                 'test_used_for_selection': False,
                 'policies': policies,
                 'candidate_count': len(names)*len(spec.get('horizons_seconds', [1, 5, 30, 60]))}
    return frame, metrics, daily, selection


def run_research(root, spec_path):
    root = resolve_root(root)
    spec = load_hf_spec(spec_path)
    store = storage_io.store_for(root)
    frames, sources, quality = [], [], []
    for date in spec['dates']:
        path = f'crypto/okx/hf_samples_v1/BTC-USDT/400/{date}.parquet'
        meta = store.manifest(path)
        if meta is None:
            raise DataContractError(f'Missing reconstructed samples: {date}; run hf build first')
        part = store.read_frame(path, as_of=meta['version'])
        if len(part) < 86400*.95:
            raise DataContractError(f'Insufficient daily sample coverage: {date}')
        frames.append(part)
        sources.append({'path': path, 'version': meta['version']})
        quality.append(json.loads(store.read_blob(
            path.removesuffix('.parquet')+'.json', as_of=meta['version'])))
    frame, metrics, daily, selection = evaluate(pd.concat(frames, ignore_index=True), spec)
    engine_hash = hashlib.sha256(b''.join((Path(__file__).parent/name).read_bytes()
        for name in ('hf_book.py', 'hf_factors.py', 'hf_research.py'))).hexdigest()
    identity = json.dumps({'spec': spec, 'sources': sources, 'engine': engine_hash}, sort_keys=True)
    report_id = 'hf-' + hashlib.sha256(identity.encode()).hexdigest()[:16]
    prefix = f'factor_library/reports/{report_id}/'
    manifest = {'report_id': report_id, 'mode': 'exploratory', 'spec': spec,
                'sources': sources, 'quality': quality, 'factors': factor_catalog(),
                'engine_hash': engine_hash,
                'samples': len(frame), 'selection': selection,
                'prediction_target': 'executable long/short quote returns',
                'limitations': [
                    'One week, one venue and one spot instrument; no durable alpha claim.',
                    'Exchange timestamps only; sequence/checksum/receipt timestamps unavailable.',
                    'Short leg is hypothetical spot inventory/borrow; no borrow costs modeled.',
                    'BBO capacity filter is optimistic; no queue fills or market impact simulation.',
                    'Capacity is checked ex post at both legs; this is a quote-return diagnostic.',
                    'Fees are assumptions, not an account-specific exchange tariff.',
                    'Non-overlapping observations remain serially dependent; no IID significance claim.',
                    '2026 crypto experiment is separate from the A-share monitoring-only policy.',
                ]}
    report = render_report(manifest, metrics, daily)
    with store.batch() as batch:
        batch.blob(prefix+'spec.yaml', yaml.safe_dump(spec).encode())
        batch.blob(prefix+'manifest.json', json.dumps(manifest, default=str).encode())
        batch.blob(prefix+'report.html', report.encode())
        batch.blob(prefix+'report.md', report_summary(manifest, metrics).encode())
        batch.frame(prefix+'summary.csv', metrics)
        batch.frame(prefix+'daily.parquet', daily)
        feature_names = [item['name'] for item in factor_catalog()]
        batch.frame(prefix+'features.parquet', frame[['ts', 'date', 'block', *feature_names]],
                    keys=('ts',))
        batch.frame(prefix+'correlation.parquet', frame[[x['name'] for x in factor_catalog()]][
            frame.date.between(*map(str, spec['splits']['development']))].corr(method='spearman'),
            index=True)
    return {'ok': True, 'report_id': report_id, 'path': str(root/prefix),
            'samples': len(frame), 'selected': selection['selected']}


def illustrative_candidate(metrics):
    rows = metrics[(metrics.split == 'development') & (metrics.fee_bps_per_side == 0)]
    if rows.empty:
        rows = metrics[metrics.split == 'development']
    return rows.loc[rows.rank_ic.abs().idxmax()]


def report_summary(manifest, metrics):
    candidate = illustrative_candidate(metrics)
    rows = metrics[(metrics.factor == candidate.factor)
                   & (metrics.horizon_seconds == candidate.horizon_seconds)]
    lines = ['# AlphaGYM BTC-USDT 高频探索', '',
             f"报告：{manifest['report_id']}；盘口样本：{manifest['samples']:,}。",
             f"开发/验证/测试：{manifest['spec']['splits']}。", '',
             f"预设成本门槛通过数：{len(manifest['selection']['passing_candidates'])}。",
             '一周探索不能证明长期可交易收益。', '',
             '## 开发段价格预测相关性最高的示例（不是成本门槛通过的策略）', '',
             f"因子：{candidate.factor}；持有期：{int(candidate.horizon_seconds)} 秒。", '',
             '|阶段|单边费率 bps|Rank IC|笔数|每笔净 bps|纯多头每笔净 bps|',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows.itertuples():
        lines.append(f'|{row.split}|{row.fee_bps_per_side}|{row.rank_ic:.4f}|{row.trades}'
                     f'|{row.mean_net_bps:.4f}|{row.long_only_mean_net_bps:.4f}|')
    lines += ['', '## 限制', ''] + ['- '+x for x in manifest['limitations']]
    return '\n'.join(lines)+'\n'


def render_report(manifest, metrics, daily):
    fee = manifest['selection']['primary_fee_bps_per_side']
    primary = metrics[metrics.fee_bps_per_side == fee]
    labels = {'factor':'因子', 'horizon_seconds':'持有秒数', 'split':'阶段',
              'rank_ic':'Rank IC', 'trades':'笔数', 'mean_net_bps':'每笔净收益 bps',
              'long_only_mean_net_bps':'纯多头每笔净收益 bps',
              'break_even_fee_bps_per_side':'单边盈亏平衡费率 bps',
              'fee_bps_per_side':'单边手续费 bps'}
    split_labels = {'development':'开发', 'validation':'验证', 'test':'测试'}
    table_frame = primary[['factor', 'horizon_seconds', 'split', 'rank_ic', 'trades',
                     'mean_net_bps', 'long_only_mean_net_bps',
                     'break_even_fee_bps_per_side']].copy()
    table_frame['split'] = table_frame.split.map(split_labels)
    table = table_frame.rename(columns=labels).to_html(
        index=False, float_format=lambda x:f'{x:.4f}')
    candidate = illustrative_candidate(metrics)
    example = metrics[(metrics.factor == candidate.factor)
                      & (metrics.horizon_seconds == candidate.horizon_seconds)].copy()
    example['split'] = example.split.map(split_labels)
    example_table = example[['split', 'fee_bps_per_side', 'rank_ic', 'mean_net_bps',
                             'long_only_mean_net_bps']].rename(columns=labels).to_html(
        index=False, float_format=lambda x:f'{x:.4f}')
    # Embed local ECharts so the explicitly exported report works fully offline.
    echarts = (Path(__file__).parent/'static'/'echarts.min.js').read_text(encoding='utf-8')
    ranked = primary[(primary.split == 'development') & (primary.horizon_seconds == 5)].copy()
    ranked['abs_ic'] = ranked.rank_ic.abs()
    chart_names = ranked.nlargest(12, 'abs_ic').factor.tolist()
    chart = daily[(daily.fee_bps_per_side == fee) & (daily.horizon_seconds == 5)
                  & daily.factor.isin(chart_names)]
    dates = sorted(chart.date.unique())
    series = [{'name': name, 'type': 'line', 'data': [
        None if not np.isfinite(v) else float(v)
        for v in group.set_index('date').reindex(dates).mean_net_bps
    ]} for name, group in chart.groupby('factor')]
    option = {'tooltip': {'trigger':'axis'}, 'legend': {'type':'scroll'},
              'xAxis': {'type':'category', 'data':dates},
              'yAxis': {'type':'value', 'name':'每笔净收益 bps', 'scale':True},
              'series': series}
    limitations = ''.join('<li>'+html.escape(x)+'</li>' for x in manifest['limitations'])
    selected = ('没有候选通过' if not manifest['selection']['selected'] else
                html.escape(json.dumps(manifest['selection']['selected'], ensure_ascii=False)))
    splits = '；'.join(f'{split_labels[k]}：{v[0]} 至 {v[1]}'
                      for k, v in manifest['spec']['splits'].items())
    return f'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>AlphaGYM BTC-USDT 高频探索</title><style>
body{{font:15px system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#183047}}
table{{border-collapse:collapse;font-size:12px}}td,th{{padding:7px;border:1px solid #ddd}}
th{{background:#eef3f8}}h1{{color:#123b5d}}#chart{{height:480px}}code{{word-break:break-all}}
</style><h1>AlphaGYM — BTC-USDT 高频因子探索</h1>
<p>探索实验 · OKX 现货 · 秒级信号 · 一周样本</p>
<p>报告 {manifest['report_id']} · {manifest['samples']:,} 个盘口样本</p>
<p>{html.escape(splits)}（UTC）</p>
<p>信号后延迟至少一秒成交，按买卖报价往返，固定 UTC 相位的不重叠持仓；
主成本情景单边 {fee} bps。多空诊断含假设空头，现货多头指标单独列示。</p>
<p><strong>成本门槛：{selected}。</strong>仅开发与验证段参与选择，不能从测试榜单重新挑选。</p>
<h2>信号示例：{html.escape(candidate.factor)} · {int(candidate.horizon_seconds)} 秒</h2>
<p>下表示例按开发段 |Rank IC| 选取，用于解释价格预测与扣费收益的区别，
不代表通过成本门槛。1 bps = 0.01%；0 费率仍包含买卖价差。</p>{example_table}
<h2>5 秒持有期：每日扣费均值</h2><div id="chart"></div>
<details><summary>查看全部 {len(manifest['factors'])} 个因子的分段结果（主成本情景）</summary>{table}</details>
<h2>边界与局限</h2><ul>{limitations}</ul>
<p>研究依据：<a href="https://arxiv.org/abs/1011.6402">Cont 等：OFI</a> ·
<a href="https://arxiv.org/abs/1512.03492">Gould/Bonart：队列不平衡</a> ·
<a href="https://arxiv.org/abs/1907.06230">Xu 等：多档订单流</a>。
多档订单流论文研究同期价格解释，不直接等于未来收益预测；当前实现事件级最佳档 OFI 和多档静态深度。</p>
<script>{echarts}</script><script>echarts.init(document.getElementById('chart')).setOption(
{json.dumps(option).replace('</', '<\\/')});</script></html>'''
