"""Point-in-time hourly research for liquid USDT perpetual swaps."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from alphagym import storage_io
from alphagym.okx_api import OKXDemoClient

BAR_HOURS = {'1H': 1, '2H': 2, '4H': 4}
DEFAULT_CORE_UNIVERSE = (
    'BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'XRP-USDT-SWAP',
    'DOGE-USDT-SWAP', 'ADA-USDT-SWAP', 'LTC-USDT-SWAP', 'BCH-USDT-SWAP',
    'LINK-USDT-SWAP', 'DOT-USDT-SWAP', 'AVAX-USDT-SWAP', 'TRX-USDT-SWAP',
)
FACTOR_COLUMNS = (
    'momentum_1', 'momentum_3', 'momentum_6', 'momentum_12', 'momentum_24',
    'volatility_6', 'volatility_24', 'volume_surprise_24',
    'range_position_24', 'candle_strength', 'illiquidity_24',
    'volume_confirmed_momentum_3', 'residual_momentum_12',
    'funding_rate', 'funding_z_42', 'funding_change',
)


@dataclass(frozen=True)
class HourlySpec:
    bar: str = '4H'
    universe_size: int = 20
    history_days: int = 365
    horizons: tuple[int, ...] = (1, 2, 6)
    fee_bps_per_side: float = 5.0
    minimum_bars: int = 250
    instruments: tuple[str, ...] = DEFAULT_CORE_UNIVERSE
    workers: int = 4
    minimum_listing_age_before_sample_days: int = 365

    def validate(self):
        if self.bar not in BAR_HOURS:
            raise ValueError(f'bar must be one of {tuple(BAR_HOURS)}')
        if self.universe_size < 5:
            raise ValueError('cross-sectional research requires at least 5 instruments')
        if self.history_days < 30 or self.minimum_bars < 100:
            raise ValueError('insufficient requested history')
        if not 1 <= self.workers <= 8:
            raise ValueError('workers must be between 1 and 8')
        if self.minimum_listing_age_before_sample_days < 0:
            raise ValueError('minimum listing age before sample must be nonnegative')


def _utc_ms(value):
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
    return int(stamp.timestamp() * 1000)


def liquid_usdt_swaps(client: OKXDemoClient, count=20, *, listed_before_ms=None):
    instruments = {row['instId']: row for row in client.list_instruments('SWAP')
                   if row.get('state') == 'live' and row.get('settleCcy') == 'USDT'
                   and row.get('ctType') == 'linear'
                   and (listed_before_ms is None
                        or int(row.get('listTime') or 0) <= int(listed_before_ms))}
    tickers = [row for row in client.tickers('SWAP') if row.get('instId') in instruments]
    # volCcy24h is denominated in the underlying coin and cannot be compared
    # directly across BTC, ETH and low-price tokens. Convert it to quote notional.
    tickers.sort(key=lambda row: float(row.get('volCcy24h') or 0)
                 * float(row.get('last') or 0), reverse=True)
    return [row['instId'] for row in tickers[:count]]


def download_panel(spec: HourlySpec, client=None, *, now=None):
    spec.validate()
    client = client or OKXDemoClient()
    now = pd.Timestamp(now or datetime.now(UTC)).tz_convert('UTC')
    start = now - timedelta(days=spec.history_days)
    # Hourly research defaults to a full year before sample start. Callers that
    # construct a point-in-time universe may download younger contracts and
    # apply the listing-age rule independently at each signal date.
    listing_cutoff = start - timedelta(days=spec.minimum_listing_age_before_sample_days)
    catalogue = {row['instId']: row for row in client.list_instruments('SWAP')}
    requested = spec.instruments[:spec.universe_size] if spec.instruments else tuple(
        liquid_usdt_swaps(client, spec.universe_size,
                          listed_before_ms=_utc_ms(listing_cutoff)))
    instruments = []
    for instrument in requested:
        row = catalogue.get(instrument)
        if (row and row.get('state') == 'live' and row.get('settleCcy') == 'USDT'
                and row.get('ctType') == 'linear'
                and int(row.get('listTime') or 0) <= _utc_ms(listing_cutoff)):
            instruments.append(instrument)
    if len(instruments) < 5:
        raise ValueError('fewer than five instruments meet the download listing-age rule')
    pages = int(np.ceil(spec.history_days * 24 / BAR_HOURS[spec.bar] / 300)) + 1
    columns = ('ts', 'open', 'high', 'low', 'close', 'volume_contracts',
               'volume_base', 'volume_quote', 'confirm')

    def fetch(instrument):
        rows = client.candle_history(
            instrument, bar=spec.bar, start_ms=_utc_ms(start), end_ms=_utc_ms(now),
            max_pages=pages, pause_seconds=.45)
        if not rows:
            return None
        frame = pd.DataFrame(rows, columns=columns)
        frame['instrument'] = instrument
        frame['listing_time'] = pd.to_datetime(
            int(catalogue[instrument]['listTime']), unit='ms', utc=True)
        for column in columns[1:8]:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
        frame['ts'] = pd.to_datetime(pd.to_numeric(frame['ts']), unit='ms', utc=True)
        return frame.drop(columns='confirm')

    frames = []
    with ThreadPoolExecutor(max_workers=min(spec.workers, len(instruments))) as executor:
        futures = {executor.submit(fetch, instrument): instrument for instrument in instruments}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                frames.append(result)
    if not frames:
        raise ValueError('OKX returned no confirmed candles')
    panel = pd.concat(frames, ignore_index=True).sort_values(['ts', 'instrument'])
    funding_pages = int(np.ceil(spec.history_days * 3 / 100)) + 1

    def fetch_funding(instrument):
        rows = client.funding_rate_history(
            instrument, start_ms=_utc_ms(start), end_ms=_utc_ms(now),
            max_pages=funding_pages, pause_seconds=.45)
        if not rows:
            return pd.DataFrame(columns=['ts', 'instrument', 'funding_event'])
        return pd.DataFrame({
            'ts': pd.to_datetime([int(row['fundingTime']) for row in rows], unit='ms', utc=True),
            'instrument': instrument,
            'funding_event': [float(row.get('realizedRate') or row['fundingRate']) for row in rows],
        })

    funding = []
    with ThreadPoolExecutor(max_workers=min(spec.workers, len(instruments))) as executor:
        futures = [executor.submit(fetch_funding, instrument) for instrument in instruments]
        for future in as_completed(futures):
            funding.append(future.result())
    funding = pd.concat(funding, ignore_index=True)
    panel = panel.merge(funding, on=['ts', 'instrument'], how='left')
    panel['funding_rate'] = panel.groupby('instrument', observed=True)['funding_event'].ffill()
    panel['funding_event'] = panel['funding_event'].fillna(0.0)
    counts = panel.groupby('instrument').size()
    keep = counts[counts >= spec.minimum_bars].index
    panel = panel[panel['instrument'].isin(keep)].reset_index(drop=True)
    if panel['instrument'].nunique() < 5:
        raise ValueError('fewer than five instruments have sufficient history')
    return panel


def compute_factors(panel):
    required = {'ts', 'instrument', 'open', 'high', 'low', 'close', 'volume_quote'}
    missing = required - set(panel)
    if missing:
        raise ValueError(f'missing candle columns: {sorted(missing)}')
    data = panel.sort_values(['instrument', 'ts']).copy()
    grouped = data.groupby('instrument', observed=True, group_keys=False)
    data['log_return'] = grouped['close'].transform(lambda x: np.log(x).diff())
    for window in (1, 3, 6, 12, 24):
        data[f'momentum_{window}'] = grouped['close'].transform(
            lambda x, w=window: np.log(x).diff(w))
    data['reversal_1'] = -data['momentum_1']
    for window in (6, 24):
        data[f'volatility_{window}'] = grouped['log_return'].transform(
            lambda x, w=window: x.rolling(w, min_periods=max(3, w // 2)).std())
    log_volume = np.log1p(data['volume_quote'].clip(lower=0))
    data['volume_surprise_24'] = log_volume - log_volume.groupby(data['instrument']).transform(
        lambda x: x.rolling(24, min_periods=12).median())
    rolling_low = grouped['low'].transform(lambda x: x.rolling(24, min_periods=12).min())
    rolling_high = grouped['high'].transform(lambda x: x.rolling(24, min_periods=12).max())
    data['range_position_24'] = (data['close'] - rolling_low) / (rolling_high - rolling_low) - .5
    data['candle_strength'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    raw_illiq = data['log_return'].abs() / data['volume_quote'].replace(0, np.nan)
    data['illiquidity_24'] = raw_illiq.groupby(data['instrument']).transform(
        lambda x: np.log(x.rolling(24, min_periods=12).mean().clip(lower=1e-16)))
    data['volume_confirmed_momentum_3'] = data['momentum_3'] * data['volume_surprise_24']
    market = data.groupby('ts', observed=True)['momentum_12'].transform('median')
    data['residual_momentum_12'] = data['momentum_12'] - market
    if 'funding_rate' not in data:
        data['funding_rate'] = np.nan
    if 'funding_event' not in data:
        data['funding_event'] = 0.0
    funding_mean = grouped['funding_rate'].transform(
        lambda x: x.rolling(42, min_periods=14).mean())
    funding_std = grouped['funding_rate'].transform(
        lambda x: x.rolling(42, min_periods=14).std())
    data['funding_z_42'] = (data['funding_rate'] - funding_mean) / funding_std.replace(0, np.nan)
    data['funding_change'] = grouped['funding_rate'].diff()
    return data


def _rank_ic(frame, factor, label):
    values = frame.groupby('ts', observed=True).apply(
        lambda x: x[factor].corr(x[label], method='spearman'), include_groups=False)
    return float(values.mean()) if len(values) else np.nan


def _time_series_ic(frame, factor, label):
    values = frame.groupby('instrument', observed=True).apply(
        lambda x: x[factor].corr(x[label], method='spearman'), include_groups=False)
    return float(values.mean()) if len(values) else np.nan


def _portfolio_returns(frame, factor, label, style, direction, fee_bps):
    clean = frame[['ts', 'instrument', factor, label]].dropna().copy()
    if style == 'cross_sectional':
        ranks = clean.groupby('ts', observed=True)[factor].rank(pct=True)
        clean['position'] = np.where(ranks >= .8, direction, np.where(ranks <= .2, -direction, 0.0))
    else:
        center = clean.groupby('instrument', observed=True)[factor].transform(
            lambda x: x.rolling(84, min_periods=24).median())
        clean['position'] = np.sign(clean[factor] - center) * direction
    gross = clean.groupby('ts', observed=True)['position'].transform(lambda x: x.abs().sum())
    clean['weight'] = clean['position'] / gross.replace(0, np.nan)
    previous = clean.pivot(index='ts', columns='instrument', values='weight').fillna(0).shift().fillna(0)
    current = clean.pivot(index='ts', columns='instrument', values='weight').fillna(0)
    turnover = (current - previous).abs().sum(axis=1)
    pnl = (clean['weight'] * clean[label]).groupby(clean['ts']).sum().reindex(current.index)
    # A unit of turnover is one changed leg; each leg pays one-side fees.
    net = pnl - turnover * fee_bps / 10_000
    return net, turnover


def evaluate(panel, spec: HourlySpec):
    data = compute_factors(panel)
    times = pd.Index(sorted(data['ts'].unique()))
    dev_end, val_end = times[int(len(times) * .6)], times[int(len(times) * .8)]
    data['split'] = np.where(data['ts'] < dev_end, 'development',
                             np.where(data['ts'] < val_end, 'validation', 'test'))
    bar_number = {ts: number for number, ts in enumerate(times)}
    data['bar_number'] = data['ts'].map(bar_number)
    rows = []
    curves = []
    for horizon in spec.horizons:
        label = f'forward_open_return_{horizon}'
        data[label] = data.groupby('instrument', observed=True)['open'].transform(
            lambda x, h=horizon: np.log(x.shift(-(h + 1)) / x.shift(-1)))
        forward_funding = sum(
            data.groupby('instrument', observed=True)['funding_event'].shift(-step).fillna(0)
            for step in range(1, horizon + 1))
        # Positive funding is paid by longs and received by shorts.
        data[label] = data[label] - forward_funding
        sampled = data[data['bar_number'] % horizon == 0]
        for factor in FACTOR_COLUMNS:
            dev = sampled[sampled['split'] == 'development']
            for style in ('cross_sectional', 'time_series'):
                ic = (_rank_ic(dev, factor, label) if style == 'cross_sectional'
                      else _time_series_ic(dev, factor, label))
                direction = 1 if not np.isfinite(ic) or ic >= 0 else -1
                net, turnover = _portfolio_returns(
                    sampled, factor, label, style, direction, spec.fee_bps_per_side)
                indexed_split = sampled.drop_duplicates('ts').set_index('ts')['split'].reindex(net.index)
                for split in ('development', 'validation', 'test'):
                    sample = net[indexed_split == split].dropna()
                    ann = 365 * 24 / BAR_HOURS[spec.bar]
                    mean = float(sample.mean()) if len(sample) else np.nan
                    std = float(sample.std(ddof=1)) if len(sample) > 1 else np.nan
                    rows.append({
                        'factor': factor, 'style': style, 'horizon_bars': horizon,
                        'split': split, 'direction': direction, 'development_rank_ic': ic,
                        'observations': len(sample), 'mean_return_bps': mean * 10_000,
                        'sharpe': mean / std * np.sqrt(ann / horizon) if std > 0 else np.nan,
                        'average_turnover': float(turnover[indexed_split == split].mean()),
                    })
                curve = pd.DataFrame({'ts': net.index, 'net_return': net.values})
                curve['factor'], curve['style'], curve['horizon_bars'] = factor, style, horizon
                curve['cumulative_return'] = np.exp(curve['net_return'].fillna(0).cumsum()) - 1
                curves.append(curve)
    summary = pd.DataFrame(rows)
    # Selection sees development and validation only; test is reporting-only.
    eligible = summary[summary['split'].isin(['development', 'validation'])].pivot_table(
        index=['factor', 'style', 'horizon_bars'], columns='split', values='sharpe')
    if {'development', 'validation'} <= set(eligible):
        eligible['selection_score'] = eligible[['development', 'validation']].min(axis=1)
        selected = eligible.sort_values('selection_score', ascending=False).reset_index().head(10)
    else:
        selected = pd.DataFrame()
    return data, summary, pd.concat(curves, ignore_index=True), selected


def _publish_hourly_research(root, spec, panel, *, source_report_id=None):
    root = Path(root).expanduser().resolve()
    features, summary, curves, selected = evaluate(panel, spec)
    report_id = 'crypto-' + hashlib.sha256(
        f'{datetime.now(UTC).isoformat()}:{spec}'.encode()).hexdigest()[:16]
    manifest = {
        'ok': True, 'report_id': report_id, 'created_at': datetime.now(UTC).isoformat(),
        'market': 'OKX USDT linear perpetual swaps', 'bar': spec.bar,
        'universe': sorted(panel['instrument'].unique().tolist()),
        'rows': len(panel), 'start': panel['ts'].min().isoformat(),
        'end': panel['ts'].max().isoformat(), 'factors': list(FACTOR_COLUMNS),
        'horizons_bars': list(spec.horizons), 'fee_bps_per_side': spec.fee_bps_per_side,
        'funding_in_return': True,
        'source_report_id': source_report_id,
        'selection_rule': 'direction on development IC; rank by min(development, validation) Sharpe; test untouched',
        'universe_rule': ('explicit fixed core universe; every contract listed at least '
                          f'{spec.minimum_listing_age_before_sample_days} days before sample start'),
        'selected': json.loads(selected.to_json(orient='records')),
    }
    with storage_io.store_for(root, initialize=True).batch() as batch:
        prefix = f'factor_library/reports/{report_id}'
        batch.frame(f'{prefix}/candles.parquet', panel)
        batch.frame(f'{prefix}/features.parquet', features)
        batch.frame(f'{prefix}/summary.csv', summary)
        batch.frame(f'{prefix}/curves.parquet', curves)
        batch.frame(f'{prefix}/selected.csv', selected)
        batch.blob(f'{prefix}/manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2).encode())
    return manifest


def run_hourly_research(root, spec: HourlySpec, *, client=None, now=None):
    panel = download_panel(spec, client, now=now)
    return _publish_hourly_research(root, spec, panel)


def reanalyze_hourly_report(root, source_report_id, spec: HourlySpec):
    root = Path(root).expanduser().resolve()
    if not source_report_id.startswith('crypto-'):
        raise ValueError('invalid crypto source report ID')
    source = root / 'factor_library' / 'reports' / source_report_id
    manifest = json.loads(storage_io.read_text(source / 'manifest.json'))
    if manifest.get('bar') != spec.bar:
        raise ValueError('source report bar does not match requested bar')
    panel = storage_io.read_frame(source / 'candles.parquet')
    return _publish_hourly_research(root, spec, panel, source_report_id=source_report_id)
