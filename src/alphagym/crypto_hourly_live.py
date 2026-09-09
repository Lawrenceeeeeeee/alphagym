"""Small, auditable OKX demo portfolio for frozen hourly research signals."""
from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from pathlib import Path

import pandas as pd

from alphagym import storage_io
from alphagym.crypto_hourly import compute_factors
from alphagym.equity_data import DataContractError
from alphagym.hf_live import _market
from alphagym.okx_api import OKXCredentials, OKXDemoClient

ALLOWED_REPORT = 'crypto-0f6bc5eb4b79046f'
ALLOWED_FACTOR = 'momentum_1'
ALLOWED_HORIZON_BARS = 6


def _rounded_contracts(instrument, price, notional):
    lot = Decimal(instrument['lotSz'])
    minimum = Decimal(instrument['minSz'])
    contract_value = Decimal(instrument.get('ctVal') or '1')
    raw = Decimal(str(notional)) / (Decimal(str(price)) * contract_value)
    return max(minimum, (raw / lot).to_integral_value(rounding=ROUND_UP) * lot)


def build_demo_plan(root, report_id=ALLOWED_REPORT, *, notional_per_leg=25, client=None):
    if report_id != ALLOWED_REPORT:
        raise DataContractError('Only the frozen 4H candidate report is allowed for demo trading')
    root = Path(root).expanduser().resolve()
    source = root / 'factor_library' / 'reports' / report_id
    manifest = json.loads(storage_io.read_text(source / 'manifest.json'))
    if manifest.get('bar') != '4H' or manifest.get('fee_bps_per_side') != 5.0:
        raise DataContractError('Frozen demo report must be the audited 4H / 5 bps specification')
    candles = storage_io.read_frame(source / 'candles.parquet')
    client = client or OKXDemoClient()
    refreshed = []
    columns = ('ts', 'open', 'high', 'low', 'close', 'volume_contracts',
               'volume_base', 'volume_quote', 'confirm')
    for instrument in manifest['universe']:
        rows = client.candle_history(instrument, bar='4H', max_pages=1, pause_seconds=0)
        if rows:
            frame = pd.DataFrame(rows, columns=columns)
            frame['instrument'] = instrument
            frame['ts'] = pd.to_datetime(pd.to_numeric(frame['ts']), unit='ms', utc=True)
            for column in columns[1:8]:
                frame[column] = pd.to_numeric(frame[column], errors='coerce')
            refreshed.append(frame.drop(columns='confirm'))
    if refreshed:
        candles = pd.concat([candles, *refreshed], ignore_index=True)
        candles = candles.sort_values(['instrument', 'ts']).drop_duplicates(
            ['instrument', 'ts'], keep='last')
    features = compute_factors(candles).sort_values(['instrument', 'ts'])
    features['center'] = features.groupby('instrument', observed=True)[ALLOWED_FACTOR].transform(
        lambda x: x.rolling(84, min_periods=24).median())
    latest_common = features.groupby('ts', observed=True)['instrument'].nunique()
    signal_ts = latest_common[latest_common == features['instrument'].nunique()].index.max()
    latest = features[features['ts'] == signal_ts].copy()
    latest['side'] = (latest[ALLOWED_FACTOR] - latest['center']).map(
        lambda value: 'buy' if value > 0 else 'sell')
    latest['direction'] = latest['side'].map({'buy': 'long', 'sell': 'short'})
    latest['notional_usdt'] = float(notional_per_leg)
    latest['entry_due_at'] = pd.Timestamp(signal_ts) + timedelta(hours=4)
    return latest[['ts', 'instrument', 'close', ALLOWED_FACTOR, 'center', 'side',
                   'direction', 'notional_usdt', 'entry_due_at']].reset_index(drop=True)


def start_demo_portfolio(root, report_id=ALLOWED_REPORT, *, notional_per_leg=25,
                         execute_demo=False, client=None):
    root = Path(root).expanduser().resolve()
    client = client or OKXDemoClient(OKXCredentials.from_env())
    plan = build_demo_plan(
        root, report_id, notional_per_leg=notional_per_leg, client=client)
    config = client.account_config()
    if config.get('posMode') != 'net_mode':
        raise DataContractError('Hourly demo requires OKX net position mode')
    if execute_demo and 'trade' not in str(config.get('perm', '')):
        raise DataContractError('OKX demo API key does not have trade permission')
    entry_due = pd.Timestamp(plan['entry_due_at'].iloc[0]).to_pydatetime()
    if execute_demo and not timedelta(0) <= datetime.now(UTC) - entry_due <= timedelta(minutes=5):
        raise DataContractError('Demo orders are only allowed in the first five minutes of a 4H bar')
    existing = client.positions()
    nonzero = [row for row in existing if Decimal(row.get('pos') or '0') != 0]
    if nonzero:
        raise DataContractError('Demo portfolio requires a zero-position account at startup')
    session_id = 'cryptopaper-' + uuid.uuid4().hex[:16]
    prepared = []
    for row in plan.to_dict(orient='records'):
        instrument = client.instruments(row['instrument'])
        book = client.book(row['instrument'], depth=1)
        price = Decimal(book['asks'][0][0] if row['side'] == 'buy' else book['bids'][0][0])
        contracts = _rounded_contracts(instrument, price, row['notional_usdt'])
        prepared.append({
            'instrument': row['instrument'], 'direction': row['direction'],
            'signal_ts': row['ts'], 'signal_close': row['close'],
            'requested_notional_usdt': row['notional_usdt'], 'side': row['side'],
            'requested_size': str(contracts),
        })
    started = datetime.now(UTC)
    status = {
        'ok': True, 'session_id': session_id, 'status': 'opening' if execute_demo else 'planned',
        'environment': 'demo',
        'execute_demo': execute_demo, 'report_id': report_id, 'factor': ALLOWED_FACTOR,
        'bar': '4H', 'holding_bars': ALLOWED_HORIZON_BARS,
        'started_at': started.isoformat(),
        'entry_due_at': entry_due.isoformat(),
        'close_due_at': (started + timedelta(hours=24)).isoformat(),
        'notional_per_leg': float(notional_per_leg), 'legs': len(prepared),
        'filled_legs': 0, 'error': None,
    }
    prefix = f'factor_library/crypto_live/{session_id}'
    with storage_io.store_for(root).batch() as batch:
        batch.frame(f'{prefix}/plan.parquet', plan)
        batch.frame(f'{prefix}/prepared.parquet', pd.DataFrame(prepared))
        batch.blob(f'{prefix}/status.json', json.dumps(status, indent=2).encode())
    if not execute_demo:
        return status
    fills = []
    try:
        for row in prepared:
            fill = _market(
                client, row['instrument'], row['side'], Decimal(row['requested_size']))
            fills.append({**row, **fill})
            storage_io.write_frame(
                pd.DataFrame(fills), root / f'{prefix}/fills.parquet', index=False)
        if len(fills) != len(prepared) or any(
                Decimal(row['filled_size']) <= 0 for row in fills):
            raise DataContractError('One or more demo legs did not fill')
        status['status'] = 'open'
        status['filled_legs'] = len(fills)
    except Exception as error:
        rollback = _close_filled_legs(client, fills)
        status.update({'ok': False, 'status': 'rolled_back', 'error': type(error).__name__,
                       'rolled_back_legs': len(rollback)})
        if rollback:
            storage_io.write_frame(
                pd.DataFrame(rollback), root / f'{prefix}/rollback.parquet', index=False)
        storage_io.write_text(
            root / f'{prefix}/status.json', json.dumps(status, indent=2))
        raise
    storage_io.write_text(root / f'{prefix}/status.json', json.dumps(status, indent=2))
    return status


def _close_filled_legs(client, fills):
    position_map = {row['instId']: Decimal(row.get('pos') or '0')
                    for row in client.positions()}
    closes = []
    for fill in fills:
        position = position_map.get(fill['instrument'], Decimal(0))
        expected_sign = 1 if fill['direction'] == 'long' else -1
        if position * expected_sign <= 0:
            continue
        lot = Decimal(client.instruments(fill['instrument'])['lotSz'])
        requested = min(abs(position), Decimal(fill['filled_size']))
        close_size = (requested / lot).to_integral_value(rounding=ROUND_DOWN) * lot
        if close_size <= 0:
            continue
        side = 'sell' if expected_sign > 0 else 'buy'
        closes.append({'instrument': fill['instrument'],
                       **_market(client, fill['instrument'], side, close_size,
                                 reduce_only=True)})
    return closes


def close_demo_portfolio(root, session_id, *, client=None):
    root = Path(root).expanduser().resolve()
    prefix = root / 'factor_library' / 'crypto_live' / session_id
    status = json.loads(storage_io.read_text(prefix / 'status.json'))
    if status['status'] != 'open' or not status['execute_demo']:
        raise DataContractError('Session has no open demo portfolio')
    client = client or OKXDemoClient(OKXCredentials.from_env())
    fills = storage_io.read_frame(prefix / 'fills.parquet').to_dict(orient='records')
    closes = _close_filled_legs(client, fills)
    close_map = {row['instrument']: row for row in closes}
    realized = []
    for fill in fills:
        close = close_map.get(fill['instrument'])
        if not close or not fill.get('average_price') or not close.get('average_price'):
            continue
        contracts = min(Decimal(fill['filled_size']), Decimal(close['filled_size']))
        contract_value = Decimal(client.instruments(fill['instrument']).get('ctVal') or '1')
        open_price = Decimal(str(fill['average_price']))
        close_price = Decimal(str(close['average_price']))
        sign = Decimal(1) if fill['direction'] == 'long' else Decimal(-1)
        gross = sign * (close_price - open_price) * contracts * contract_value
        fees = Decimal(str(fill.get('fee') or '0')) + Decimal(str(close.get('fee') or '0'))
        realized.append({
            'instrument': fill['instrument'], 'direction': fill['direction'],
            'contracts': float(contracts), 'open_price': float(open_price),
            'close_price': float(close_price), 'gross_pnl_usdt': float(gross),
            'fees_usdt': float(-fees), 'net_pnl_before_funding_usdt': float(gross + fees),
        })
    status['status'] = 'closed'
    status['closed_at'] = datetime.now(UTC).isoformat()
    status['closed_legs'] = len(closes)
    status['net_pnl_before_funding_usdt'] = sum(
        row['net_pnl_before_funding_usdt'] for row in realized)
    with storage_io.store_for(root).batch() as batch:
        if closes:
            batch.frame(f'factor_library/crypto_live/{session_id}/closes.parquet',
                        pd.DataFrame(closes))
        if realized:
            batch.frame(f'factor_library/crypto_live/{session_id}/realized.parquet',
                        pd.DataFrame(realized))
        batch.blob(f'factor_library/crypto_live/{session_id}/status.json',
                   json.dumps(status, indent=2).encode())
    return status


def read_demo_status(root, session_id):
    root = Path(root).expanduser().resolve()
    if not session_id.startswith('cryptopaper-'):
        raise ValueError('invalid crypto paper session ID')
    return json.loads(storage_io.read_text(
        root / 'factor_library' / 'crypto_live' / session_id / 'status.json'))


def list_demo_sessions(root):
    root = Path(root).expanduser().resolve()
    store = storage_io.store_for(root)
    paths = [row['path'] for row in store.list('factor_library/crypto_live/')
             if row['path'].endswith('/status.json') and row['kind'] != 'deleted']
    sessions = [json.loads(store.read_blob(path)) for path in paths]
    return sorted(sessions, key=lambda row: row['started_at'], reverse=True)


def wait_and_close_demo_portfolio(root, session_id, *, poll_seconds=60):
    """Local process monitor; updates ClickHouse without chat or scheduler heartbeats."""
    root = Path(root).expanduser().resolve()
    while True:
        status = read_demo_status(root, session_id)
        if status['status'] != 'open':
            return status
        due = datetime.fromisoformat(status['close_due_at'])
        remaining = (due - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            break
        time.sleep(min(float(poll_seconds), max(1.0, remaining)))
    try:
        return close_demo_portfolio(root, session_id)
    except Exception as error:
        status = read_demo_status(root, session_id)
        status['monitor_error'] = f'{type(error).__name__}: {error}'
        status['monitor_failed_at'] = datetime.now(UTC).isoformat()
        storage_io.write_text(
            root / 'factor_library' / 'crypto_live' / session_id / 'status.json',
            json.dumps(status, indent=2))
        raise
