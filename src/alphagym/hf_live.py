"""Short, bounded OKX demo-trading sessions for frozen high-frequency models."""
from __future__ import annotations

import asyncio
import json
import os
import pickle
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from pathlib import Path

import pandas as pd

from alphagym import storage_io
from alphagym.config import resolve_root
from alphagym.equity_data import DataContractError
from alphagym.hf_book import BookSampler
from alphagym.hf_factors import compute_features
from alphagym.okx_api import OKXCredentials, OKXDemoClient
from alphagym.optional import require

DEMO_PUBLIC_WS = 'wss://wspap.okx.com:8443/ws/v5/public'


def _status_path(session_id):
    return f'factor_library/hf_live/{session_id}/status.json'


def _write_status(root, row):
    value = {**row, 'updated_at': datetime.now(UTC).isoformat()}
    storage_io.store_for(root, initialize=True).write_blob(
        _status_path(row['session_id']), json.dumps(value).encode())
    return value


def read_live_status(root, session_id):
    return json.loads(storage_io.store_for(resolve_root(root)).read_blob(_status_path(session_id)))


def read_live_log(root, session_id, limit=50):
    store = storage_io.store_for(resolve_root(root))
    path = f'factor_library/hf_live/{session_id}/events.parquet'
    try:
        frame = store.read_frame(path)
    except (FileNotFoundError, KeyError):
        return {'session_id': session_id, 'events': []}
    frame = frame.sort_values('seq').tail(limit)
    return {'session_id': session_id, 'events': frame.to_dict(orient='records')}


def _load_model(root, report_id, horizon):
    if not report_id.startswith('hfml-'):
        raise DataContractError('Live session requires an hfml report')
    store = storage_io.store_for(root)
    prefix = f'factor_library/reports/{report_id}/'
    manifest = json.loads(store.read_blob(prefix+'manifest.json'))
    chosen = next((row for row in manifest['selection']['chosen']
                   if int(row['horizon_seconds']) == horizon), None)
    if chosen is None:
        raise DataContractError(f'Report has no frozen selection for {horizon}s')
    method = chosen['method']
    path = prefix+f'models/{horizon}/{method}.pkl'
    if path.removeprefix(prefix) not in manifest.get('model_resources', []):
        raise DataContractError(f'Frozen model is unavailable for {horizon}s {method}')
    # This is an internal artifact whose hash-bound report was created locally by AlphaGYM.
    model = pickle.loads(store.read_blob(path))
    policy = next(row for row in manifest['policies']
                  if row['method'] == method and int(row['horizon_seconds']) == horizon)
    return manifest, model, policy


def _client_id(prefix):
    return (prefix+datetime.now(UTC).strftime('%m%d%H%M%S%f'))[:32]


def _ioc(client, instrument, side, size, bid, ask):
    tick = Decimal(instrument['tickSz'])
    if side == 'buy':
        price = ((Decimal(str(ask))*Decimal('1.0002'))/tick).to_integral_value(
            rounding=ROUND_UP)*tick
    else:
        price = ((Decimal(str(bid))*Decimal('.9998'))/tick).to_integral_value(
            rounding=ROUND_DOWN)*tick
    placed = client.place_order(
        inst_id='BTC-USDT', side=side, order_type='ioc', size=size, price=price,
        client_order_id=_client_id('agpaper'))
    order = client.order(inst_id='BTC-USDT', order_id=placed['ordId'])
    return {
        'side': side, 'requested_size': str(size), 'limit_price': str(price),
        'state': order.get('state'), 'filled_size': order.get('accFillSz') or '0',
        'average_price': order.get('avgPx') or None,
        'fee': order.get('fee') or None, 'fee_currency': order.get('feeCcy') or None,
    }


def _market(client, inst_id, side, size, *, reduce_only=False):
    is_swap = inst_id.endswith('-SWAP')
    placed = client.place_order(
        inst_id=inst_id, side=side, order_type='market', size=size,
        target_currency=None if is_swap else 'base_ccy',
        trade_mode='cross' if is_swap else 'cash', reduce_only=reduce_only,
        client_order_id=_client_id('agpaper'))
    order = client.order(inst_id=inst_id, order_id=placed['ordId'])
    for _ in range(5):
        if order.get('state') in {'filled', 'canceled'}:
            break
        time.sleep(.1)
        order = client.order(inst_id=inst_id, order_id=placed['ordId'])
    return {
        'side': side, 'requested_size': str(size), 'limit_price': None,
        'state': order.get('state'), 'filled_size': order.get('accFillSz') or '0',
        'average_price': order.get('avgPx') or None,
        'fee': order.get('fee') or None, 'fee_currency': order.get('feeCcy') or None,
    }


def _closable_size(fill, lot):
    size = Decimal(fill['filled_size'])
    if fill['side'] == 'buy' and fill.get('fee_currency') == 'BTC':
        size -= abs(Decimal(fill.get('fee') or '0'))
    size = max(size, Decimal(0))
    return (size/lot).to_integral_value(rounding=ROUND_DOWN)*lot


def pair_round_trips(trades, contract_value=Decimal('.01')):
    """Pair open/close fills and calculate realized linear-contract P&L."""
    completed, opened = [], None
    for row in trades:
        if row.get('event') == 'open':
            opened = row
            continue
        if opened is None or not str(row.get('event', '')).startswith('close'):
            continue
        open_price = Decimal(str(opened['average_price']))
        close_price = Decimal(str(row['average_price']))
        contracts = min(Decimal(str(opened['filled_size'])),
                        Decimal(str(row['filled_size'])))
        base_quantity = contracts*contract_value
        direction = opened['direction']
        gross = ((close_price-open_price) if direction == 'long'
                 else (open_price-close_price))*base_quantity
        fees = Decimal(0)
        for fill, price in ((opened, open_price), (row, close_price)):
            fee = Decimal(str(fill.get('fee') or '0'))
            if fill.get('fee_currency') == 'BTC':
                fee *= price
            elif fill.get('fee_currency') not in {None, '', 'USDT'}:
                continue
            fees += fee
        notional = open_price*base_quantity
        net = gross+fees
        completed.append({
            'trade_id': len(completed)+1, 'direction': direction,
            'open_ts': int(opened['ts']), 'close_ts': int(row['ts']),
            'holding_seconds': (int(row['ts'])-int(opened['ts']))/1000,
            'contracts': float(contracts), 'base_quantity_btc': float(base_quantity),
            'open_price': float(open_price), 'close_price': float(close_price),
            'gross_pnl_usdt': float(gross), 'fees_usdt': float(-fees),
            'net_pnl_usdt': float(net),
            'gross_return_bps': float(gross/notional*Decimal(10000)) if notional else None,
            'net_return_bps': float(net/notional*Decimal(10000)) if notional else None,
        })
        opened = None
    cumulative = 0.0
    for row in completed:
        cumulative += row['net_pnl_usdt']
        row['cumulative_pnl_usdt'] = cumulative
    return completed


def _event(events, kind, message, **details):
    events.append({
        'seq': len(events)+1, 'at': datetime.now(UTC).isoformat(),
        'kind': kind, 'message': message, 'details': json.dumps(details, ensure_ascii=False),
    })


def _flush(root, session_id, signals, trades, events, contract_value=Decimal('.01')):
    store = storage_io.store_for(root)
    prefix = f'factor_library/hf_live/{session_id}/'
    if signals:
        store.write_frame(prefix+'signals.parquet', pd.DataFrame(signals), keys=('ts',))
    if trades:
        store.write_frame(prefix+'trades.parquet', pd.DataFrame(trades), keys=('ts', 'event'))
        round_trips = pair_round_trips(trades, contract_value)
        if round_trips:
            store.write_frame(prefix+'round_trips.parquet', pd.DataFrame(round_trips),
                              keys=('trade_id',))
    if events:
        store.write_frame(prefix+'events.parquet', pd.DataFrame(events), keys=('seq',))


async def run_live_session(root, report_id, *, horizon=5, duration_minutes=60,
                           execute_demo=True, quantity_btc=Decimal('.0001'), session_id=None,
                           allow_short=True, instrument_id='BTC-USDT-SWAP',
                           min_edge_bps=12.0):
    require('websockets', 'live')
    from websockets.asyncio.client import connect

    root = resolve_root(root)
    manifest, model, policy = _load_model(root, report_id, horizon)
    credentials = OKXCredentials.from_env()
    client = OKXDemoClient(credentials)
    config = await asyncio.to_thread(client.account_config)
    if 'trade' not in str(config.get('perm', '')):
        raise DataContractError('OKX demo API key does not have trade permission')
    instrument = await asyncio.to_thread(client.instruments, instrument_id)
    lot = Decimal(instrument['lotSz'])
    contract_value = Decimal(instrument.get('ctVal') or '1')
    requested_size = quantity_btc/contract_value if instrument_id.endswith('-SWAP') else quantity_btc
    quantity = max(Decimal(instrument['minSz']),
                   (requested_size/lot).to_integral_value(rounding=ROUND_UP)*lot)
    positions = await asyncio.to_thread(client.positions, instrument_id)
    baseline = sum((Decimal(row.get('pos') or '0') for row in positions), Decimal(0))
    session_id = session_id or 'hfpaper-'+uuid.uuid4().hex[:16]
    base = {
        'session_id': session_id, 'status': 'running', 'phase': 'warming_up',
        'environment': 'demo', 'report_id': report_id,
        'instrument_id': instrument_id,
        'source_report_id': manifest['source_report_id'], 'horizon_seconds': horizon,
        'method': model['method'], 'execute_demo': bool(execute_demo),
        'allow_short': bool(allow_short),
        'requested_quantity_btc': str(quantity_btc),
        'order_size_contracts': str(quantity),
        'started_at': datetime.now(UTC).isoformat(),
        'baseline_position_contracts': str(baseline),
        'contract_value': str(contract_value),
        'minimum_live_edge_bps': float(min_edge_bps),
        'signals': 0, 'orders': 0, 'completed_round_trips': 0, 'error': None,
    }
    _write_status(root, base)
    stop_at = datetime.now(UTC)+timedelta(minutes=duration_minutes)
    sampler, source_row, processed = BookSampler(), 0, 0
    signals, trades, events, pending, position = [], [], [], None, None
    _event(events, 'session_start', '会话已启动，正在连接 OKX 模拟盘',
           report_id=report_id, horizon_seconds=horizon, allow_short=allow_short,
           instrument_id=instrument_id, order_size=str(quantity),
           baseline_position_contracts=str(baseline))
    _flush(root, session_id, signals, trades, events)
    last_flush = datetime.now(UTC)
    try:
        async with connect(DEMO_PUBLIC_WS, ping_interval=20, ping_timeout=20) as websocket:
            await websocket.send(json.dumps({'op': 'subscribe', 'args': [
                {'channel': 'books', 'instId': instrument_id}]}))
            _event(events, 'connected', '已连接订单簿，开始积累 1 秒样本')
            while datetime.now(UTC) < stop_at:
                message = json.loads(await asyncio.wait_for(websocket.recv(), timeout=30))
                if message.get('event') == 'error':
                    raise DataContractError(f"OKX websocket error: {message.get('code')}")
                if 'data' not in message:
                    continue
                action = message.get('action', 'update')
                for data in message['data']:
                    sampler.consume([{'source_row': source_row, 'ts': int(data['ts']),
                                      'action': action, 'bids': data['bids'],
                                      'asks': data['asks']}])
                    source_row += 1
                while processed < len(sampler.rows):
                    row = sampler.rows[processed]
                    processed += 1
                    ts = int(row['ts'])
                    if position and ts >= position['close_ts']:
                        close_side = 'sell' if position['direction'] == 'long' else 'buy'
                        fill = await asyncio.to_thread(
                            _market, client, instrument_id, close_side,
                            Decimal(position['close_size']),
                            reduce_only=position['close_reduce_only'])
                        trades.append({'ts': ts, 'event': 'close',
                                       'direction': position['direction'], **fill})
                        base['completed_round_trips'] += 1
                        _event(events, 'position_closed', '持仓已按持有期平仓',
                               direction=position['direction'], fill=fill)
                        position = None
                    if pending and not position:
                        open_side = 'buy' if pending['direction'] == 'long' else 'sell'
                        if execute_demo:
                            fill = await asyncio.to_thread(
                                _market, client, instrument_id, open_side, quantity,
                                reduce_only=pending['open_reduce_only'])
                        else:
                            fill = {'side': open_side, 'requested_size': str(quantity),
                                    'filled_size': str(quantity), 'average_price': (
                                        str(row['ask']) if open_side == 'buy' else str(row['bid'])),
                                    'state': 'paper', 'fee': None, 'fee_currency': None,
                                    'limit_price': None}
                        trades.append({'ts': ts, 'event': 'open',
                                       'direction': pending['direction'], **fill})
                        base['orders'] += int(execute_demo)
                        _event(events, 'position_opened', '信号订单已成交',
                               direction=pending['direction'], fill=fill)
                        if Decimal(fill['filled_size']) > 0:
                            position = {'direction': pending['direction'],
                                        'filled_size': fill['filled_size'],
                                        'close_size': (fill['filled_size'] if
                                                       instrument_id.endswith('-SWAP') else
                                                       str(_closable_size(fill, lot))),
                                        'close_reduce_only': pending['close_reduce_only'],
                                        'close_ts': ts+horizon*1000}
                        pending = None
                    # Score at the holding-period frequency while consuming every book update.
                    # Recomputing all rolling factors every second creates an avoidable queue.
                    if (ts//1000) % horizon != 0:
                        continue
                    history = pd.DataFrame(sampler.rows[max(0, processed-360):processed])
                    if len(history) < 301:
                        continue
                    feature = compute_features(history, workers=1).iloc[[-1]]
                    x = feature[model['features']]
                    long_edge = float(model['long_estimator'].predict(x)[0])
                    short_edge = float(model['short_estimator'].predict(x)[0])
                    research_threshold = float(policy['predicted_gross_edge_threshold_bps'])
                    threshold = max(research_threshold, float(min_edge_bps))
                    direction = 'long' if long_edge >= short_edge else 'short'
                    edge = max(long_edge, short_edge)
                    act = edge > threshold and position is None and pending is None
                    short_blocked = bool(act and direction == 'short' and not allow_short)
                    if short_blocked:
                        act = False
                    signals.append({'ts': ts, 'direction': direction,
                                    'predicted_long_bps': long_edge,
                                    'predicted_short_bps': short_edge,
                                    'threshold_bps': threshold,
                                    'research_threshold_bps': research_threshold,
                                    'actionable': act,
                                    'short_blocked': short_blocked})
                    base['signals'] += int(act)
                    base['phase'] = 'trading'
                    if act:
                        overlays_existing = ((direction == 'short' and baseline > 0
                                              and quantity <= baseline)
                                             or (direction == 'long' and baseline < 0
                                                 and quantity <= abs(baseline)))
                        pending = {'direction': direction, 'signal_ts': ts,
                                   'open_reduce_only': overlays_existing,
                                   'close_reduce_only': not overlays_existing}
                        _event(events, 'signal', '模型产生可执行信号', direction=direction,
                               predicted_edge_bps=edge, threshold_bps=threshold)
                    elif short_blocked:
                        _event(events, 'signal_skipped', '做空信号已跳过：现货模拟盘默认只做多',
                               predicted_edge_bps=edge, threshold_bps=threshold)
                if datetime.now(UTC)-last_flush >= timedelta(seconds=30):
                    _event(events, 'heartbeat', '进程运行正常', phase=base['phase'],
                           samples=len(sampler.rows), predictions=len(signals),
                           signals=base['signals'], orders=base['orders'])
                    _flush(root, session_id, signals, trades, events)
                    _write_status(root, base)
                    last_flush = datetime.now(UTC)
        if position and execute_demo:
            book = await asyncio.to_thread(client.book, instrument_id, 1)
            close_side = 'sell' if position['direction'] == 'long' else 'buy'
            fill = await asyncio.to_thread(
                _market, client, instrument_id, close_side,
                Decimal(position['close_size']),
                reduce_only=position['close_reduce_only'])
            trades.append({'ts': int(book['ts']), 'event': 'close_at_session_end',
                           'direction': position['direction'], **fill})
        _event(events, 'session_complete', '会话已完成')
        _flush(root, session_id, signals, trades, events)
        return _write_status(root, {**base, 'status': 'succeeded', 'phase': 'complete',
                                    'ended_at': datetime.now(UTC).isoformat()})
    except Exception as error:
        _event(events, 'error', '会话异常退出', code=type(error).__name__, message=str(error))
        if position and execute_demo:
            try:
                book = await asyncio.to_thread(client.book, instrument_id, 1)
                close_side = 'sell' if position['direction'] == 'long' else 'buy'
                fill = await asyncio.to_thread(
                    _market, client, instrument_id, close_side,
                    Decimal(position['close_size']),
                    reduce_only=position['close_reduce_only'])
                trades.append({'ts': int(book['ts']), 'event': 'emergency_close',
                               'direction': position['direction'], **fill})
            except Exception as close_error:  # noqa: BLE001 -- preserve both demo failures
                base['emergency_close_error'] = {
                    'code': type(close_error).__name__, 'message': str(close_error)}
        _flush(root, session_id, signals, trades, events)
        _write_status(root, {**base, 'status': 'failed', 'phase': 'failed',
                             'error': {'code': type(error).__name__, 'message': str(error)}})
        raise


def execute_live(root, report_id, *, horizon, duration_minutes, execute_demo, session_id,
                 allow_short=True, instrument_id='BTC-USDT-SWAP', min_edge_bps=12.0):
    return asyncio.run(run_live_session(
        root, report_id, horizon=horizon, duration_minutes=duration_minutes,
        execute_demo=execute_demo, session_id=session_id, allow_short=allow_short,
        instrument_id=instrument_id, min_edge_bps=min_edge_bps))


def start_live(root, report_id, *, horizon=5, duration_minutes=60, execute_demo=True,
               allow_short=True, instrument_id='BTC-USDT-SWAP', min_edge_bps=12.0):
    root = resolve_root(root)
    _load_model(root, report_id, horizon)
    session_id = 'hfpaper-'+uuid.uuid4().hex[:16]
    row = _write_status(root, {
        'session_id': session_id, 'status': 'queued', 'phase': 'queued',
        'environment': 'demo', 'report_id': report_id, 'horizon_seconds': horizon,
        'instrument_id': instrument_id,
        'duration_minutes': duration_minutes, 'execute_demo': bool(execute_demo),
        'allow_short': bool(allow_short),
        'minimum_live_edge_bps': float(min_edge_bps),
        'error': None,
    })
    command = [sys.executable, '-m', 'alphagym.cli', 'hf', 'execute-live',
               '--root', str(root), '--report-id', report_id, '--horizon', str(horizon),
               '--duration-minutes', str(duration_minutes), '--session-id', session_id, '--json']
    command.extend(['--instrument-id', instrument_id])
    command.extend(['--min-edge-bps', str(min_edge_bps)])
    if execute_demo:
        command.append('--execute-demo')
    if allow_short:
        command.append('--allow-short')
    process = subprocess.Popen(
        command, cwd=str(Path(__file__).resolve().parents[2]), env={**os.environ},
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return {**row, 'pid': process.pid}
