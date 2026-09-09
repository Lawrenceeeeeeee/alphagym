"""Minimal OKX V5 client with private requests locked to demo trading."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal

from alphagym.equity_data import DataContractError

BASE_URL = 'https://www.okx.com'


@dataclass(frozen=True)
class OKXCredentials:
    api_key: str
    secret_key: str
    passphrase: str

    @classmethod
    def from_env(cls):
        names = ('OKX_API_KEY', 'OKX_SECRET_KEY', 'OKX_API_PASSPHRASE')
        values = [os.environ.get(name, '').strip() for name in names]
        if not all(values):
            missing = [name for name, value in zip(names, values) if not value]
            raise DataContractError(f'Missing OKX credentials: {", ".join(missing)}')
        return cls(*values)


class OKXAPIError(DataContractError):
    def __init__(self, code, message):
        super().__init__(f'OKX API error {code}: {message}')
        self.code = str(code)


class OKXDemoClient:
    """Authenticated operations always carry the demo environment header."""

    def __init__(self, credentials=None, *, base_url=BASE_URL, timeout=10):
        self.credentials = credentials
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    @staticmethod
    def _timestamp():
        return datetime.now(UTC).isoformat(timespec='milliseconds').replace('+00:00', 'Z')

    def _headers(self, timestamp, method, request_path, body):
        if self.credentials is None:
            raise DataContractError('Private OKX request requires credentials')
        payload = f'{timestamp}{method}{request_path}{body}'.encode()
        signature = base64.b64encode(hmac.new(
            self.credentials.secret_key.encode(), payload, hashlib.sha256).digest()).decode()
        return {
            'Content-Type': 'application/json', 'User-Agent': 'AlphaGYM/0.2',
            'OK-ACCESS-KEY': self.credentials.api_key,
            'OK-ACCESS-SIGN': signature,
            'OK-ACCESS-PASSPHRASE': self.credentials.passphrase,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'x-simulated-trading': '1',
        }

    def request(self, method, path, *, params=None, payload=None, private=False):
        method = method.upper()
        query = urllib.parse.urlencode(params or {})
        request_path = path+('?' + query if query else '')
        body = json.dumps(payload, separators=(',', ':')) if payload is not None else ''
        headers = {'Content-Type': 'application/json', 'User-Agent': 'AlphaGYM/0.2'}
        if private:
            timestamp = self._timestamp()
            headers = self._headers(timestamp, method, request_path, body)
        request = urllib.request.Request(
            self.base_url+request_path, data=body.encode() if body else None,
            headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as error:
            try:
                result = json.loads(error.read())
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise OKXAPIError(error.code, 'HTTP request failed') from None
        except (urllib.error.URLError, TimeoutError) as error:
            raise OKXAPIError('transport', type(error).__name__) from None
        if (str(result.get('code', '0')) != '0' and result.get('data')
                and any('sCode' in row for row in result['data'])):
            return result['data']
        if str(result.get('code', '0')) != '0':
            raise OKXAPIError(result.get('code', 'unknown'), result.get('msg') or 'request failed')
        return result.get('data', [])

    def account_config(self):
        return self.request('GET', '/api/v5/account/config', private=True)[0]

    def balance(self):
        return self.request('GET', '/api/v5/account/balance', private=True)

    def positions(self, inst_id=None):
        params = {'instId': inst_id} if inst_id else None
        return self.request('GET', '/api/v5/account/positions', params=params, private=True)

    def instruments(self, inst_id='BTC-USDT', inst_type=None):
        inst_type = inst_type or ('SWAP' if inst_id.endswith('-SWAP') else 'SPOT')
        rows = self.request('GET', '/api/v5/public/instruments',
                            params={'instType': inst_type, 'instId': inst_id})
        if not rows:
            raise OKXAPIError('empty', f'Instrument not found: {inst_id}')
        return rows[0]

    def list_instruments(self, inst_type='SWAP'):
        """Return the public instrument catalogue without requiring credentials."""
        return self.request('GET', '/api/v5/public/instruments',
                            params={'instType': inst_type})

    def tickers(self, inst_type='SWAP'):
        return self.request('GET', '/api/v5/market/tickers',
                            params={'instType': inst_type})

    def history_candles(self, inst_id, *, bar='4H', after=None, limit=100):
        params = {'instId': inst_id, 'bar': bar, 'limit': str(min(int(limit), 300))}
        if after is not None:
            params['after'] = str(after)
        return self.request('GET', '/api/v5/market/history-candles', params=params)

    def candle_history(self, inst_id, *, bar='4H', start_ms=None, end_ms=None,
                       max_pages=100, pause_seconds=.11):
        """Page backwards through confirmed candles and return oldest first."""
        rows = []
        after = end_ms
        seen = set()
        for _ in range(max_pages):
            page = self.history_candles(inst_id, bar=bar, after=after, limit=300)
            page = [row for row in page if len(row) >= 9 and str(row[8]) == '1']
            fresh = [row for row in page if int(row[0]) not in seen]
            if not fresh:
                break
            rows.extend(fresh)
            seen.update(int(row[0]) for row in fresh)
            oldest = min(int(row[0]) for row in fresh)
            if start_ms is not None and oldest <= int(start_ms):
                break
            after = oldest
            if pause_seconds:
                time.sleep(pause_seconds)
        rows = [row for row in rows
                if (start_ms is None or int(row[0]) >= int(start_ms))
                and (end_ms is None or int(row[0]) <= int(end_ms))]
        return sorted(rows, key=lambda row: int(row[0]))

    def funding_rate_history(self, inst_id, *, start_ms=None, end_ms=None,
                             max_pages=100, pause_seconds=.11):
        """Page backwards through realized perpetual-swap funding rates."""
        rows = []
        after = end_ms
        seen = set()
        for _ in range(max_pages):
            params = {'instId': inst_id, 'limit': '100'}
            if after is not None:
                params['after'] = str(after)
            page = self.request('GET', '/api/v5/public/funding-rate-history', params=params)
            fresh = [row for row in page if int(row['fundingTime']) not in seen]
            if not fresh:
                break
            rows.extend(fresh)
            seen.update(int(row['fundingTime']) for row in fresh)
            oldest = min(int(row['fundingTime']) for row in fresh)
            if start_ms is not None and oldest <= int(start_ms):
                break
            after = oldest
            if pause_seconds:
                time.sleep(pause_seconds)
        return sorted((row for row in rows
                       if (start_ms is None or int(row['fundingTime']) >= int(start_ms))
                       and (end_ms is None or int(row['fundingTime']) <= int(end_ms))),
                      key=lambda row: int(row['fundingTime']))

    def trading_statistics_support_coins(self):
        """Return currencies supported by OKX public trading-statistics endpoints."""
        rows = self.request('GET', '/api/v5/rubik/stat/trading-data/support-coin')
        return rows if isinstance(rows, dict) else (rows[0] if rows else {})

    def trading_statistics(self, metric, *, ccy, period='1D'):
        """Read a current public Rubik trading-statistics history window.

        OKX currently accepts 5m, 1H and 1D. These endpoints expose a bounded
        recent window, so callers must persist snapshots instead of treating the
        current response as a complete historical dataset.
        """
        paths = {
            'long_short_account_ratio': (
                '/api/v5/rubik/stat/contracts/long-short-account-ratio', {}),
            'open_interest_volume': (
                '/api/v5/rubik/stat/contracts/open-interest-volume', {}),
            'taker_volume': (
                '/api/v5/rubik/stat/taker-volume', {'instType': 'CONTRACTS'}),
        }
        if metric not in paths:
            raise ValueError(f'unsupported OKX trading-statistics metric: {metric}')
        if period not in {'5m', '1H', '1D'}:
            raise ValueError('OKX trading-statistics period must be 5m, 1H or 1D')
        path, extra = paths[metric]
        return self.request('GET', path, params={'ccy': ccy, 'period': period, **extra})

    def book(self, inst_id='BTC-USDT', depth=100):
        rows = self.request('GET', '/api/v5/market/books',
                            params={'instId': inst_id, 'sz': str(depth)})
        if not rows:
            raise OKXAPIError('empty', f'Order book not found: {inst_id}')
        return rows[0]

    def place_order(self, *, inst_id, side, order_type, size, price=None,
                    client_order_id=None, target_currency=None, trade_mode='cash',
                    reduce_only=False):
        payload = {'instId': inst_id, 'tdMode': trade_mode, 'side': side,
                   'ordType': order_type, 'sz': str(size)}
        if price is not None:
            payload['px'] = str(price)
        if client_order_id:
            payload['clOrdId'] = client_order_id
        if target_currency:
            payload['tgtCcy'] = target_currency
        if reduce_only:
            payload['reduceOnly'] = True
        rows = self.request('POST', '/api/v5/trade/order', payload=payload, private=True)
        if not rows or str(rows[0].get('sCode', '0')) != '0':
            row = rows[0] if rows else {}
            raise OKXAPIError(row.get('sCode', 'empty'), row.get('sMsg') or 'order rejected')
        return rows[0]

    def cancel_order(self, *, inst_id, order_id):
        rows = self.request('POST', '/api/v5/trade/cancel-order',
                            payload={'instId': inst_id, 'ordId': order_id}, private=True)
        if not rows or str(rows[0].get('sCode', '0')) != '0':
            row = rows[0] if rows else {}
            raise OKXAPIError(row.get('sCode', 'empty'), row.get('sMsg') or 'cancel rejected')
        return rows[0]

    def order(self, *, inst_id, order_id):
        rows = self.request('GET', '/api/v5/trade/order',
                            params={'instId': inst_id, 'ordId': order_id}, private=True)
        if not rows:
            raise OKXAPIError('empty', f'Order not found: {order_id}')
        return rows[0]


def demo_order_smoke(client, inst_id='BTC-USDT'):
    """Place a non-marketable minimum demo order and cancel it immediately."""
    instrument = client.instruments(inst_id)
    book = client.book(inst_id, depth=1)
    best_bid = Decimal(book['bids'][0][0])
    tick = Decimal(instrument['tickSz'])
    lot = Decimal(instrument['lotSz'])
    minimum = Decimal(instrument['minSz'])
    price = ((best_bid*Decimal('.5'))/tick).to_integral_value(rounding=ROUND_DOWN)*tick
    notional_floor_size = ((Decimal(2)/price)/lot).to_integral_value(rounding=ROUND_UP)*lot
    size = max(lot, minimum, notional_floor_size)
    client_id = 'alphagymdemo'+datetime.now(UTC).strftime('%m%d%H%M%S%f')[:18]
    placed = client.place_order(
        inst_id=inst_id, side='buy', order_type='post_only', size=size,
        price=price, client_order_id=client_id)
    order_id = placed['ordId']
    before = None
    try:
        before = client.order(inst_id=inst_id, order_id=order_id)
    finally:
        cancelled = client.cancel_order(inst_id=inst_id, order_id=order_id)
    after = client.order(inst_id=inst_id, order_id=order_id)
    return {
        'ok': True, 'environment': 'demo', 'instrument': inst_id,
        'size': str(size), 'price': str(price), 'best_bid_at_submit': str(best_bid),
        'state_before_cancel': before.get('state') if before else None,
        'cancel_accepted': str(cancelled.get('sCode', '0')) == '0',
        'state_after_cancel': after.get('state'),
    }
