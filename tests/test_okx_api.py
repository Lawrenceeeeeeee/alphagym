from __future__ import annotations

import json
from decimal import Decimal

from alphagym.hf_live import _closable_size, pair_round_trips
from alphagym.okx_api import OKXCredentials, OKXDemoClient, demo_order_smoke


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


def test_private_request_is_always_demo_and_signed(monkeypatch):
    captured = {}

    def open_(request, timeout):
        captured['request'] = request
        captured['timeout'] = timeout
        return Response({'code': '0', 'data': [{'acctLv': '1'}]})

    monkeypatch.setattr('urllib.request.urlopen', open_)
    client = OKXDemoClient(OKXCredentials('key', 'secret', 'pass'))
    assert client.account_config()['acctLv'] == '1'
    headers = dict(captured['request'].header_items())
    assert headers['X-simulated-trading'] == '1'
    assert headers['Ok-access-key'] == 'key'
    assert headers['Ok-access-sign']
    assert 'secret' not in str(headers)


def test_public_book_does_not_send_credentials(monkeypatch):
    captured = {}

    def open_(request, timeout):
        captured['request'] = request
        return Response({'code': '0', 'data': [{'bids': [], 'asks': []}]})

    monkeypatch.setattr('urllib.request.urlopen', open_)
    OKXDemoClient().book()
    headers = dict(captured['request'].header_items())
    assert 'Ok-access-key' not in headers
    assert 'X-simulated-trading' not in headers


def test_trading_statistics_request_is_public_and_uses_contract_scope(monkeypatch):
    captured = {}

    def open_(request, timeout):
        captured['request'] = request
        return Response({'code': '0', 'data': [['1', '2', '3']]})

    monkeypatch.setattr('urllib.request.urlopen', open_)
    rows = OKXDemoClient().trading_statistics(
        'taker_volume', ccy='BTC', period='1H')
    assert rows == [['1', '2', '3']]
    url = captured['request'].full_url
    assert '/api/v5/rubik/stat/taker-volume?' in url
    assert 'ccy=BTC' in url and 'period=1H' in url and 'instType=CONTRACTS' in url
    assert 'Ok-access-key' not in dict(captured['request'].header_items())


def test_trading_statistics_support_coins_accepts_object_payload(monkeypatch):
    def open_(_request, timeout):
        assert timeout == 10
        return Response({'code': '0', 'data': {'contract': ['BTC'], 'spot': ['BTC']}})

    monkeypatch.setattr('urllib.request.urlopen', open_)
    assert OKXDemoClient().trading_statistics_support_coins()['contract'] == ['BTC']


def test_demo_order_smoke_places_non_marketable_order_and_cancels():
    class Client:
        cancelled = False

        def instruments(self, _inst_id):
            return {'tickSz': '.1', 'lotSz': '.00000001', 'minSz': '.00001'}

        def book(self, _inst_id, depth):
            assert depth == 1
            return {'bids': [['80000', '1']], 'asks': [['80001', '1']]}

        def place_order(self, **order):
            self.placed = order
            return {'ordId': 'demo-order'}

        def order(self, **_query):
            return {'state': 'canceled' if self.cancelled else 'live'}

        def cancel_order(self, **_query):
            self.cancelled = True
            return {'sCode': '0'}

    client = Client()
    result = demo_order_smoke(client)
    assert result['state_before_cancel'] == 'live'
    assert result['state_after_cancel'] == 'canceled'
    assert client.placed['order_type'] == 'post_only'
    assert float(client.placed['price']) < 80000
    assert float(client.placed['price'])*float(client.placed['size']) >= 2


def test_buy_fee_is_removed_from_live_close_size():
    fill = {'side': 'buy', 'filled_size': '0.0001',
            'fee': '-0.0000001', 'fee_currency': 'BTC'}
    assert _closable_size(fill, Decimal('.00000001')) == Decimal('.0000999')


def test_live_round_trip_reports_net_pnl_after_both_fees():
    trades = [
        {'ts': 1000, 'event': 'open', 'direction': 'short', 'filled_size': '.01',
         'average_price': '78340', 'fee': '-.003917', 'fee_currency': 'USDT'},
        {'ts': 6000, 'event': 'close', 'direction': 'short', 'filled_size': '.01',
         'average_price': '78349.8', 'fee': '-.00391749', 'fee_currency': 'USDT'},
    ]
    row = pair_round_trips(trades)[0]
    assert row['holding_seconds'] == 5
    assert row['gross_pnl_usdt'] == -0.00098
    assert row['fees_usdt'] == 0.00783449
    assert row['net_pnl_usdt'] == -0.00881449
    assert round(row['net_return_bps'], 3) == -11.252
