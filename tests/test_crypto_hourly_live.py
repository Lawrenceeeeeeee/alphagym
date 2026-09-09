import json
from decimal import Decimal

import pandas as pd

from alphagym.crypto_hourly_live import _close_filled_legs, _rounded_contracts


def test_contract_size_rounds_up_to_lot_and_minimum():
    instrument = {'lotSz': '.01', 'minSz': '.01', 'ctVal': '.01'}
    assert _rounded_contracts(instrument, Decimal(80000), 25) == Decimal('.04')
    assert _rounded_contracts(instrument, Decimal(80000), 1) == Decimal('.01')


def test_close_only_touches_instruments_and_sizes_owned_by_session():
    class Client:
        def positions(self):
            return [{'instId': 'BTC-USDT-SWAP', 'pos': '3'},
                    {'instId': 'ETH-USDT-SWAP', 'pos': '9'}]

        def instruments(self, _instrument):
            return {'lotSz': '1'}

    calls = []

    def market(_client, instrument, side, size, reduce_only=False):
        calls.append((instrument, side, size, reduce_only))
        return {'filled_size': str(size)}

    from unittest.mock import patch
    fills = [{'instrument': 'BTC-USDT-SWAP', 'direction': 'long', 'filled_size': '2'}]
    with patch('alphagym.crypto_hourly_live._market', market):
        _close_filled_legs(Client(), fills)
    assert calls == [('BTC-USDT-SWAP', 'sell', Decimal(2), True)]


def test_crypto_web_reads_stored_session_without_trading(tmp_path):
    from fastapi.testclient import TestClient

    from alphagym.factor_web import create_app
    from alphagym.storage import ClickHouseStore

    session_id = 'cryptopaper-' + 'a' * 16
    prefix = f'factor_library/crypto_live/{session_id}'
    status = {
        'ok': True, 'session_id': session_id, 'status': 'closed',
        'environment': 'demo', 'report_id': 'crypto-' + 'b' * 16,
        'factor': 'momentum_1', 'bar': '4H', 'holding_bars': 6,
        'started_at': '2026-09-09T00:00:00+00:00',
        'close_due_at': '2026-09-10T00:00:00+00:00',
        'notional_per_leg': 25, 'legs': 1, 'filled_legs': 1,
    }
    database = ClickHouseStore(tmp_path, initialize=True)
    database.write_blob(f'{prefix}/status.json', json.dumps(status).encode())
    database.write_frame(f'{prefix}/plan.parquet', pd.DataFrame([{
        'instrument': 'BTC-USDT-SWAP', 'direction': 'long'}]))
    client = TestClient(create_app(tmp_path))
    assert session_id in client.get('/crypto').text
    assert client.get(f'/crypto/live/{session_id}').status_code == 200
    payload = client.get(f'/api/crypto/live/{session_id}').json()
    assert payload['status']['status'] == 'closed'
    assert payload['positions'] == []
