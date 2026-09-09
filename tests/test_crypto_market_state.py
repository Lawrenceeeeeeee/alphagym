from __future__ import annotations

import pandas as pd

from alphagym.crypto_market_state import compute_market_states, fetch_trading_statistics
from alphagym.okx_api import OKXDemoClient


class TradingStatisticsClient:
    def trading_statistics_support_coins(self):
        return {"contract": ["BTC", "ETH"]}

    def trading_statistics(self, metric, *, ccy, period):
        assert period == "1D"
        base = 100 if ccy == "BTC" else 80
        rows = []
        for number, ts in enumerate(pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC")):
            stamp = str(int(ts.timestamp() * 1000))
            if metric == "long_short_account_ratio":
                rows.append([stamp, str(1 + number / 100)])
            elif metric == "open_interest_volume":
                rows.append([stamp, str(base + number), str(20 + number)])
            else:
                rows.append([stamp, str(10 + number / 2), str(12 + number)])
        return rows


def test_trading_statistics_are_normalized_and_market_states_are_causal_inputs():
    raw = fetch_trading_statistics(client=TradingStatisticsClient())
    assert len(raw) == 80
    assert {"open_interest_usd", "taker_buy_volume_usd"} <= set(raw)
    states = compute_market_states(raw)
    assert len(states) == 40
    assert {"risk_on", "crowded_long", "neutral"} & set(states["market_state"])
    assert states["taker_imbalance"].dropna().gt(0).all()


def test_okx_client_validates_trading_statistics_metric_and_period():
    client = OKXDemoClient()
    try:
        client.trading_statistics("unknown", ccy="BTC")
    except ValueError as error:
        assert "unsupported" in str(error)
    else:
        raise AssertionError("unknown metric should fail before network access")
    try:
        client.trading_statistics("taker_volume", ccy="BTC", period="4H")
    except ValueError as error:
        assert "5m, 1H or 1D" in str(error)
    else:
        raise AssertionError("unsupported period should fail before network access")
