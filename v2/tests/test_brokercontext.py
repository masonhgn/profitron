# tests/test_broker_context.py

import pytest
from brokers.BrokerContext import BrokerContext
from signals.Signal import Signal


@pytest.fixture
def paper_config():
    return {
        "mode": "paper",
        "starting_cash": 5000
    }


@pytest.fixture
def live_config():
    return {
        "mode": "live",
        "starting_cash": 10000,
        "default_trade_size": 100,
        "order_timeout": 10
    }


@pytest.fixture
def signal_limit():
    return Signal(
        symbol="ETH-USDT",
        action="buy",
        quantity=0.5,
        order_type="limit",
        meta={"limit_price": 1800}
    )


@pytest.fixture
def signal_market():
    return Signal(
        symbol="BTC-USDT",
        action="sell",
        target_pct=0.1,
        order_type="market"
    )


def test_paper_mode_prints_order(capfd, paper_config, signal_limit):
    ctx = BrokerContext("binanceus", paper_config)
    ctx.place_order(signal_limit)

    out, _ = capfd.readouterr()
    assert "[PAPER] BUY ETH-USDT x 0.5" in out


def test_market_order_executes_with_mock_api(monkeypatch, live_config, signal_market):
    mock_api = type("MockAPI", (), {
        "create_market_order": lambda self, symbol, side, amount: print(f"MARKET ORDER: {symbol} {side} x{amount}"),
        "fetch_ticker": lambda self, symbol: {"last": 200}
    })()

    monkeypatch.setattr("brokers.BrokerContext.ccxt.binanceus", lambda *args, **kwargs: mock_api)

    ctx = BrokerContext("binanceus", live_config)
    ctx.api = mock_api  # override for test safety
    ctx.place_order(signal_market)

    # No exception = pass (output tested manually or optionally captured)


def test_sync_populates_portfolio(monkeypatch, live_config):
    class MockAPI:
        def fetch_balance(self):
            return {
                "total": {"BTC": 0.1, "ETH": 2.0},
                "USDT": {"free": 1000}
            }

    monkeypatch.setattr("brokers.BrokerContext.ccxt.binanceus", lambda *args, **kwargs: MockAPI())

    ctx = BrokerContext("binanceus", live_config)
    ctx.api = MockAPI()
    ctx.sync()

    assert ctx.portfolio.cash == 1000
    assert ctx.portfolio.positions["BTC-USDT"] == 0.1
    assert ctx.portfolio.positions["ETH-USDT"] == 2.0


def test_invalid_broker_raises():
    with pytest.raises(ValueError):
        BrokerContext("unsupported_broker", {"mode": "live"})
