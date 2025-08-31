# tests/test_meta_portfolio.py

import pytest
from portfolio.MetaPortfolio import MetaPortfolio
from signals.Signal import Signal


@pytest.fixture
def mock_brokers():
    class MockPortfolio:
        def __init__(self, equity, exposures):
            self.equity = equity
            self.exposures = exposures

        def get_total_equity(self, prices):
            return self.equity

        def get_exposure(self, symbol, price):
            return self.exposures.get(symbol, 0.0)

    class MockBroker:
        def __init__(self, equity, exposures):
            self.portfolio = MockPortfolio(equity, exposures)
            self.synced = False
            self.orders = []

        def sync(self):
            self.synced = True

        def place_order(self, signal):
            self.orders.append(signal)

    return {
        "binanceus": MockBroker(12000, {"BTC-USDT": 5000}),
        "alpaca": MockBroker(8000, {"AAPL": 3000})
    }


def test_register_and_sync_all(mock_brokers):
    meta = MetaPortfolio()
    for name, broker in mock_brokers.items():
        meta.register_broker(name, broker)

    meta.sync_all()
    assert all(b.synced for b in mock_brokers.values())


def test_place_order_routing_crypto(mock_brokers):
    meta = MetaPortfolio()
    meta.brokers = mock_brokers  # directly inject mocks

    signal = Signal(symbol="BTC-USDT", action="buy", asset_type="crypto")
    meta.place_order(signal)

    assert signal in mock_brokers["binanceus"].orders


def test_place_order_routing_equity(mock_brokers):
    meta = MetaPortfolio()
    meta.brokers = mock_brokers

    signal = Signal(symbol="AAPL", action="sell", asset_type="equity")
    meta.place_order(signal)

    assert signal in mock_brokers["alpaca"].orders


def test_place_order_unknown_asset_type(capfd, mock_brokers):
    meta = MetaPortfolio()
    meta.brokers = mock_brokers

    signal = Signal(symbol="XYZ-OPTION", action="buy", asset_type="futures")
    meta.place_order(signal)

    out, _ = capfd.readouterr()
    assert "[MetaPortfolio] no broker found for signal" in out


def test_total_equity_and_exposure(mock_brokers):
    meta = MetaPortfolio()
    meta.brokers = mock_brokers

    prices = {"BTC-USDT": 30000, "AAPL": 180}

    total_equity = meta.get_total_equity(prices)
    assert total_equity == 12000 + 8000

    btc_exp = meta.get_exposure("BTC-USDT", 30000)
    assert btc_exp == 5000

    summary = meta.summary(prices)
    assert summary["total_equity"] == 20000
    assert summary["exposures"]["BTC-USDT"] == 5000
    assert summary["exposures"]["AAPL"] == 3000
