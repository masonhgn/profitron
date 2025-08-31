# tests/test_order_manager.py

import pytest
import time
from execution.OrderManager import OrderManager
from signals.Signal import Signal


@pytest.fixture
def mock_client():
    class MockClient:
        def __init__(self):
            self.cancelled_orders = []
            self.market_orders = []
            self.created_orders = []
            self.order_counter = 0

        def create_limit_order(self, symbol, side, amount, price):
            self.order_counter += 1
            order_id = f"order_{self.order_counter}"
            self.created_orders.append((symbol, side, amount, price))
            return {"id": order_id}

        def fetch_ticker(self, symbol):
            return {"last": 200.0}

        def cancel_order(self, order_id):
            self.cancelled_orders.append(order_id)

        def create_market_order(self, symbol, side, amount):
            self.market_orders.append((symbol, side, amount))

    return MockClient()


@pytest.fixture
def signal():
    return Signal(
        symbol="ETH-USDT",
        action="buy",
        quantity=0.5,
        order_type="limit",
        meta={"limit_price": 1800}
    )


def test_submit_order(mock_client, signal):
    manager = OrderManager(client=mock_client, timeout=10)
    order_id = manager.submit_order(signal)

    assert order_id is not None
    assert order_id in manager.open_orders
    assert mock_client.created_orders[-1] == ("ETH/USDT", "buy", 0.5, 1800)


def test_infer_quantity(mock_client):
    manager = OrderManager(client=mock_client)
    signal = Signal(symbol="BTC-USDT", action="buy", target_pct=0.2)
    quantity = manager._infer_quantity(signal)

    assert quantity == pytest.approx(0.2 * 10000 / 200.0)


def test_cancel_stale_order(mock_client, signal):
    manager = OrderManager(client=mock_client, timeout=0)  # auto-expire
    order_id = manager.submit_order(signal)

    # simulate time passage
    manager.open_orders[order_id]["timestamp"] -= 60
    manager.check_and_cancel_stale_orders()

    assert order_id in mock_client.cancelled_orders
    assert manager.open_orders[order_id]["status"] == "cancelled"


def test_replace_with_market(mock_client):
    manager = OrderManager(client=mock_client)
    signal = Signal(
        symbol="ETH-USDT",
        action="buy",
        quantity=0.25,
        order_type="limit",
        meta={"limit_price": 1800, "replace_with_market": True}
    )

    order_id = manager.submit_order(signal)
    manager.open_orders[order_id]["timestamp"] -= 60  # force expiration
    manager.check_and_cancel_stale_orders()

    assert mock_client.market_orders[-1] == ("ETH/USDT", "buy", 0.25)


def test_submit_order_handles_exception():
    class FailingClient:
        def create_limit_order(self, *args, **kwargs):
            raise Exception("boom")

    manager = OrderManager(client=FailingClient())
    signal = Signal(symbol="ETH-USDT", action="buy", quantity=1.0)

    order_id = manager.submit_order(signal)
    assert order_id is None
