# tests/test_signal.py

import pytest
from signals.Signal import Signal


def test_signal_required_fields():
    sig = Signal(symbol="BTC-USDT", action="buy")
    assert sig.symbol == "BTC-USDT"
    assert sig.action == "buy"


def test_signal_defaults():
    sig = Signal(symbol="ETH-USDT", action="sell")
    assert sig.asset_type == "crypto"
    assert sig.order_type == "market"
    assert sig.target_pct is None
    assert sig.quantity is None
    assert isinstance(sig.meta, dict)
    assert sig.meta == {}


def test_signal_with_quantity_and_target_pct():
    sig = Signal(
        symbol="AAPL",
        action="buy",
        asset_type="equity",
        quantity=10,
        target_pct=0.1,
        order_type="limit",
        meta={"limit_price": 182.5}
    )
    assert sig.symbol == "AAPL"
    assert sig.action == "buy"
    assert sig.asset_type == "equity"
    assert sig.quantity == 10
    assert sig.target_pct == 0.1
    assert sig.order_type == "limit"
    assert sig.meta["limit_price"] == 182.5


def test_signal_meta_field_is_mutable():
    sig = Signal(symbol="BTC-USDT", action="buy")
    sig.meta["custom_field"] = 123
    assert sig.meta["custom_field"] == 123


def test_signal_invalid_action_should_pass():  # no validation yet
    sig = Signal(symbol="DOGE-USDT", action="moon")
    assert sig.action == "moon"  # No validation = accepts any string


