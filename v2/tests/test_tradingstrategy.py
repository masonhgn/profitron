# tests/test_trading_strategy.py

import pytest
import pandas as pd
from strategy.TradingStrategy import TradingStrategy
from signals.Signal import Signal


def test_cannot_instantiate_base_class():
    with pytest.raises(TypeError):
        TradingStrategy()


def test_subclass_must_implement_generate_signals():
    class IncompleteStrategy(TradingStrategy):
        pass

    with pytest.raises(TypeError):
        IncompleteStrategy()


def test_dummy_strategy_returns_signals():
    class DummyStrategy(TradingStrategy):
        def generate_signals(self, data: pd.DataFrame):
            return [
                Signal(symbol="BTC-USDT", action="buy", target_pct=0.1),
                Signal(symbol="ETH-USDT", action="sell", target_pct=0.05)
            ]

    strat = DummyStrategy()
    df = pd.DataFrame()  # no-op input
    signals = strat.generate_signals(df)

    assert isinstance(signals, list)
    assert all(isinstance(s, Signal) for s in signals)
    assert signals[0].symbol == "BTC-USDT"
    assert signals[1].action == "sell"
