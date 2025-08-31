# tests/conftest.py

import pytest
import pandas as pd
import numpy as np
from signals.Signal import Signal


@pytest.fixture
def sample_prices():
    return pd.Series({
        "AAPL": 150.0,
        "MSFT": 250.0,
        "GOOG": 1000.0
    })


@pytest.fixture
def price_data():
    return pd.DataFrame({
        "AAPL": np.linspace(150, 160, 5),
        "MSFT": np.linspace(250, 260, 5),
        "GOOG": np.linspace(1000, 1020, 5)
    }, index=pd.date_range("2023-01-01", periods=5))


@pytest.fixture
def signal_buy():
    return Signal(symbol="AAPL", action="buy", target_pct=0.3, asset_type="equity")


@pytest.fixture
def signal_sell():
    return Signal(symbol="MSFT", action="sell", target_pct=0.2, asset_type="equity")


@pytest.fixture
def dummy_signal_list(signal_buy, signal_sell):
    return [signal_buy, signal_sell]


@pytest.fixture
def dummy_strategy():
    class DummyStrategy:
        def generate_signals(self, data: pd.DataFrame):
            return [
                Signal(symbol="AAPL", action="buy", target_pct=0.1),
                Signal(symbol="MSFT", action="sell", target_pct=0.05)
            ]
    return DummyStrategy()


@pytest.fixture
def dummy_data_manager():
    class DummyDataManager:
        def get_price_data(self, source_name, symbols, start_date, end_date, frequency, fields):
            return pd.DataFrame({
                "AAPL": np.linspace(150, 155, 100)
            }, index=pd.date_range("2023-01-01", periods=100, freq="H"))
    return DummyDataManager()
