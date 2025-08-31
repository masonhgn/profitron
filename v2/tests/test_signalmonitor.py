# tests/test_signal_monitor.py

import pytest
import pandas as pd
from signals.SignalMonitor import SignalMonitor
from signals.Signal import Signal


@pytest.fixture
def dummy_strategy():
    class DummyStrategy:
        def generate_signals(self, data: pd.DataFrame):
            return [
                Signal(symbol="BTC-USDT", action="buy", target_pct=0.1)
            ]
    return DummyStrategy()


@pytest.fixture
def dummy_data_manager():
    class DummyDataManager:
        def get_price_data(self, assets, start_date, end_date, frequency, fields):
            index = pd.date_range("2023-01-01", periods=100, freq="h")
            return pd.DataFrame({
                "BTC-USDT": [40000 + i for i in range(100)]
            }, index=index)
    return DummyDataManager()


@pytest.fixture
def monitor_config():
    return {
        "frequency": "1h",
        "fields": ["close"],
        "lookback_window": 100,
        "poll_interval": 1
    }


def test_run_once_returns_signals(dummy_strategy, dummy_data_manager, monitor_config):
    monitor = SignalMonitor(dummy_strategy, dummy_data_manager, monitor_config)
    signals = monitor.run_once()

    assert isinstance(signals, list)
    assert len(signals) == 1
    assert isinstance(signals[0], Signal)
    assert signals[0].symbol == "BTC-USDT"
    assert signals[0].action == "buy"
    assert signals[0].target_pct == 0.1


def test_handles_short_data(dummy_strategy, monitor_config):
    class ShortDataManager:
        def get_price_data(self, *args, **kwargs):
            # Not enough rows to satisfy lookback
            return pd.DataFrame({
                "BTC-USDT": [40000, 40100]
            }, index=pd.date_range("2023-01-01", periods=2, freq="h"))

    monitor = SignalMonitor(dummy_strategy, ShortDataManager(), monitor_config)
    signals = monitor.run_once()

    assert isinstance(signals, list)
    # Dummy strategy doesn’t care about data length, but this test sets up for future guards
