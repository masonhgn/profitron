# tests/test_backtester.py

import pytest
import pandas as pd
from backtest.Backtester import Backtester, BacktestConfig, combine_signals
from signals.Signal import Signal


@pytest.fixture
def price_data():
    return pd.DataFrame({
        "A": [100, 102, 101, 103],
        "B": [200, 202, 204, 206]
    }, index=pd.date_range("2023-01-01", periods=4))


@pytest.fixture
def backtest_config():
    return BacktestConfig(
        start_date="2023-01-01",
        end_date="2023-01-04",
        capital=10000,
        slippage_bps=5,
        commission_per_trade=1,
        rebalance_frequency="daily"
    )


@pytest.fixture
def dummy_strategy():
    class DummyStrategy:
        def generate_signals(self, data):
            # Only generate signal on the last day
            if len(data) < 4:
                return []
            return [
                Signal(symbol="A", action="buy", target_pct=0.5),
                Signal(symbol="B", action="sell", target_pct=0.5)
            ]
    return DummyStrategy()


def test_backtester_runs(price_data, backtest_config, dummy_strategy):
    bt = Backtester(strategies=[dummy_strategy], price_data=price_data, config=backtest_config)
    result = bt.run()

    assert "pnl" in result
    assert "positions" in result
    assert "stats" in result

    pnl = result["pnl"]
    assert isinstance(pnl, pd.Series)

    expected_index = price_data.index[1:]  # skip first date
    assert pnl.index.equals(expected_index)
    assert not pnl.isnull().all()

    positions = result["positions"]
    assert isinstance(positions, pd.DataFrame)
    assert set(positions.columns) == {"A", "B"}


def test_combine_signals_additive_logic():
    signals = [
        Signal(symbol="A", action="buy", target_pct=0.2),
        Signal(symbol="A", action="buy", target_pct=0.3),
        Signal(symbol="B", action="sell", target_pct=0.4),
        Signal(symbol="B", action="buy", target_pct=0.2)
    ]
    weights = combine_signals(signals, all_assets=["A", "B", "C"])
    assert weights["A"] == pytest.approx(0.5)
    assert weights["B"] == pytest.approx(-0.2)
    assert weights["C"] == 0.0
