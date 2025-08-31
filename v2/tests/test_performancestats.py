# tests/test_performance_stats.py

import pandas as pd
import numpy as np
from backtest.PerformanceStats import PerformanceStats


def test_performance_metrics_on_known_series():
    pnl = pd.Series([100, 102, 101, 105, 107])

    stats = PerformanceStats.compute(pnl)

    assert isinstance(stats, dict)
    assert stats["total_return"] == 7  # 107 - 100
    assert stats["max_drawdown"] < 0   # there is a drop from 102 to 101
    assert stats["sharpe_ratio"] > 0
    assert stats["return_std"] > 0
    assert stats["avg_daily_return"] > 0


def test_max_drawdown_correctness():
    # Simulate a 10% drop
    pnl = pd.Series([100, 120, 108])  # drop from peak (120) to 108 = -10%
    stats = PerformanceStats.compute(pnl)

    assert np.isclose(stats["max_drawdown"], -0.10, atol=1e-4)


def test_empty_pnl_series():
    pnl = pd.Series(dtype=float)
    stats = PerformanceStats.compute(pnl)

    assert stats["total_return"] == 0.0
    assert stats["sharpe_ratio"] == 0.0
    assert np.isnan(stats["max_drawdown"]) or stats["max_drawdown"] == 0.0


def test_flat_pnl_series():
    pnl = pd.Series([100, 100, 100, 100])
    stats = PerformanceStats.compute(pnl)

    assert stats["total_return"] == 0.0
    assert stats["sharpe_ratio"] == 0.0
    assert stats["max_drawdown"] == 0.0
    assert stats["return_std"] == 0.0
    assert stats["avg_daily_return"] == 0.0
